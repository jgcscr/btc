#!/usr/bin/env python3
"""
ARIMA+GARCH Forecasting Pipeline for BTC Price Prediction.

Features:
- Manual ARIMA (p,q) grid search without pmdarima dependency
- Student-t distribution with fallback to normal distribution
- EGARCH fallback on convergence warnings
- Optional resampling (1m/5m/15m) for noise reduction
- Monte Carlo simulation for interval estimation
- Analytical log-normal approximation
- Horizon extraction for 4h/8h/12h forecasts

Usage:
    python forecasting/arima_garch_pipeline.py --config config/forecast_config.yaml

Author: BTC Forecasting System
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
import warnings
from typing import Tuple, Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from arch.univariate.volatility import GARCH, EGARCH
from forecasting.utils import setup_logging, resample_data, validate_data_structure
from forecasting.utils.logging_utils import log_data_summary, log_forecast_summary, ProgressLogger

logger = None


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_arima_data(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Prepare price series for ARIMA modeling.
    
    Args:
        df: Input DataFrame with OHLC data
        config: Configuration dictionary
        
    Returns:
        Price series ready for ARIMA modeling
    """
    arima_config = config['arima']
    
    # Optionally resample data to reduce microstructure noise
    resample_freq = arima_config.get('resample')
    if resample_freq:
        logger.info(f"Resampling data to {resample_freq}")
        df = resample_data(df, resample_freq, 'timestamp', 'close')
    
    # Extract price series
    price_series = df.set_index('timestamp')['close']
    
    # Calculate returns for GARCH modeling
    returns = price_series.pct_change().dropna() * 100  # Percentage returns
    
    logger.info(f"Prepared ARIMA data: {len(returns)} return observations")
    return returns


def grid_search_arima(returns: pd.Series, max_p: int, max_q: int) -> Tuple[int, int, float]:
    """
    Manual grid search for optimal ARIMA(p,0,q) parameters.
    
    Args:
        returns: Return series
        max_p: Maximum AR order to test
        max_q: Maximum MA order to test
        
    Returns:
        Tuple of (best_p, best_q, best_aic)
    """
    best_aic = np.inf
    best_p, best_q = 1, 1
    results = []
    
    logger.info(f"Starting ARIMA grid search: p=0-{max_p}, q=0-{max_q}")
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
                
            try:
                model = ARIMA(returns, order=(p, 0, q))
                fitted = model.fit()
                aic = fitted.aic
                
                results.append({'p': p, 'q': q, 'aic': aic})
                
                if aic < best_aic:
                    best_aic = aic
                    best_p, best_q = p, q
                    
                logger.debug(f"ARIMA({p},0,{q}): AIC = {aic:.2f}")
                
            except Exception as e:
                logger.debug(f"ARIMA({p},0,{q}) failed: {str(e)}")
                continue
    
    logger.info(f"Best ARIMA model: ({best_p},0,{best_q}) with AIC = {best_aic:.2f}")
    return best_p, best_q, best_aic


def fit_arima_garch(returns: pd.Series, p: int, q: int, config: dict) -> Tuple[Any, Any, Dict]:
    """
    Fit ARIMA+GARCH model with specified parameters.
    
    Args:
        returns: Return series
        p: ARIMA AR order
        q: ARIMA MA order
        config: Configuration dictionary
        
    Returns:
        Tuple of (arima_model, garch_model, model_info)
    """
    arima_config = config['arima']
    
    # Fit ARIMA model
    logger.info(f"Fitting ARIMA({p},0,{q}) model...")
    arima_model = ARIMA(returns, order=(p, 0, q))
    arima_fitted = arima_model.fit()
    
    # Get ARIMA residuals for GARCH
    residuals = arima_fitted.resid
    
    # Fit GARCH model
    garch_p = arima_config.get('garch_p', 1)
    garch_q = arima_config.get('garch_q', 1)
    dist = arima_config.get('dist', 'student_t')
    
    logger.info(f"Fitting GARCH({garch_p},{garch_q}) model with {dist} distribution...")
    
    # Try Student-t distribution first
    garch_model = None
    garch_fitted = None
    used_egarch = False
    
    try:
        garch_model = arch_model(
            residuals,
            vol='GARCH',
            p=garch_p,
            q=garch_q,
            dist=dist
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            garch_fitted = garch_model.fit(disp='off')
            
            # Check for convergence warnings
            convergence_warning = any(
                "convergence" in str(warning.message).lower() or
                "optimization" in str(warning.message).lower()
                for warning in w
            )
            
            if convergence_warning and arima_config.get('egarch_fallback', False):
                logger.warning("GARCH convergence warning detected, falling back to EGARCH")
                raise RuntimeError("Convergence warning triggered EGARCH fallback")
                
    except Exception as e:
        # Fallback to EGARCH if enabled
        if arima_config.get('egarch_fallback', False):
            logger.info("Attempting EGARCH fallback...")
            try:
                garch_model = arch_model(
                    residuals,
                    vol='EGARCH',
                    p=1,
                    q=1,
                    dist='normal'  # EGARCH with normal distribution
                )
                garch_fitted = garch_model.fit(disp='off')
                used_egarch = True
                logger.info("Successfully fitted EGARCH model")
            except Exception as egarch_error:
                logger.error(f"EGARCH fallback also failed: {str(egarch_error)}")
                raise
        else:
            logger.error(f"GARCH fitting failed: {str(e)}")
            raise
    
    # Model information
    model_info = {
        'arima_order': (p, 0, q),
        'garch_order': (garch_p, garch_q),
        'distribution': dist if not used_egarch else 'normal',
        'volatility_model': 'EGARCH' if used_egarch else 'GARCH',
        'arima_aic': arima_fitted.aic,
        'garch_loglikelihood': garch_fitted.loglikelihood if garch_fitted else None
    }
    
    logger.info(f"Model fitted successfully: {model_info}")
    return arima_fitted, garch_fitted, model_info


def generate_forecasts(arima_fitted: Any, garch_fitted: Any, horizons: list, 
                      config: dict, last_price: float) -> Dict[str, np.ndarray]:
    """
    Generate forecasts using fitted ARIMA+GARCH models.
    
    Args:
        arima_fitted: Fitted ARIMA model
        garch_fitted: Fitted GARCH model
        horizons: List of forecast horizons in minutes
        config: Configuration dictionary
        last_price: Last observed price for level conversion
        
    Returns:
        Dictionary with analytical and Monte Carlo forecasts
    """
    arima_config = config['arima']
    mc_paths = arima_config.get('monte_carlo_paths', 5000)
    max_horizon = max(horizons)
    
    # ARIMA forecasts (mean) - for multi-step ahead
    arima_forecast = arima_fitted.forecast(steps=max_horizon)
    
    # For GARCH multi-step forecasting, we'll use simulation
    # Get the last conditional variance and parameters
    last_variance = garch_fitted.conditional_volatility.iloc[-1] ** 2
    
    # Simulate variance paths for multi-step ahead forecasting
    logger.info(f"Running Monte Carlo simulation with {mc_paths} paths...")
    
    mc_results = []
    variance_paths = []
    
    for _ in range(mc_paths):
        # Simulate variance path using GARCH parameters
        variance_path = [last_variance]
        
        for step in range(max_horizon):
            # Get GARCH parameters (simplified approach)
            try:
                if 'alpha[1]' in garch_fitted.params:
                    alpha = garch_fitted.params['alpha[1]']
                    beta = garch_fitted.params['beta[1]'] if 'beta[1]' in garch_fitted.params else 0.8
                    omega = garch_fitted.params['omega'] if 'omega' in garch_fitted.params else 0.01
                else:
                    # Default GARCH parameters
                    alpha, beta, omega = 0.1, 0.8, 0.01
                
                # Simple GARCH(1,1) evolution
                if step == 0:
                    # Use last return for shock
                    last_return = arima_fitted.resid.iloc[-1] if hasattr(arima_fitted, 'resid') else 0
                    next_var = omega + alpha * (last_return ** 2) + beta * variance_path[-1]
                else:
                    # Use simulated shock
                    shock = np.random.normal(0, np.sqrt(variance_path[-1]))
                    next_var = omega + alpha * (shock ** 2) + beta * variance_path[-1]
            except:
                # Fallback to constant variance
                next_var = last_variance
                
            variance_path.append(max(next_var, 1e-6))  # Ensure positive variance
        
        variance_paths.append(variance_path[1:])  # Remove initial value
        
        # Generate return path using Monte Carlo
        return_shocks = []
        for step in range(max_horizon):
            if hasattr(garch_fitted.model, 'distribution') and garch_fitted.model.distribution.name == 'StudentsT':
                nu = garch_fitted.params['nu'] if 'nu' in garch_fitted.params else 5
                shock = np.random.standard_t(nu) * np.sqrt(variance_path[step + 1])
            else:
                shock = np.random.normal(0, np.sqrt(variance_path[step + 1]))
            return_shocks.append(shock)
        
        # Combine ARIMA forecast with GARCH shocks
        return_path = arima_forecast.values + np.array(return_shocks)
        
        # Convert to price path
        price_path = last_price * np.cumprod(1 + return_path / 100)
        mc_results.append(price_path)
    
    mc_paths_array = np.array(mc_results)
    variance_paths_array = np.array(variance_paths)
    
    # Calculate quantiles for prediction intervals
    mc_quantiles = {
        '05': np.percentile(mc_paths_array, 5, axis=0),
        '25': np.percentile(mc_paths_array, 25, axis=0),
        '50': np.percentile(mc_paths_array, 50, axis=0),
        '75': np.percentile(mc_paths_array, 75, axis=0),
        '95': np.percentile(mc_paths_array, 95, axis=0)
    }
    
    # Analytical approximation (log-normal) using average simulated variance
    analytical_mean = last_price * np.cumprod(1 + arima_forecast / 100)
    avg_variance = np.mean(variance_paths_array, axis=0)
    
    # For log-normal approximation of prediction intervals
    log_var = np.cumsum(avg_variance / 10000)  # Convert percentage variance
    analytical_lower = analytical_mean * np.exp(-1.645 * np.sqrt(log_var))  # 90% interval
    analytical_upper = analytical_mean * np.exp(1.645 * np.sqrt(log_var))
    
    return {
        'analytical_mean': analytical_mean,
        'analytical_lower': analytical_lower,
        'analytical_upper': analytical_upper,
        'mc_quantiles': mc_quantiles,
        'raw_return_forecast': arima_forecast.values,
        'conditional_variance': avg_variance
    }


def extract_horizon_forecasts(forecasts: Dict, horizons: list, last_timestamp: pd.Timestamp) -> pd.DataFrame:
    """
    Extract specific horizon forecasts from full forecast results.
    
    Args:
        forecasts: Dictionary with forecast results
        horizons: List of forecast horizons in minutes
        last_timestamp: Last timestamp from training data
        
    Returns:
        DataFrame with horizon-specific forecasts
    """
    horizon_data = []
    
    for horizon in horizons:
        idx = horizon - 1  # 0-based indexing
        
        if idx < len(forecasts['analytical_mean']):
            forecast_time = last_timestamp + timedelta(minutes=horizon)
            
            horizon_data.append({
                'timestamp': forecast_time,
                'horizon_minutes': horizon,
                'analytical_mean': forecasts['analytical_mean'].iloc[idx] if hasattr(forecasts['analytical_mean'], 'iloc') else forecasts['analytical_mean'][idx],
                'analytical_lower': forecasts['analytical_lower'].iloc[idx] if hasattr(forecasts['analytical_lower'], 'iloc') else forecasts['analytical_lower'][idx],
                'analytical_upper': forecasts['analytical_upper'].iloc[idx] if hasattr(forecasts['analytical_upper'], 'iloc') else forecasts['analytical_upper'][idx],
                'mc_p05': forecasts['mc_quantiles']['05'][idx],
                'mc_p25': forecasts['mc_quantiles']['25'][idx],
                'mc_p50': forecasts['mc_quantiles']['50'][idx],
                'mc_p75': forecasts['mc_quantiles']['75'][idx],
                'mc_p95': forecasts['mc_quantiles']['95'][idx],
                'return_forecast': forecasts['raw_return_forecast'][idx],
                'conditional_variance': forecasts['conditional_variance'][idx],
                'forecast_time': datetime.now()
            })
    
    return pd.DataFrame(horizon_data)


def save_outputs(horizon_df: pd.DataFrame, model_info: Dict, config: dict) -> str:
    """
    Save ARIMA+GARCH forecast outputs.
    
    Args:
        horizon_df: Horizon-specific forecasts
        model_info: Model information dictionary
        config: Configuration dictionary
        
    Returns:
        Path to saved file
    """
    output_dir = Path(config['paths']['arima_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add model info to DataFrame (scalar values only)
    for key, value in model_info.items():
        if not isinstance(value, (list, tuple, np.ndarray)):
            horizon_df[f'model_{key}'] = value
    
    # Save horizon forecasts
    output_path = output_dir / f"arima_garch_horizons_{timestamp}.parquet"
    horizon_df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved ARIMA+GARCH forecasts to: {output_path}")
    return str(output_path)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='ARIMA+GARCH Forecasting Pipeline')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    global logger
    logger = setup_logging(level=config.get('logging', {}).get('level', 'INFO'))
    
    with ProgressLogger(logger, "ARIMA+GARCH Forecasting Pipeline"):
        try:
            # Load data
            data_path = config['paths']['data_parquet']
            logger.info(f"Loading data from: {data_path}")
            
            df = pd.read_parquet(data_path)
            log_data_summary(logger, df, "Input Data")
            
            # Validate data
            validate_data_structure(df, ['timestamp', 'close'])
            
            # Prepare return series
            returns = prepare_arima_data(df, config)
            last_price = df['close'].iloc[-1]
            last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
            
            # Grid search for optimal ARIMA parameters
            arima_config = config['arima']
            best_p, best_q, best_aic = grid_search_arima(
                returns, 
                arima_config.get('max_p', 5),
                arima_config.get('max_q', 5)
            )
            
            # Fit ARIMA+GARCH model
            arima_fitted, garch_fitted, model_info = fit_arima_garch(
                returns, best_p, best_q, config
            )
            
            # Generate forecasts
            horizons = config['horizons']
            logger.info(f"Generating forecasts for horizons: {horizons}")
            
            forecasts = generate_forecasts(
                arima_fitted, garch_fitted, horizons, config, last_price
            )
            
            # Extract horizon forecasts
            horizon_df = extract_horizon_forecasts(forecasts, horizons, last_timestamp)
            
            # Log forecast summaries
            for horizon in horizons:
                horizon_subset = horizon_df[horizon_df['horizon_minutes'] == horizon]
                log_forecast_summary(logger, horizon, horizon_subset, "ARIMA+GARCH")
            
            # Save outputs
            logger.info("Saving forecast outputs...")
            output_path = save_outputs(horizon_df, model_info, config)
            
            logger.info("ARIMA+GARCH forecasting completed successfully!")
            
        except Exception as e:
            logger.error(f"ARIMA+GARCH forecasting failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()