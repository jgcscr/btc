#!/usr/bin/env python3
"""
Multi-Horizon Backtesting Framework for BTC Forecasting Models.

Features:
- Rolling window evaluation scheme
- Configurable train window and step size
- Support for Prophet and ARIMA+GARCH models
- Comprehensive metrics: RMSE, MAE, MAPE, sMAPE, coverage, interval width
- Per-cut and summary results
- Alignment of realized future prices

Usage:
    python forecasting/backtest_multi_horizon.py --config config/forecast_config.yaml

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
from typing import Dict, List, Tuple, Optional
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from forecasting.utils import setup_logging, validate_data_structure, handle_missing_ohlc
from forecasting.utils.eval_utils import calculate_all_metrics, calculate_coverage, calculate_interval_width
from forecasting.utils.logging_utils import log_data_summary, ProgressLogger

logger = None


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_backtest_cuts(df: pd.DataFrame, config: dict) -> List[Tuple[int, int]]:
    """
    Generate rolling window cuts for backtesting.
    
    Args:
        df: Input DataFrame with timestamp data
        config: Configuration dictionary
        
    Returns:
        List of (train_end_idx, cutoff_timestamp_idx) tuples
    """
    backtest_config = config['backtest']
    train_window_minutes = backtest_config['train_window_minutes']
    step_minutes = backtest_config['step_minutes']
    horizons = config['horizons']
    max_horizon = max(horizons)
    
    # Convert timestamps to datetime if needed
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    cuts = []
    
    # Start from first possible training window
    start_idx = 0
    min_train_size = 100  # Minimum training samples
    
    while start_idx + min_train_size < len(df):
        # Find training window end
        cutoff_time = df['timestamp'].iloc[start_idx] + timedelta(minutes=train_window_minutes)
        
        # Find index closest to cutoff time
        cutoff_idx = df[df['timestamp'] <= cutoff_time].index[-1] if len(df[df['timestamp'] <= cutoff_time]) > 0 else start_idx + min_train_size
        
        # Check if we have enough future data for evaluation
        future_needed_time = cutoff_time + timedelta(minutes=max_horizon)
        if df['timestamp'].iloc[-1] >= future_needed_time:
            cuts.append((start_idx, cutoff_idx))
        
        # Move to next cut
        start_idx = cutoff_idx + 1
        
        # Check if we should step by time instead of index
        next_cutoff_time = cutoff_time + timedelta(minutes=step_minutes)
        next_start_idx = df[df['timestamp'] >= next_cutoff_time].index
        if len(next_start_idx) > 0:
            start_idx = next_start_idx[0]
    
    logger.info(f"Generated {len(cuts)} backtest cuts")
    return cuts


def run_prophet_backtest(train_df: pd.DataFrame, config: dict, horizons: List[int]) -> Optional[pd.DataFrame]:
    """
    Run Prophet model for a single backtest cut.
    
    Args:
        train_df: Training data
        config: Configuration dictionary
        horizons: List of forecast horizons
        
    Returns:
        DataFrame with Prophet forecasts or None if failed
    """
    try:
        # Direct implementation instead of importing
        from prophet import Prophet
        import numpy as np
        
        # Prepare data for Prophet
        prophet_config = config['prophet']
        prophet_df = train_df[['timestamp', 'close']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Apply log transformation if specified
        if prophet_config.get('use_log_price', False):
            prophet_df['y'] = np.log(prophet_df['y'])
        
        # Add time features for regressors
        from forecasting.utils import add_time_features
        df_with_features = add_time_features(train_df, 'timestamp')
        prophet_df['hour_sin'] = df_with_features['hour_sin']
        prophet_df['hour_cos'] = df_with_features['hour_cos']
        prophet_df['minute_sin'] = df_with_features['minute_sin']
        prophet_df['minute_cos'] = df_with_features['minute_cos']
        
        # Create Prophet model
        model = Prophet(
            changepoint_prior_scale=prophet_config.get('changepoint_prior_scale', 0.2),
            changepoint_range=prophet_config.get('changepoint_range', 0.98),
            seasonality_mode=prophet_config.get('seasonality_mode', 'additive'),
            interval_width=prophet_config.get('interval_width', 0.9),
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        
        # Add seasonality and regressors
        intraday_fourier = prophet_config.get('intraday_fourier', 20)
        model.add_seasonality(name='intraday', period=1440, fourier_order=intraday_fourier)
        
        model.add_regressor('hour_sin')
        model.add_regressor('hour_cos')
        model.add_regressor('minute_sin')
        model.add_regressor('minute_cos')
        
        # Fit model
        model.fit(prophet_df)
        
        # Generate forecast
        max_horizon = max(horizons)
        future = model.make_future_dataframe(periods=max_horizon, freq='T')
        
        # Add time features to future dates
        future_with_features = add_time_features(future, 'ds')
        future['hour_sin'] = future_with_features['hour_sin']
        future['hour_cos'] = future_with_features['hour_cos']
        future['minute_sin'] = future_with_features['minute_sin']
        future['minute_cos'] = future_with_features['minute_cos']
        
        forecast = model.predict(future)
        
        # Extract horizon forecasts
        use_log_price = prophet_config.get('use_log_price', False)
        horizon_data = []
        
        for horizon in horizons:
            horizon_idx = len(forecast) - max(horizons) + horizon - 1
            
            if horizon_idx >= 0 and horizon_idx < len(forecast):
                row = forecast.iloc[horizon_idx].copy()
                
                # Transform back from log space if needed
                if use_log_price:
                    row['yhat'] = np.exp(row['yhat'])
                    row['yhat_lower'] = np.exp(row['yhat_lower'])
                    row['yhat_upper'] = np.exp(row['yhat_upper'])
                
                horizon_data.append({
                    'timestamp': row['ds'],
                    'horizon_minutes': horizon,
                    'yhat': row['yhat'],
                    'yhat_lower': row['yhat_lower'],
                    'yhat_upper': row['yhat_upper'],
                    'forecast_time': datetime.now()
                })
        
        return pd.DataFrame(horizon_data)
        
    except Exception as e:
        logger.error(f"Prophet backtest failed: {str(e)}")
        return None


def run_arima_garch_backtest(train_df: pd.DataFrame, config: dict, horizons: List[int]) -> Optional[pd.DataFrame]:
    """
    Run ARIMA+GARCH model for a single backtest cut.
    
    Args:
        train_df: Training data
        config: Configuration dictionary
        horizons: List of forecast horizons
        
    Returns:
        DataFrame with ARIMA+GARCH forecasts or None if failed
    """
    try:
        # Import ARIMA+GARCH modules
        sys.path.append(str(Path(__file__).parent))
        import arima_garch_pipeline as agp
        
        # Prepare data
        returns = agp.prepare_arima_data(train_df, config)
        last_price = train_df['close'].iloc[-1]
        last_timestamp = pd.to_datetime(train_df['timestamp'].iloc[-1])
        
        # Grid search and fit model
        arima_config = config['arima']
        best_p, best_q, _ = agp.grid_search_arima(
            returns,
            arima_config.get('max_p', 5),
            arima_config.get('max_q', 5)
        )
        
        arima_fitted, garch_fitted, model_info = agp.fit_arima_garch(
            returns, best_p, best_q, config
        )
        
        # Generate forecasts
        forecasts = agp.generate_forecasts(
            arima_fitted, garch_fitted, horizons, config, last_price
        )
        
        # Extract horizon forecasts
        horizon_df = agp.extract_horizon_forecasts(forecasts, horizons, last_timestamp)
        
        return horizon_df
        
    except Exception as e:
        logger.error(f"ARIMA+GARCH backtest failed: {str(e)}")
        return None


def get_realized_prices(df: pd.DataFrame, cutoff_idx: int, horizons: List[int]) -> pd.DataFrame:
    """
    Get realized prices for evaluation at specified horizons.
    
    Args:
        df: Full dataset
        cutoff_idx: Index of cutoff point
        horizons: List of forecast horizons in minutes
        
    Returns:
        DataFrame with realized prices
    """
    cutoff_time = pd.to_datetime(df['timestamp'].iloc[cutoff_idx])
    realized_data = []
    
    for horizon in horizons:
        target_time = cutoff_time + timedelta(minutes=horizon)
        
        # Find closest timestamp to target
        time_diffs = np.abs(pd.to_datetime(df['timestamp']) - target_time)
        closest_idx = time_diffs.idxmin()
        
        if time_diffs.loc[closest_idx] <= timedelta(minutes=5):  # Within 5 minutes tolerance
            realized_data.append({
                'horizon_minutes': horizon,
                'timestamp': target_time,
                'actual_price': df['close'].iloc[closest_idx],
                'actual_timestamp': df['timestamp'].iloc[closest_idx]
            })
    
    return pd.DataFrame(realized_data)


def evaluate_cut(forecast_df: pd.DataFrame, realized_df: pd.DataFrame, model_name: str) -> List[Dict]:
    """
    Evaluate forecast performance for a single cut.
    
    Args:
        forecast_df: Model forecasts
        realized_df: Realized prices
        model_name: Name of the model
        
    Returns:
        List of metric dictionaries by horizon
    """
    results = []
    
    # Merge forecasts with realized prices
    merged = pd.merge(forecast_df, realized_df, on='horizon_minutes', how='inner')
    
    if len(merged) == 0:
        logger.warning(f"No matching data for {model_name} evaluation")
        return results
    
    for _, row in merged.iterrows():
        horizon = row['horizon_minutes']
        actual = row['actual_price']
        
        if model_name == 'prophet':
            predicted = row['yhat']
            lower = row['yhat_lower']
            upper = row['yhat_upper']
        else:  # arima_garch
            predicted = row['analytical_mean']
            lower = row['mc_p05']
            upper = row['mc_p95']
        
        # Calculate metrics
        metrics = calculate_all_metrics(
            np.array([actual]),
            np.array([predicted]),
            np.array([lower]),
            np.array([upper])
        )
        
        metrics.update({
            'model': model_name,
            'horizon_minutes': horizon,
            'actual_price': actual,
            'predicted_price': predicted,
            'prediction_error': predicted - actual,
            'prediction_error_pct': ((predicted - actual) / actual) * 100
        })
        
        results.append(metrics)
    
    return results


def run_backtest(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Run complete backtesting framework.
    
    Args:
        df: Full dataset
        config: Configuration dictionary
        
    Returns:
        Tuple of (summary_results, individual_results)
    """
    backtest_config = config['backtest']
    horizons = config['horizons']
    
    # Generate backtest cuts
    cuts = generate_backtest_cuts(df, config)
    
    all_results = []
    individual_results = []
    
    with ProgressLogger(logger, f"Backtesting {len(cuts)} cuts") as progress:
        
        for cut_idx, (start_idx, cutoff_idx) in enumerate(cuts):
            progress.step(f"Processing cut {cut_idx + 1}/{len(cuts)}")
            
            # Extract training data
            train_df = df.iloc[start_idx:cutoff_idx + 1].copy()
            
            if len(train_df) < 50:  # Skip cuts with insufficient data
                continue
            
            cut_results = []
            
            # Run Prophet if enabled
            if backtest_config.get('enable_prophet', True):
                prophet_forecast = run_prophet_backtest(train_df, config, horizons)
                if prophet_forecast is not None:
                    realized_prices = get_realized_prices(df, cutoff_idx, horizons)
                    prophet_metrics = evaluate_cut(prophet_forecast, realized_prices, 'prophet')
                    cut_results.extend(prophet_metrics)
            
            # Run ARIMA+GARCH if enabled
            if backtest_config.get('enable_arima_garch', True):
                arima_forecast = run_arima_garch_backtest(train_df, config, horizons)
                if arima_forecast is not None:
                    realized_prices = get_realized_prices(df, cutoff_idx, horizons)
                    arima_metrics = evaluate_cut(arima_forecast, realized_prices, 'arima_garch')
                    cut_results.extend(arima_metrics)
            
            # Add cut metadata
            for result in cut_results:
                result.update({
                    'cut_idx': cut_idx,
                    'cutoff_timestamp': df['timestamp'].iloc[cutoff_idx],
                    'train_start_timestamp': df['timestamp'].iloc[start_idx],
                    'train_samples': len(train_df)
                })
            
            all_results.extend(cut_results)
            
            # Store individual results if requested
            if backtest_config.get('save_individual', False):
                individual_results.extend(cut_results)
    
    # Convert to DataFrame
    if not all_results:
        logger.warning("No backtest results generated")
        return pd.DataFrame(), None
    
    results_df = pd.DataFrame(all_results)
    
    # Calculate summary statistics by model and horizon
    summary_stats = []
    
    for model in results_df['model'].unique():
        model_results = results_df[results_df['model'] == model]
        
        for horizon in model_results['horizon_minutes'].unique():
            horizon_results = model_results[model_results['horizon_minutes'] == horizon]
            
            summary = {
                'model': model,
                'horizon_minutes': horizon,
                'n_cuts': len(horizon_results),
                'mean_rmse': horizon_results['rmse'].mean(),
                'std_rmse': horizon_results['rmse'].std(),
                'mean_mae': horizon_results['mae'].mean(),
                'mean_mape': horizon_results['mape'].mean(),
                'mean_smape': horizon_results['smape'].mean(),
                'mean_coverage': horizon_results['coverage'].mean(),
                'mean_interval_width': horizon_results['avg_interval_width'].mean(),
                'mean_prediction_error_pct': horizon_results['prediction_error_pct'].mean(),
                'std_prediction_error_pct': horizon_results['prediction_error_pct'].std()
            }
            
            summary_stats.append(summary)
    
    summary_df = pd.DataFrame(summary_stats)
    individual_df = pd.DataFrame(individual_results) if individual_results else None
    
    return summary_df, individual_df


def save_backtest_results(summary_df: pd.DataFrame, individual_df: Optional[pd.DataFrame], 
                         config: dict) -> Tuple[str, Optional[str]]:
    """
    Save backtesting results.
    
    Args:
        summary_df: Summary results
        individual_df: Individual cut results (optional)
        config: Configuration dictionary
        
    Returns:
        Tuple of (summary_path, individual_path)
    """
    output_dir = Path(config['paths']['backtest_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary results
    summary_path = output_dir / f"backtest_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved backtest summary to: {summary_path}")
    
    # Save individual results if available
    individual_path = None
    if individual_df is not None:
        individual_path = output_dir / f"backtest_individual_{timestamp}.parquet"
        individual_df.to_parquet(individual_path, index=False)
        logger.info(f"Saved individual results to: {individual_path}")
    
    return str(summary_path), str(individual_path) if individual_path else None


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Multi-Horizon Backtesting Framework')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    global logger
    logger = setup_logging(level=config.get('logging', {}).get('level', 'INFO'))
    
    with ProgressLogger(logger, "Multi-Horizon Backtesting"):
        try:
            # Load data
            data_path = config['paths']['data_parquet']
            logger.info(f"Loading data from: {data_path}")
            
            df = pd.read_parquet(data_path)
            log_data_summary(logger, df, "Input Data")
            
            # Validate and prepare data
            df = handle_missing_ohlc(df)
            validate_data_structure(df, ['timestamp', 'close'])
            
            # Run backtesting
            logger.info("Starting backtesting framework...")
            summary_df, individual_df = run_backtest(df, config)
            
            if len(summary_df) == 0:
                logger.error("No backtest results generated")
                return
            
            # Log summary statistics
            logger.info("=== Backtest Summary ===")
            for _, row in summary_df.iterrows():
                logger.info(f"{row['model']} ({row['horizon_minutes']}min): "
                          f"RMSE={row['mean_rmse']:.2f}±{row['std_rmse']:.2f}, "
                          f"MAPE={row['mean_mape']:.2f}%, "
                          f"Coverage={row['mean_coverage']:.2f}")
            
            # Save results
            logger.info("Saving backtest results...")
            summary_path, individual_path = save_backtest_results(summary_df, individual_df, config)
            
            logger.info("Backtesting completed successfully!")
            
        except Exception as e:
            logger.error(f"Backtesting failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()