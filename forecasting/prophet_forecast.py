#!/usr/bin/env python3
"""
Enhanced Prophet Forecasting Pipeline for BTC Price Prediction.

Features:
- Optional log price transformation
- Intraday seasonality with configurable Fourier terms
- Adaptive changepoint parameters
- Deterministic future regressors (cyclic hour/minute features)
- Horizon extraction for 4h/8h/12h forecasts
- Comprehensive output with prediction intervals

Usage:
    python forecasting/prophet_forecast.py --config config/forecast_config.yaml

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
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from prophet import Prophet
from forecasting.utils import setup_logging, add_time_features, handle_missing_ohlc, validate_data_structure
from forecasting.utils.logging_utils import log_data_summary, log_forecast_summary, ProgressLogger

logger = None


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_prophet_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Prepare data for Prophet model training.
    
    Args:
        df: Input DataFrame with OHLC data
        config: Configuration dictionary
        
    Returns:
        DataFrame formatted for Prophet
    """
    prophet_config = config['prophet']
    
    # Create Prophet format (ds, y)
    prophet_df = df[['timestamp', 'close']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Apply log transformation if specified
    if prophet_config.get('use_log_price', False):
        prophet_df['y'] = np.log(prophet_df['y'])
        logger.info("Applied log transformation to price data")
    
    # Add time features for regressors
    df_with_features = add_time_features(df, 'timestamp')
    prophet_df['hour_sin'] = df_with_features['hour_sin']
    prophet_df['hour_cos'] = df_with_features['hour_cos']
    prophet_df['minute_sin'] = df_with_features['minute_sin']
    prophet_df['minute_cos'] = df_with_features['minute_cos']
    
    logger.info(f"Prepared Prophet dataset with {len(prophet_df)} observations")
    return prophet_df


def create_prophet_model(config: dict) -> Prophet:
    """
    Create and configure Prophet model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured Prophet model
    """
    prophet_config = config['prophet']
    
    model = Prophet(
        changepoint_prior_scale=prophet_config.get('changepoint_prior_scale', 0.2),
        changepoint_range=prophet_config.get('changepoint_range', 0.98),
        seasonality_mode=prophet_config.get('seasonality_mode', 'additive'),
        interval_width=prophet_config.get('interval_width', 0.9),
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False  # We'll add custom intraday seasonality
    )
    
    # Add intraday seasonality (1440 minutes = 1 day)
    intraday_fourier = prophet_config.get('intraday_fourier', 20)
    model.add_seasonality(
        name='intraday',
        period=1440,  # minutes in a day
        fourier_order=intraday_fourier
    )
    
    # Add time-based regressors
    model.add_regressor('hour_sin')
    model.add_regressor('hour_cos')
    model.add_regressor('minute_sin')
    model.add_regressor('minute_cos')
    
    logger.info(f"Created Prophet model with intraday seasonality (fourier_order={intraday_fourier})")
    return model


def generate_future_dataframe(model: Prophet, periods: int, freq: str = 'T') -> pd.DataFrame:
    """
    Generate future dataframe with regressors.
    
    Args:
        model: Fitted Prophet model
        periods: Number of periods to forecast
        freq: Frequency for forecast (default: 'T' for minutes)
        
    Returns:
        Future dataframe with regressors
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    
    # Add time features for future dates
    future_with_features = add_time_features(future, 'ds')
    future['hour_sin'] = future_with_features['hour_sin']
    future['hour_cos'] = future_with_features['hour_cos'] 
    future['minute_sin'] = future_with_features['minute_sin']
    future['minute_cos'] = future_with_features['minute_cos']
    
    return future


def extract_horizon_forecasts(forecast_df: pd.DataFrame, horizons: list, 
                             use_log_price: bool = False) -> pd.DataFrame:
    """
    Extract specific horizon forecasts from full forecast.
    
    Args:
        forecast_df: Full Prophet forecast DataFrame
        horizons: List of forecast horizons in minutes
        use_log_price: Whether log transformation was applied
        
    Returns:
        DataFrame with horizon-specific forecasts
    """
    horizon_data = []
    
    for horizon in horizons:
        # Get forecast at specific horizon (relative to last training point)
        horizon_idx = len(forecast_df) - max(horizons) + horizon - 1
        
        if horizon_idx >= 0 and horizon_idx < len(forecast_df):
            row = forecast_df.iloc[horizon_idx].copy()
            
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


def save_outputs(forecast_df: pd.DataFrame, horizon_df: pd.DataFrame, 
                config: dict, use_log_price: bool = False) -> tuple:
    """
    Save Prophet forecast outputs.
    
    Args:
        forecast_df: Full forecast DataFrame
        horizon_df: Horizon-specific forecasts
        config: Configuration dictionary
        use_log_price: Whether log transformation was applied
        
    Returns:
        Tuple of (full_forecast_path, horizon_forecast_path)
    """
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Transform full forecast back from log space if needed
    forecast_output = forecast_df.copy()
    if use_log_price:
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            if col in forecast_output.columns:
                forecast_output[col] = np.exp(forecast_output[col])
    
    # Save full forecast
    full_path = output_dir / f"prophet_forecast_full_{timestamp}.parquet"
    forecast_output.to_parquet(full_path, index=False)
    
    # Save horizon forecasts
    horizon_path = output_dir / f"prophet_forecast_horizons_{timestamp}.parquet"
    horizon_df.to_parquet(horizon_path, index=False)
    
    logger.info(f"Saved full forecast to: {full_path}")
    logger.info(f"Saved horizon forecasts to: {horizon_path}")
    
    return str(full_path), str(horizon_path)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Prophet Forecasting Pipeline')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    global logger
    logger = setup_logging(level=config.get('logging', {}).get('level', 'INFO'))
    
    with ProgressLogger(logger, "Prophet Forecasting Pipeline"):
        try:
            # Load data
            data_path = config['paths']['data_parquet']
            logger.info(f"Loading data from: {data_path}")
            
            df = pd.read_parquet(data_path)
            log_data_summary(logger, df, "Input Data")
            
            # Validate and prepare data
            df = handle_missing_ohlc(df)
            validate_data_structure(df, ['timestamp', 'close'])
            
            # Prepare Prophet data
            prophet_df = prepare_prophet_data(df, config)
            use_log_price = config['prophet'].get('use_log_price', False)
            
            # Create and fit model
            logger.info("Creating Prophet model...")
            model = create_prophet_model(config)
            
            logger.info("Fitting Prophet model...")
            model.fit(prophet_df)
            
            # Generate forecasts
            horizons = config['horizons']
            max_horizon = max(horizons)
            
            logger.info(f"Generating forecasts for {max_horizon} minutes ahead...")
            future = generate_future_dataframe(model, periods=max_horizon)
            forecast = model.predict(future)
            
            # Extract horizon forecasts
            logger.info("Extracting horizon-specific forecasts...")
            horizon_df = extract_horizon_forecasts(forecast, horizons, use_log_price)
            
            # Log forecast summaries
            for horizon in horizons:
                horizon_subset = horizon_df[horizon_df['horizon_minutes'] == horizon]
                log_forecast_summary(logger, horizon, horizon_subset, "Prophet")
            
            # Save outputs
            logger.info("Saving forecast outputs...")
            full_path, horizon_path = save_outputs(forecast, horizon_df, config, use_log_price)
            
            logger.info("Prophet forecasting completed successfully!")
            
        except Exception as e:
            logger.error(f"Prophet forecasting failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()