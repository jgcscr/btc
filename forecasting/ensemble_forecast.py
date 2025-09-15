#!/usr/bin/env python3
"""
Ensemble Forecasting Module for BTC Price Prediction.

Features:
- Model combination with configurable weighting strategies
- Static weights or inverse RMSE weighting from backtest results
- Weighted average of forecasts and prediction intervals
- Automatic detection of latest model outputs
- Comprehensive ensemble forecast table

Usage:
    python forecasting/ensemble_forecast.py --config config/forecast_config.yaml

Author: BTC Forecasting System
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from forecasting.utils import setup_logging
from forecasting.utils.logging_utils import log_data_summary, ProgressLogger

logger = None


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_latest_forecasts(config: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the most recent Prophet and ARIMA+GARCH forecast files.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (prophet_file_path, arima_file_path)
    """
    prophet_dir = Path(config['paths']['output_dir'])
    arima_dir = Path(config['paths']['arima_output_dir'])
    
    # Find latest Prophet horizon forecast
    prophet_pattern = str(prophet_dir / "prophet_forecast_horizons_*.parquet")
    prophet_files = glob.glob(prophet_pattern)
    latest_prophet = max(prophet_files, key=lambda x: Path(x).stat().st_mtime) if prophet_files else None
    
    # Find latest ARIMA+GARCH horizon forecast
    arima_pattern = str(arima_dir / "arima_garch_horizons_*.parquet")
    arima_files = glob.glob(arima_pattern)
    latest_arima = max(arima_files, key=lambda x: Path(x).stat().st_mtime) if arima_files else None
    
    if latest_prophet:
        logger.info(f"Found latest Prophet forecast: {latest_prophet}")
    else:
        logger.warning("No Prophet forecast files found")
    
    if latest_arima:
        logger.info(f"Found latest ARIMA+GARCH forecast: {latest_arima}")
    else:
        logger.warning("No ARIMA+GARCH forecast files found")
    
    return latest_prophet, latest_arima


def load_model_forecasts(prophet_path: Optional[str], arima_path: Optional[str]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load forecast results from model output files.
    
    Args:
        prophet_path: Path to Prophet forecast file
        arima_path: Path to ARIMA+GARCH forecast file
        
    Returns:
        Tuple of (prophet_df, arima_df)
    """
    prophet_df = None
    arima_df = None
    
    if prophet_path and Path(prophet_path).exists():
        try:
            prophet_df = pd.read_parquet(prophet_path)
            logger.info(f"Loaded Prophet forecasts: {len(prophet_df)} horizon predictions")
        except Exception as e:
            logger.error(f"Failed to load Prophet forecasts: {str(e)}")
    
    if arima_path and Path(arima_path).exists():
        try:
            arima_df = pd.read_parquet(arima_path)
            logger.info(f"Loaded ARIMA+GARCH forecasts: {len(arima_df)} horizon predictions")
        except Exception as e:
            logger.error(f"Failed to load ARIMA+GARCH forecasts: {str(e)}")
    
    return prophet_df, arima_df


def get_weights_from_backtest(config: dict) -> Optional[Dict[str, float]]:
    """
    Calculate model weights based on inverse RMSE from backtest results.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with model weights or None if no backtest results
    """
    backtest_dir = Path(config['paths']['backtest_dir'])
    
    # Find latest backtest summary
    summary_pattern = str(backtest_dir / "backtest_summary_*.csv")
    summary_files = glob.glob(summary_pattern)
    
    if not summary_files:
        logger.warning("No backtest summary files found for inverse RMSE weighting")
        return None
    
    latest_summary = max(summary_files, key=lambda x: Path(x).stat().st_mtime)
    logger.info(f"Using backtest results from: {latest_summary}")
    
    try:
        backtest_df = pd.read_csv(latest_summary)
        
        # Calculate average RMSE across all horizons for each model
        model_rmse = backtest_df.groupby('model')['mean_rmse'].mean()
        
        if len(model_rmse) == 0:
            logger.warning("No RMSE data found in backtest results")
            return None
        
        # Calculate inverse RMSE weights
        inverse_rmse = 1 / model_rmse
        weights = inverse_rmse / inverse_rmse.sum()
        
        weight_dict = weights.to_dict()
        
        # Map model names to standard format
        weight_mapping = {}
        if 'prophet' in weight_dict:
            weight_mapping['prophet'] = weight_dict['prophet']
        if 'arima_garch' in weight_dict:
            weight_mapping['arima_garch'] = weight_dict['arima_garch']
        
        logger.info(f"Calculated inverse RMSE weights: {weight_mapping}")
        return weight_mapping
        
    except Exception as e:
        logger.error(f"Failed to calculate weights from backtest: {str(e)}")
        return None


def get_model_weights(config: dict) -> Dict[str, float]:
    """
    Get model weights based on configuration method.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with model weights
    """
    ensemble_config = config['ensemble']
    method = ensemble_config.get('method', 'static')
    
    if method == 'inverse_rmse':
        weights = get_weights_from_backtest(config)
        if weights is None:
            logger.warning("Falling back to static weights")
            weights = ensemble_config.get('static_weights', {'prophet': 0.5, 'arima_garch': 0.5})
    else:  # static
        weights = ensemble_config.get('static_weights', {'prophet': 0.5, 'arima_garch': 0.5})
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    logger.info(f"Using model weights: {weights}")
    return weights


def standardize_forecast_columns(prophet_df: Optional[pd.DataFrame], arima_df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Standardize column names across different model forecasts.
    
    Args:
        prophet_df: Prophet forecast DataFrame
        arima_df: ARIMA+GARCH forecast DataFrame
        
    Returns:
        Tuple of standardized DataFrames
    """
    # Standardize Prophet columns
    if prophet_df is not None:
        prophet_std = prophet_df.copy()
        # Prophet already has yhat, yhat_lower, yhat_upper
        prophet_std['model'] = 'prophet'
        prophet_std['mean_forecast'] = prophet_std['yhat']
        prophet_std['lower_forecast'] = prophet_std['yhat_lower']
        prophet_std['upper_forecast'] = prophet_std['yhat_upper']
    else:
        prophet_std = None
    
    # Standardize ARIMA+GARCH columns
    if arima_df is not None:
        arima_std = arima_df.copy()
        arima_std['model'] = 'arima_garch'
        arima_std['mean_forecast'] = arima_std['analytical_mean']
        arima_std['lower_forecast'] = arima_std['mc_p05']  # 5th percentile for 90% interval
        arima_std['upper_forecast'] = arima_std['mc_p95']  # 95th percentile for 90% interval
    else:
        arima_std = None
    
    return prophet_std, arima_std


def combine_forecasts(prophet_df: Optional[pd.DataFrame], arima_df: Optional[pd.DataFrame], 
                     weights: Dict[str, float]) -> pd.DataFrame:
    """
    Combine model forecasts using weighted averaging.
    
    Args:
        prophet_df: Standardized Prophet forecasts
        arima_df: Standardized ARIMA+GARCH forecasts
        weights: Model weights dictionary
        
    Returns:
        Combined ensemble forecast DataFrame
    """
    # Determine which models are available
    available_models = []
    if prophet_df is not None and 'prophet' in weights:
        available_models.append('prophet')
    if arima_df is not None and 'arima_garch' in weights:
        available_models.append('arima_garch')
    
    if not available_models:
        raise ValueError("No model forecasts available for ensemble")
    
    # Normalize weights for available models only
    available_weights = {model: weights[model] for model in available_models}
    total_weight = sum(available_weights.values())
    available_weights = {k: v / total_weight for k, v in available_weights.items()}
    
    logger.info(f"Combining forecasts with weights: {available_weights}")
    
    # Start with one of the available forecasts as base
    if prophet_df is not None:
        ensemble_df = prophet_df[['timestamp', 'horizon_minutes']].copy()
        base_df = prophet_df
    else:
        ensemble_df = arima_df[['timestamp', 'horizon_minutes']].copy()
        base_df = arima_df
    
    # Initialize ensemble columns
    ensemble_df['mean_price'] = 0.0
    ensemble_df['lower_price'] = 0.0
    ensemble_df['upper_price'] = 0.0
    
    # Combine forecasts for each horizon
    for horizon in ensemble_df['horizon_minutes'].unique():
        horizon_mask = ensemble_df['horizon_minutes'] == horizon
        
        weighted_mean = 0.0
        weighted_lower = 0.0
        weighted_upper = 0.0
        
        # Add Prophet contribution
        if 'prophet' in available_weights and prophet_df is not None:
            prophet_horizon = prophet_df[prophet_df['horizon_minutes'] == horizon]
            if len(prophet_horizon) > 0:
                weight = available_weights['prophet']
                weighted_mean += weight * prophet_horizon['mean_forecast'].iloc[0]
                weighted_lower += weight * prophet_horizon['lower_forecast'].iloc[0]
                weighted_upper += weight * prophet_horizon['upper_forecast'].iloc[0]
        
        # Add ARIMA+GARCH contribution
        if 'arima_garch' in available_weights and arima_df is not None:
            arima_horizon = arima_df[arima_df['horizon_minutes'] == horizon]
            if len(arima_horizon) > 0:
                weight = available_weights['arima_garch']
                weighted_mean += weight * arima_horizon['mean_forecast'].iloc[0]
                weighted_lower += weight * arima_horizon['lower_forecast'].iloc[0]
                weighted_upper += weight * arima_horizon['upper_forecast'].iloc[0]
        
        # Set ensemble values for this horizon
        ensemble_df.loc[horizon_mask, 'mean_price'] = weighted_mean
        ensemble_df.loc[horizon_mask, 'lower_price'] = weighted_lower
        ensemble_df.loc[horizon_mask, 'upper_price'] = weighted_upper
    
    # Add metadata
    ensemble_df['ensemble_method'] = 'weighted_average'
    ensemble_df['models_used'] = ','.join(available_models)
    
    # Add model weights as columns
    for model, weight in available_weights.items():
        ensemble_df[f'weight_{model}'] = weight
    
    ensemble_df['forecast_time'] = datetime.now()
    
    return ensemble_df


def save_ensemble_forecast(ensemble_df: pd.DataFrame, config: dict) -> str:
    """
    Save ensemble forecast results.
    
    Args:
        ensemble_df: Ensemble forecast DataFrame
        config: Configuration dictionary
        
    Returns:
        Path to saved ensemble forecast file
    """
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ensemble_forecast_{timestamp}.parquet"
    
    ensemble_df.to_parquet(output_path, index=False)
    logger.info(f"Saved ensemble forecast to: {output_path}")
    
    return str(output_path)


def log_ensemble_summary(ensemble_df: pd.DataFrame) -> None:
    """
    Log summary of ensemble forecast results.
    
    Args:
        ensemble_df: Ensemble forecast DataFrame
    """
    logger.info("=== Ensemble Forecast Summary ===")
    
    for horizon in sorted(ensemble_df['horizon_minutes'].unique()):
        horizon_data = ensemble_df[ensemble_df['horizon_minutes'] == horizon]
        if len(horizon_data) > 0:
            row = horizon_data.iloc[0]
            interval_width = row['upper_price'] - row['lower_price']
            logger.info(f"Horizon {horizon}min: "
                       f"Price={row['mean_price']:.2f}, "
                       f"Interval=[{row['lower_price']:.2f}, {row['upper_price']:.2f}], "
                       f"Width={interval_width:.2f}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Ensemble Forecasting Module')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    global logger
    logger = setup_logging(level=config.get('logging', {}).get('level', 'INFO'))
    
    with ProgressLogger(logger, "Ensemble Forecasting"):
        try:
            # Find latest forecast files
            logger.info("Looking for latest model forecasts...")
            prophet_path, arima_path = find_latest_forecasts(config)
            
            if not prophet_path and not arima_path:
                logger.error("No model forecast files found. Run individual models first.")
                return
            
            # Load model forecasts
            logger.info("Loading model forecasts...")
            prophet_df, arima_df = load_model_forecasts(prophet_path, arima_path)
            
            if prophet_df is not None:
                log_data_summary(logger, prophet_df, "Prophet Forecasts")
            if arima_df is not None:
                log_data_summary(logger, arima_df, "ARIMA+GARCH Forecasts")
            
            # Get model weights
            logger.info("Calculating model weights...")
            weights = get_model_weights(config)
            
            # Standardize forecast columns
            logger.info("Standardizing forecast formats...")
            prophet_std, arima_std = standardize_forecast_columns(prophet_df, arima_df)
            
            # Combine forecasts
            logger.info("Combining model forecasts...")
            ensemble_df = combine_forecasts(prophet_std, arima_std, weights)
            
            # Log ensemble summary
            log_ensemble_summary(ensemble_df)
            
            # Save ensemble forecast
            logger.info("Saving ensemble forecast...")
            output_path = save_ensemble_forecast(ensemble_df, config)
            
            logger.info("Ensemble forecasting completed successfully!")
            
        except Exception as e:
            logger.error(f"Ensemble forecasting failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()