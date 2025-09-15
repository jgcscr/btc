"""
Evaluation utility functions for BTC forecasting system.

Provides metrics for forecast accuracy and interval coverage.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values  
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE value as percentage
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    # Avoid division by zero
    y_true_adj = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    
    return np.mean(np.abs((y_true - y_pred) / y_true_adj)) * 100


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        sMAPE value as percentage
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator < epsilon, epsilon, denominator)
    
    return np.mean(numerator / denominator) * 100


def calculate_coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Calculate interval coverage rate.
    
    Args:
        y_true: Actual values
        y_lower: Lower bounds of prediction intervals
        y_upper: Upper bounds of prediction intervals
        
    Returns:
        Coverage rate (proportion of actuals within intervals)
    """
    if not (len(y_true) == len(y_lower) == len(y_upper)):
        raise ValueError("All arrays must have same length")
    
    within_interval = (y_true >= y_lower) & (y_true <= y_upper)
    return np.mean(within_interval)


def calculate_interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Calculate average interval width.
    
    Args:
        y_lower: Lower bounds of prediction intervals
        y_upper: Upper bounds of prediction intervals
        
    Returns:
        Average interval width
    """
    if len(y_lower) != len(y_upper):
        raise ValueError("y_lower and y_upper must have same length")
    
    return np.mean(y_upper - y_lower)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_lower: Optional[np.ndarray] = None, 
                         y_upper: Optional[np.ndarray] = None) -> dict:
    """
    Calculate all available metrics for forecast evaluation.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_lower: Lower bounds (optional)
        y_upper: Upper bounds (optional)
        
    Returns:
        Dictionary with all calculated metrics
    """
    metrics = {
        'rmse': calculate_rmse(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred),
        'smape': calculate_smape(y_true, y_pred)
    }
    
    # Add interval metrics if bounds provided
    if y_lower is not None and y_upper is not None:
        metrics['coverage'] = calculate_coverage(y_true, y_lower, y_upper)
        metrics['avg_interval_width'] = calculate_interval_width(y_lower, y_upper)
    
    return metrics


def evaluate_forecast_horizons(forecast_df: pd.DataFrame, actual_df: pd.DataFrame,
                              horizon_col: str = 'horizon_minutes',
                              pred_col: str = 'yhat',
                              actual_col: str = 'actual_price',
                              lower_col: Optional[str] = 'yhat_lower',
                              upper_col: Optional[str] = 'yhat_upper') -> pd.DataFrame:
    """
    Evaluate forecast performance across multiple horizons.
    
    Args:
        forecast_df: DataFrame with forecasts
        actual_df: DataFrame with actual values
        horizon_col: Column name for forecast horizon
        pred_col: Column name for predictions
        actual_col: Column name for actual values
        lower_col: Column name for lower bounds (optional)
        upper_col: Column name for upper bounds (optional)
        
    Returns:
        DataFrame with metrics by horizon
    """
    results = []
    
    for horizon in forecast_df[horizon_col].unique():
        horizon_forecast = forecast_df[forecast_df[horizon_col] == horizon]
        horizon_actual = actual_df[actual_df[horizon_col] == horizon]
        
        # Merge on timestamp or index
        if 'timestamp' in horizon_forecast.columns and 'timestamp' in horizon_actual.columns:
            merged = pd.merge(horizon_forecast, horizon_actual, on='timestamp', how='inner')
        else:
            merged = pd.merge(horizon_forecast, horizon_actual, left_index=True, right_index=True, how='inner')
        
        if len(merged) == 0:
            logger.warning(f"No matching data for horizon {horizon}")
            continue
            
        # Calculate metrics
        y_true = merged[actual_col].values
        y_pred = merged[pred_col].values
        
        metrics = {'horizon_minutes': horizon}
        metrics.update(calculate_all_metrics(
            y_true, y_pred,
            merged[lower_col].values if lower_col in merged.columns else None,
            merged[upper_col].values if upper_col in merged.columns else None
        ))
        
        metrics['n_samples'] = len(merged)
        results.append(metrics)
    
    return pd.DataFrame(results)


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, 
                                  baseline: Optional[np.ndarray] = None) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        baseline: Baseline values for direction calculation (if None, uses previous actual value)
        
    Returns:
        Directional accuracy as percentage
    """
    if baseline is None:
        baseline = np.concatenate([[y_true[0]], y_true[:-1]])
    
    actual_direction = np.sign(y_true - baseline)
    predicted_direction = np.sign(y_pred - baseline)
    
    # Filter out cases where actual direction is zero (no change)
    non_zero_mask = actual_direction != 0
    if np.sum(non_zero_mask) == 0:
        return 50.0  # Random chance if no directional changes
    
    correct_directions = actual_direction[non_zero_mask] == predicted_direction[non_zero_mask]
    return np.mean(correct_directions) * 100