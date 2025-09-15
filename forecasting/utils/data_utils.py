"""
Data utility functions for BTC forecasting system.

Provides functions for data validation, preprocessing, and feature engineering.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def check_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for and handle duplicate column names in DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with unique column names
    """
    original_cols = df.columns.tolist()
    
    # Find duplicates
    duplicates = df.columns[df.columns.duplicated()].tolist()
    
    if duplicates:
        logger.warning(f"Found duplicate columns: {duplicates}")
        
        # Create unique column names
        new_cols = []
        col_counts = {}
        
        for col in original_cols:
            if col in col_counts:
                col_counts[col] += 1
                new_col = f"{col}_{col_counts[col]}"
            else:
                col_counts[col] = 0
                new_col = col
            new_cols.append(new_col)
        
        df.columns = new_cols
        logger.info(f"Renamed columns to: {new_cols}")
    
    return df


def add_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Add cyclic time features (hour, minute) as sine/cosine components.
    
    Args:
        df: Input DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Extract hour and minute
    df['hour'] = df[timestamp_col].dt.hour
    df['minute'] = df[timestamp_col].dt.minute
    
    # Add cyclic features
    # Hour: 24-hour cycle
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Minute: 60-minute cycle  
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
    logger.info("Added cyclic time features: hour_sin, hour_cos, minute_sin, minute_cos")
    
    return df


def handle_missing_ohlc(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Handle missing High/Low columns by cloning Close price if absent.
    
    Args:
        df: Input DataFrame
        price_col: Name of the main price column (usually 'close')
        
    Returns:
        DataFrame with High/Low columns filled
    """
    df = df.copy()
    
    # Check and fill missing High column
    if 'high' not in df.columns:
        df['high'] = df[price_col]
        logger.warning(f"High column missing, filled with {price_col}")
    
    # Check and fill missing Low column  
    if 'low' not in df.columns:
        df['low'] = df[price_col]
        logger.warning(f"Low column missing, filled with {price_col}")
        
    return df


def validate_data_structure(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> bool:
    """
    Validate that DataFrame has required structure for forecasting.
    
    Args:
        df: Input DataFrame
        required_cols: List of required column names
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if required_cols is None:
        required_cols = ['timestamp', 'close']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for sufficient data
    if len(df) < 100:
        raise ValueError(f"Insufficient data: {len(df)} rows (minimum 100 required)")
    
    # Check timestamp is sorted
    if not df['timestamp'].is_monotonic_increasing:
        logger.warning("Timestamp column is not sorted, sorting data")
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Check for missing values in critical columns
    for col in required_cols:
        if df[col].isna().any():
            raise ValueError(f"Missing values found in required column: {col}")
    
    logger.info(f"Data validation passed for {len(df)} rows")
    return True


def resample_data(df: pd.DataFrame, freq: str, timestamp_col: str = 'timestamp', 
                  price_col: str = 'close') -> pd.DataFrame:
    """
    Resample time series data to specified frequency.
    
    Args:
        df: Input DataFrame
        freq: Pandas frequency string (e.g., '5T', '15T')
        timestamp_col: Name of timestamp column
        price_col: Name of price column
        
    Returns:
        Resampled DataFrame
    """
    df = df.copy()
    df = df.set_index(timestamp_col)
    
    # Resample using OHLC for price data
    resampled = df[price_col].resample(freq).ohlc()
    resampled.columns = ['open', 'high', 'low', 'close']
    
    # Add volume if available
    if 'volume' in df.columns:
        resampled['volume'] = df['volume'].resample(freq).sum()
    
    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={timestamp_col: 'timestamp'})
    
    logger.info(f"Resampled data from {len(df)} to {len(resampled)} rows with frequency {freq}")
    
    return resampled