"""
Sample data generator for testing BTC forecasting system.

Creates synthetic OHLCV data for testing purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_sample_btc_data(start_date='2024-01-01', periods=10080, freq='1T'):
    """
    Generate sample BTC price data.
    
    Args:
        start_date: Start date for data
        periods: Number of periods
        freq: Frequency (1T = 1 minute)
    
    Returns:
        DataFrame with OHLCV data
    """
    # Create datetime index
    timestamps = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Generate realistic price movement
    np.random.seed(42)  # For reproducibility
    
    # Start with base price
    base_price = 45000.0
    
    # Generate returns with some volatility clustering
    returns = np.random.normal(0, 0.001, periods)  # 0.1% average volatility per minute
    
    # Add some trend and mean reversion
    trend = np.sin(np.arange(periods) / 1440) * 0.0005  # Daily cycle
    returns += trend
    
    # Create price series
    prices = [base_price]
    for i in range(1, periods):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Generate OHLC from price series
    data = []
    for i in range(periods):
        # Add small random variations for OHLC
        close = prices[i]
        
        # High and low around close
        volatility = abs(returns[i]) * close
        high = close + np.random.uniform(0, volatility)
        low = close - np.random.uniform(0, volatility)
        
        # Open is previous close with small gap
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.0002))
        
        # Volume (random but realistic)
        volume = np.random.lognormal(10, 1)  # Random volume
        
        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': max(open_price, high, close),
            'low': min(open_price, low, close),
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample BTC data...")
    df = generate_sample_btc_data(periods=10080)  # 7 days of minute data
    
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    output_path = output_dir / "btcusdt_1m_ml_ready.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"Sample data saved to: {output_path}")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")