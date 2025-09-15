"""
Logging utility functions for BTC forecasting system.

Provides centralized logging configuration.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None, 
                  format_string: Optional[str] = None) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional file path for log output
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[]
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    
    # Get root logger and add console handler
    logger = logging.getLogger()
    logger.handlers = []  # Clear existing handlers
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    logger.info(f"Logging configured at {level} level")
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_model_performance(logger: logging.Logger, model_name: str, metrics: dict) -> None:
    """
    Log model performance metrics in a standardized format.
    
    Args:
        logger: Logger instance
        model_name: Name of the model
        metrics: Dictionary of performance metrics
    """
    logger.info(f"=== {model_name} Performance ===")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{metric_name}: {value:.6f}")
        else:
            logger.info(f"{metric_name}: {value}")


def log_data_summary(logger: logging.Logger, df, description: str = "Dataset") -> None:
    """
    Log summary statistics for a dataset.
    
    Args:
        logger: Logger instance
        df: DataFrame to summarize
        description: Description of the dataset
    """
    logger.info(f"=== {description} Summary ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    if 'timestamp' in df.columns:
        logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Log basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        logger.info("Numeric column statistics:")
        for col in numeric_cols:
            stats = df[col].describe()
            logger.info(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                       f"min={stats['min']:.2f}, max={stats['max']:.2f}")


def log_forecast_summary(logger: logging.Logger, horizon_minutes: int, 
                        forecast_df, model_name: str) -> None:
    """
    Log summary of forecast results.
    
    Args:
        logger: Logger instance
        horizon_minutes: Forecast horizon in minutes
        forecast_df: DataFrame with forecast results
        model_name: Name of the forecasting model
    """
    logger.info(f"=== {model_name} Forecast Summary (Horizon: {horizon_minutes}min) ===")
    logger.info(f"Forecast points: {len(forecast_df)}")
    
    if 'yhat' in forecast_df.columns:
        pred_stats = forecast_df['yhat'].describe()
        logger.info(f"Prediction range: {pred_stats['min']:.2f} to {pred_stats['max']:.2f}")
        logger.info(f"Mean prediction: {pred_stats['mean']:.2f}")
    
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
        avg_width = (forecast_df['yhat_upper'] - forecast_df['yhat_lower']).mean()
        logger.info(f"Average interval width: {avg_width:.2f}")


class ProgressLogger:
    """
    Context manager for logging progress of long-running operations.
    """
    
    def __init__(self, logger: logging.Logger, operation: str, total_steps: Optional[int] = None):
        self.logger = logger
        self.operation = operation
        self.total_steps = total_steps
        self.current_step = 0
        
    def __enter__(self):
        self.logger.info(f"Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"Completed {self.operation}")
        else:
            self.logger.error(f"Failed {self.operation}: {exc_val}")
            
    def step(self, description: str = "") -> None:
        """Log progress step."""
        self.current_step += 1
        if self.total_steps:
            progress = f"({self.current_step}/{self.total_steps})"
        else:
            progress = f"(step {self.current_step})"
            
        msg = f"{self.operation} {progress}"
        if description:
            msg += f": {description}"
            
        self.logger.info(msg)