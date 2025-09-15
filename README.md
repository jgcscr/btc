# BTC Forecasting System

A comprehensive Bitcoin price forecasting system implementing multiple time series models with backtesting and ensemble capabilities.

## Features

### Core Models
- **Prophet**: Enhanced with log-price transformation, intraday seasonality, adaptive changepoints, and deterministic future regressors
- **ARIMA+GARCH**: Manual grid search, Student-t distribution, EGARCH fallback, Monte Carlo simulation

### Advanced Capabilities
- **Multi-Horizon Forecasting**: 4h, 8h, and 12h prediction horizons
- **Rolling Backtesting**: Configurable evaluation windows with comprehensive metrics
- **Ensemble Forecasting**: Static or inverse-RMSE weighting strategies
- **Comprehensive Metrics**: RMSE, MAE, MAPE, sMAPE, interval coverage, directional accuracy

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

1. **Prepare your data**: Place 1-minute OHLCV Bitcoin data in `data/processed/btcusdt_1m_ml_ready.parquet`

2. **Configure parameters**: Edit `config/forecast_config.yaml`

3. **Run forecasting pipelines**:

```bash
# Prophet forecasting
python forecasting/prophet_forecast.py --config config/forecast_config.yaml

# ARIMA+GARCH forecasting  
python forecasting/arima_garch_pipeline.py --config config/forecast_config.yaml

# Ensemble forecasting
python forecasting/ensemble_forecast.py --config config/forecast_config.yaml

# Backtesting evaluation
python forecasting/backtest_multi_horizon.py --config config/forecast_config.yaml
```

### Generate Sample Data

```bash
python generate_sample_data.py
```

## Configuration

All parameters are centralized in `config/forecast_config.yaml`:

```yaml
# Data paths
paths:
  data_parquet: data/processed/btcusdt_1m_ml_ready.parquet
  output_dir: data/forecasts
  arima_output_dir: data/arima_garch_forecasts
  backtest_dir: data/backtests

# Forecast horizons (minutes)
horizons: [240, 480, 720]  # 4h, 8h, 12h

# Prophet configuration
prophet:
  changepoint_prior_scale: 0.2
  use_log_price: true
  intraday_fourier: 20

# ARIMA+GARCH configuration  
arima:
  max_p: 2
  max_q: 2
  dist: student_t
  egarch_fallback: true
  monte_carlo_paths: 1000

# Backtesting parameters
backtest:
  train_window_minutes: 2880  # 2 days
  step_minutes: 1440          # 1 day

# Ensemble weighting
ensemble:
  method: inverse_rmse  # or static
```

## System Architecture

```
forecasting/
├── prophet_forecast.py      # Prophet model pipeline
├── arima_garch_pipeline.py  # ARIMA+GARCH pipeline  
├── backtest_multi_horizon.py # Rolling evaluation
├── ensemble_forecast.py     # Model combination
└── utils/
    ├── data_utils.py        # Data processing utilities
    ├── eval_utils.py        # Evaluation metrics
    └── logging_utils.py     # Logging configuration
```

## Key Technical Features

### Prophet Enhancements
- **Log-price transformation**: Handles exponential price growth
- **Intraday seasonality**: 1440-minute period with 20 Fourier terms
- **Cyclic regressors**: Hour/minute sine/cosine features
- **Adaptive changepoints**: Configurable prior scale and range

### ARIMA+GARCH Implementation
- **Manual grid search**: No pmdarima dependency
- **Multi-step forecasting**: Monte Carlo simulation for variance paths
- **Distribution flexibility**: Student-t with normal fallback
- **Robust estimation**: EGARCH fallback on convergence issues

### Backtesting Framework
- **Rolling windows**: Configurable train/test splits
- **Multiple metrics**: RMSE, MAE, MAPE, sMAPE, coverage
- **Model comparison**: Side-by-side evaluation
- **Scalable**: Handles large datasets efficiently

### Ensemble Methods
- **Dynamic weighting**: Inverse RMSE from backtest results
- **Automatic discovery**: Latest model outputs
- **Interval combination**: Weighted prediction intervals
- **Fallback strategy**: Static weights when backtest unavailable

## Output Files

All outputs are timestamped and saved to configured directories:

- **Prophet**: Full forecasts + horizon-specific predictions
- **ARIMA+GARCH**: Analytical + Monte Carlo intervals  
- **Backtesting**: Summary CSV + optional individual results
- **Ensemble**: Combined weighted forecasts

## Performance Metrics

Example results from test data:
- **Prophet (4h)**: RMSE=2,616, MAPE=2.6%, Coverage=100%
- **Prophet (8h)**: RMSE=6,718, MAPE=4.6%, Coverage=100%  
- **Prophet (12h)**: RMSE=13,879, MAPE=8.8%, Coverage=100%

## Requirements

- Python 3.8+
- Prophet 1.1+
- ARCH 5.3+ (GARCH models)
- statsmodels 0.13+
- pandas, numpy, scipy
- PyArrow (parquet support)

## Data Format

Expected input: Minute-level OHLCV data with columns:
- `timestamp`: DateTime index
- `open`, `high`, `low`, `close`: Price data
- `volume`: Trading volume (optional)

Missing High/Low columns are automatically filled with Close values.

## License

This project is open source and available under the [MIT License](LICENSE).