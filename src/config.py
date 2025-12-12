import os

PROJECT_ID = "jc-financial-466902"
BQ_DATASET_CURATED = "btc_forecast_curated"
BQ_TABLE_FEATURES_1H = "btc_features_1h"

ONCHAIN_API_BASE_URL = os.getenv("ONCHAIN_API_BASE_URL", "")
ONCHAIN_API_KEY = os.getenv("ONCHAIN_API_KEY", "")
ONCHAIN_DEFAULT_INTERVAL = os.getenv("ONCHAIN_DEFAULT_INTERVAL", "1h")
ONCHAIN_METRICS = [
	"active_addresses",
	"transaction_count",
	"hash_rate",
	"market_cap",
]

DEFAULT_ONCHAIN_CACHE_DIR = os.getenv("ONCHAIN_CACHE_DIR", "data")
