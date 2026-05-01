from __future__ import annotations


LOCAL_FEATURE_OPTIONAL_PATHS: tuple[tuple[str, str], ...] = (
    ("macro_path", "macro"),
    ("onchain_path", "onchain"),
    ("funding_path", "funding"),
    ("intrabar_path", "intrabar"),
)

LOCAL_FEATURE_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "macro": tuple(),
    "funding": ("funding_rate", "funding_rate_annualized"),
    "onchain": ("onchain_active_addresses", "onchain_transaction_count"),
}