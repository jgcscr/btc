CREATE OR REPLACE TABLE `jc-financial-466902.btc_forecast_curated.btc_features_1h` AS
WITH spot AS (
  SELECT
    ts,
    open,
    high,
    low,
    close,
    volume,
    quote_volume,
    num_trades
  FROM `jc-financial-466902.btc_forecast_raw.spot_klines`
  WHERE `interval` = '1h'
),
fut AS (
  SELECT
    ts AS fut_ts,
    open AS fut_open,
    high AS fut_high,
    low AS fut_low,
    close AS fut_close,
    volume AS fut_volume,
    open_interest,
    funding_rate
  FROM `jc-financial-466902.btc_forecast_raw.futures_metrics`
  WHERE `interval` = '1h'
),
joined AS (
  SELECT
    s.ts,
    s.open,
    s.high,
    s.low,
    s.close,
    s.volume,
    s.quote_volume,
    s.num_trades,
    f.fut_open,
    f.fut_high,
    f.fut_low,
    f.fut_close,
    f.fut_volume,
    f.open_interest,
    f.funding_rate
  FROM spot s
  LEFT JOIN fut f
    ON f.fut_ts = s.ts
),
with_returns AS (
  SELECT
    *,
    SAFE.LOG(close) AS log_close
  FROM joined
),
features AS (
  SELECT
    *,
    -- Backward-looking 1h log return: uses close_t and close_(t-1h)
    SAFE.LOG(close)
      - LAG(SAFE.LOG(close)) OVER (ORDER BY ts) AS ret_1h,

    -- Optional forward-looking 3h log return target (USES FUTURE DATA)
    -- Only use this as a supervised target, never as an input feature.
    LEAD(SAFE.LOG(close), 3) OVER (ORDER BY ts)
      - SAFE.LOG(close) AS ret_fwd_3h
  FROM with_returns
)
SELECT
  ts,
  open,
  high,
  low,
  close,
  volume,
  quote_volume,
  num_trades,
  fut_open,
  fut_high,
  fut_low,
  fut_close,
  fut_volume,
  open_interest,
  funding_rate,

  -- Targets
  ret_1h,
  ret_fwd_3h,

  -- Rolling 7h and 24h moving averages of close (backward-looking only)
  AVG(close) OVER (
    ORDER BY ts
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS ma_close_7h,

  AVG(close) OVER (
    ORDER BY ts
    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
  ) AS ma_close_24h,

  SAFE_DIVIDE(
    AVG(close) OVER (
      ORDER BY ts
      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ),
    AVG(close) OVER (
      ORDER BY ts
      ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
    )
  ) AS ma_ratio_7_24,

  -- Rolling 24h volatility of 1h returns (stddev of past 24 ret_1h values)
  STDDEV(ret_1h) OVER (
    ORDER BY ts
    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
  ) AS vol_24h
FROM features
ORDER BY ts;
