# CWT Prediction Project Context

This directory contains the CWT Prediction system, a modular Python backend for forecasting short-term (5-minute) BTC/ETH prediction markets using the Kronos foundation model and sizing positions using the Kelly Criterion.

## Project Structure

- `cwt_prediction/` - Main Python package.
  - `market_finder/` - Queries Polymarket and Kalshi APIs for open crypto price prediction markets.
  - `data_fetcher/` - Pulls historical OHLCV data from Binance API (falls back to CoinGecko).
  - `forecaster/` - Autoregressively forecasts future prices using Kronos-mini or Kronos-small.
  - `risk_manager/` - Sizes positions using the Kelly Criterion formula.
  - `feedback_loop/` - Logs predictions to SQLite and resolves outcomes after expiry.
  - `orchestrator/` - Runs a single full cycle of finding, fetching, forecasting, sizing, logging, and resolving.
- `tests/` - pytest test suite.
- `hermes_integration/` - Integration instructions and SKILL.md.
- `logs/` - Structured JSON pipeline logs.
- `data/` - SQLite database storage (`predictions.db`).

## Configuration

Settings are configured via `.env` in the root directory:
- `OPENROUTER_API_KEY` - Optional fallback API key.
- `KRONOS_MODEL` - HF model identifier (defaults to `NeoQuasar/Kronos-mini`).
- `KRONOS_TOKENIZER` - HF tokenizer identifier.
- `LOOKBACK_BARS` - Number of lookback candles (default 400).
- `FORECAST_HORIZON` - Number of bars to forecast (default 6, i.e. 30 minutes).
- `KELLY_MAX_FRACTION` - Maximum position sizing cap (default 0.25).

## How to Run locally

```bash
# Run one full cycle
python -m cwt_prediction.main

# Run tests
python -m pytest tests/ -v
```
