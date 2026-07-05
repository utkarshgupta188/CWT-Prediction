# CWT Prediction System

A modular, resilient, and unit-tested crypto time-series prediction system utilizing the **Kronos** foundation model for financial forecasting and **Kelly Criterion** for position sizing, packaged with **Hermes Agent** integration.

## Project Structure

- `cwt_prediction/` - Main Python package.
  - `market_finder/` - Queries Polymarket (Gamma API) and Kalshi (v2 public API) for crypto prediction markets.
  - `data_fetcher/` - Pulls historical OHLCV data from Binance API (falls back to CoinGecko).
  - `forecaster/` - Autoregressively forecasts future prices using Kronos-mini or Kronos-small.
  - `risk_manager/` - Sizes positions using the Kelly Criterion formula.
  - `feedback_loop/` - Logs predictions to SQLite and resolves outcomes after expiry.
  - `orchestrator/` - Runs a single full cycle of finding, fetching, forecasting, sizing, logging, and resolving.
- `tests/` - pytest test suite.
- `hermes_integration/` - Integration instructions and SKILL.md.
- `logs/` - Structured JSON pipeline logs.
- `data/` - SQLite database storage (`predictions.db`).

## Setup

1. Clone or copy the project files to your workspace.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` configuration file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Specify your `OPENROUTER_API_KEY` (if using advanced parsing/fallback agents) and optional model weights paths if changing defaults.

## Usage

### Run the Pipeline
To execute a single pipeline cycle end-to-end (Find Markets -> Fetch OHLCV -> Forecast -> Risk Size -> Log -> Resolve Expired):
```bash
python -m cwt_prediction.main
```

### Run Tests
To run the automated unit and integration tests:
```bash
python -m pytest tests/ -v
```

## Hermes Agent Integration

We have created integration files in the `hermes_integration/` directory:
- [SKILL.md](hermes_integration/SKILL.md): Copy this to your local `%USERPROFILE%\.hermes\skills\cwt-prediction\SKILL.md` (Windows) or `~/.hermes/skills/cwt-prediction/SKILL.md` (Linux/macOS) to register the `/cwt-prediction` skill.
- [cron_setup.md](hermes_integration/cron_setup.md): Instructions on configuring a scheduled Hermes task to run the pipeline on a 5-minute timer using Hermes' native cron tools.
- [CONTEXT.md](hermes_integration/CONTEXT.md): Describes the repository layout for the agent to load automatically.
