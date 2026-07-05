# Crypto Short-Term Prediction System (CWT Prediction)

A production-grade, modular, and resilient time-series forecasting and risk-management backend designed for crypto prediction markets. It leverages **Kronos** (a decoder-only foundation model trained on global exchange candlesticks) to predict short-term price movements and uses the **Kelly Criterion** to calculate optimal position sizes. It integrates directly with **Hermes Agent** through native skills and automated scheduling.

---

## 🏗️ Architectural Overview

The system is organized into clean, independent modules, each containing unit tests and defensive boundaries:

```
                  ┌─────────────────┐
                  │  Market Finder  │  <-- Finds open 5m BTC/ETH markets
                  └────────┬────────┘      on Polymarket & Kalshi APIs
                           │
                           ▼
                  ┌─────────────────┐
                  │  Data Fetcher   │  <-- Pulls last 400 OHLCV candles
                  └────────┬────────┘      from Binance API (fallback: CoinGecko)
                           │
                           ▼
                  ┌─────────────────┐
                  │   Forecaster    │  <-- Feeds data to Kronos model to
                  └────────┬────────┘      forecast close price direction
                           │
                           ▼
                  ┌─────────────────┐
                  │  Risk Manager   │  <-- Applies Kelly Criterion to size
                  └────────┬────────┘      positions based on model confidence
                           │
                           ▼
                  ┌─────────────────┐
                  │  Feedback Loop  │  <-- Records results in SQLite DB and
                  └─────────────────┘      resolves accuracy post-expiration
```

---

## 📂 Project Structure

```
d:\CWT prediction\
├── .env.example                     # Environment config template
├── .env                             # Active configuration file (ignored by git)
├── pyproject.toml                   # Python packaging metadata
├── requirements.txt                 # Pinned dependencies
├── README.md                        # Project documentation (this file)
│
├── cwt_prediction/                  # Core package
│   ├── __init__.py
│   ├── config.py                    # Environment configuration loader
│   ├── logging_setup.py             # JSON structured logging framework
│   ├── llm_client.py                # OpenRouter client for optional tasks
│   ├── main.py                      # Application entrypoint
│   │
│   ├── market_finder/               # Module 1: Market Finder
│   │   ├── __init__.py
│   │   ├── base.py                  # Base Finder class
│   │   ├── polymarket.py            # Polymarket Gamma API client
│   │   ├── kalshi.py                # Kalshi public API client
│   │   └── models.py                # Dataclasses representing markets
│   │
│   ├── data_fetcher/                # Module 2: Data Fetcher
│   │   ├── __init__.py
│   │   ├── base.py                  # Base Fetcher class
│   │   ├── binance_provider.py      # Binance public kline API client
│   │   ├── coingecko_provider.py    # CoinGecko fallback API client
│   │   └── models.py                # Candle dataclasses
│   │
│   ├── forecaster/                  # Module 3: Forecaster Wrapper
│   │   ├── __init__.py
│   │   ├── kronos_wrapper.py        # Forecasting pipeline using Kronos model
│   │   ├── models.py                # Dataclasses representing forecast results
│   │   └── model/                   # Local Kronos model definitions
│   │       ├── __init__.py
│   │       ├── kronos.py
│   │       └── module.py
│   │
│   ├── risk_manager/                # Module 4: Risk Sizing
│   │   ├── __init__.py
│   │   └── kelly.py                 # Pure Kelly Criterion calculator
│   │
│   ├── feedback_loop/               # Module 5: Database & Verification
│   │   ├── __init__.py
│   │   ├── models.py                # SQLite prediction records dataclass
│   │   ├── recorder.py              # SQLite recorder client (aiosqlite)
│   │   └── resolver.py              # Expired prediction outcome checker
│   │
│   └── orchestrator/                # Module 6: Orchestration
│       ├── __init__.py
│       └── pipeline.py              # Async pipeline running execution cycles
│
├── tests/                           # Testing Suite
│   ├── __init__.py
│   ├── test_kelly.py                # Unit tests for risk manager
│   ├── test_data_fetcher.py         # Mocked-network tests for fetchers
│   └── test_forecaster.py           # Mocked-tensor tests for forecaster
│
├── hermes_integration/              # Hermes Agent Integration
│   ├── SKILL.md                     # Custom skill declaration file
│   ├── CONTEXT.md                   # Agent system context description
│   └── cron_setup.md                # Scheduler setup instructions
│
├── data/
│   └── predictions.db               # SQLite Database created at runtime
└── logs/
    └── pipeline.log                 # Output log containing structured JSON
```

---

## 🛠️ Modules Breakdown

### 1. Market Finder (`market_finder/`)
Queries prediction market APIs to find open 5-minute binary price contracts.
- **Polymarket Gamma API**: Scrapes crypto binary markets. The YES contract price (0.0 to 1.0) maps to the market's implied probability.
- **Kalshi API**: Scrapes open crypto series tickers. Mid prices are calculated from order books.
- **Resilience**: If endpoints are down or hit rate limits, the module logs errors and falls back to generating mock contracts to keep the pipeline alive.

### 2. Data Fetcher (`data_fetcher/`)
Pulls asset historical candles.
- **Binance Provider**: Fetches 5-minute kline data directly from public endpoints. Requires no API keys. Also supports specific timestamp ranges for resolving historical outcomes.
- **CoinGecko Fallback**: Intercepts queries if Binance is unreachable, fetching 30-minute intervals and generating close paths.

### 3. Forecaster (`forecaster/`)
Invokes the **Kronos** model to predict upcoming movements.
- **Auto-Loading**: Lazily downloads the tokenizer (`NeoQuasar/Kronos-Tokenizer-2k`) and weights (`NeoQuasar/Kronos-mini`) on first use.
- **Monte Carlo Inference**: Runs 10 distinct forward passes under temperature sampling ($T=1.0$). If the majority of sample paths end with a close price higher than the current close, the system issues an "up" forecast, using the percentage of matching paths (e.g. 80%) as the model confidence/probability.

### 4. Risk Manager (`risk_manager/`)
Performs optimal position sizing using the Kelly Criterion.
For prediction markets where you pay $m$ to buy a $1 contract:
$$f^* = \frac{p - m}{1 - m}$$
Where:
- $p$ = Model probability of winning.
- $m$ = Market price (implied probability) of the target outcome.
- If $p \leq m$, the model has no positive edge, and the output is `0.0`.
- The fraction is capped at a configurable safety limit (default: `0.25` or 25% of bankroll).

### 5. Feedback Loop (`feedback_loop/`)
Records and verifies predictions in a local SQLite database (`data/predictions.db`).
- **Recorder**: Writes prediction records containing market ID, asset, model probability, implied probability, and recommended fraction.
- **Resolver**: Scans the database for unresolved, expired markets. It fetches the exact historical prices on Binance at both the prediction time and expiration time to verify accuracy and compute simulated PnL.

---

## ⚡ Setup & Installation

### Prerequisites
- Python 3.10+
- `uv` package manager (recommended for speed) or `pip`

### Step 1: Install Dependencies
Ensure you install the dependencies in your active Python environment:
```bash
# Using uv (fastest)
uv pip install -r requirements.txt

# Or using standard pip
pip install -r requirements.txt
```

### Step 2: Configure Environment
Copy `.env.example` to `.env` and fill in configuration variables:
```bash
cp .env.example .env
```
Ensure you provide a valid `OPENROUTER_API_KEY` (if utilizing LLM helper functionalities).

---

## 🚀 Running the System

### Run a Single Prediction Cycle
Execute the pipeline:
```bash
python -m cwt_prediction.main
```
The output logs the steps in detail and outputs a formatted JSON summary of predictions made and past outcomes resolved.

### Run tests
Run the unit and integration tests:
```bash
python -m pytest tests/ -v
```

### Review Structured JSON Logs
The pipeline outputs structured JSON data to `logs/pipeline.log`. You can parse these logs using tools like `jq` to query parameters such as confidence, direction, or PnL:
```bash
# Print decisions from JSON logs
tail -n 20 logs/pipeline.log | jq 'select(.decision != null) | {timestamp, asset, decision, confidence, kelly_fraction}'
```

---

## 🤖 Hermes Agent Integration

The project is designed to be automated and triggered by a Hermes Agent.

### 1. Install the Skill
Copy [SKILL.md](file:///d:/CWT%20prediction/hermes_integration/SKILL.md) to your local Hermes skills folder:
- **Windows**: `%USERPROFILE%\.hermes\skills\cwt-prediction\SKILL.md`
- **Linux/macOS**: `~/.hermes/skills/cwt-prediction/SKILL.md`

### 2. Schedule Cron Execution
In your Hermes CLI or gateway channel (Slack/Telegram), ask the agent to schedule the cron job:
```text
/cronjob create name="cwt-prediction" schedule="*/5 * * * *" prompt="Execute the Python prediction system in 'd:\CWT prediction' and summarize the results"
```
Hermes will run the pipeline every 5 minutes and report forecasted opportunities.
