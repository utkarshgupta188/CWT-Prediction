import pytest
import respx
import httpx
import pandas as pd
from datetime import datetime, timezone
from cwt_prediction.data_fetcher.binance_provider import BinanceFetcher
from cwt_prediction.data_fetcher.coingecko_provider import CoinGeckoFetcher

@pytest.mark.asyncio
@respx.mock
async def test_binance_fetcher_success():
    fetcher = BinanceFetcher()
    # Mock Binance API kline response
    # format: [ [open_time, open, high, low, close, volume, close_time, quote_asset_vol, ...] ]
    mock_response = [
        [1609459200000, "29000.0", "29500.0", "28900.0", "29300.0", "100.0", 1609459499999, "2920000.0", 50, "50.0", "1460000.0", "0"]
    ]
    respx.get("https://api.binance.com/api/v3/klines").respond(json=mock_response)

    df = await fetcher.fetch_history("BTC", limit=1)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    assert df["open"].iloc[0] == 29000.0
    assert df["high"].iloc[0] == 29500.0
    assert df["low"].iloc[0] == 28900.0
    assert df["close"].iloc[0] == 29300.0
    assert df["volume"].iloc[0] == 100.0
    assert df["amount"].iloc[0] == 2920000.0
    assert isinstance(df["timestamps"].iloc[0], datetime)

@pytest.mark.asyncio
@respx.mock
async def test_binance_fetcher_rate_limit_failure():
    fetcher = BinanceFetcher()
    respx.get("https://api.binance.com/api/v3/klines").respond(status_code=429)

    with pytest.raises(httpx.HTTPStatusError):
        await fetcher.fetch_history("BTC")

@pytest.mark.asyncio
@respx.mock
async def test_coingecko_fetcher_success():
    fetcher = CoinGeckoFetcher()
    # Mock CoinGecko ohlc response format: [ [timestamp_ms, open, high, low, close] ]
    mock_response = [
        [1609459200000, 29000.0, 29500.0, 28900.0, 29300.0]
    ]
    respx.get("https://api.coingecko.com/api/v3/coins/bitcoin/ohlc").respond(json=mock_response)

    df = await fetcher.fetch_history("BTC", limit=1)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    assert df["open"].iloc[0] == 29000.0
    assert df["volume"].iloc[0] == 0.0
    assert df["amount"].iloc[0] == 0.0
