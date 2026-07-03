import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from ..config import Config
from ..market_finder.polymarket import PolymarketFinder
from ..market_finder.kalshi import KalshiFinder
from ..market_finder.models import PredictionMarket
from ..data_fetcher.binance_provider import BinanceFetcher
from ..data_fetcher.coingecko_provider import CoinGeckoFetcher
from ..forecaster.kronos_wrapper import KronosForecaster
from ..risk_manager.kelly import kelly_fraction
from ..feedback_loop.recorder import PredictionRecorder
from ..feedback_loop.resolver import PredictionResolver
from ..feedback_loop.models import PredictionRecord

logger = logging.getLogger("cwt_prediction.orchestrator.pipeline")

async def run_cycle(config: Config) -> Dict[str, Any]:
    """Runs a single execution cycle of the prediction pipeline."""
    logger.info("Starting a new prediction cycle...")
    
    # Initialize recorder and resolver
    recorder = PredictionRecorder(db_path=config.db_path)
    await recorder.init_db()
    
    resolver = PredictionResolver(recorder=recorder)
    
    # 1. Resolve past expired predictions
    logger.info("Checking for expired, unresolved predictions to resolve...")
    resolved_count = await resolver.resolve_all()
    logger.info(f"Resolved {resolved_count} past predictions.")
    
    # 2. Find active prediction markets
    logger.info("Querying Polymarket and Kalshi for active BTC/ETH prediction markets...")
    poly_finder = PolymarketFinder()
    kalshi_finder = KalshiFinder()
    
    markets = []
    
    # Try finding on Polymarket
    try:
        poly_markets = await poly_finder.find_markets()
        logger.info(f"Found {len(poly_markets)} markets on Polymarket.")
        markets.extend(poly_markets)
    except Exception as e:
        logger.error(f"Polymarket finder failed: {e}")
        
    # Try finding on Kalshi
    try:
        kalshi_markets = await kalshi_finder.find_markets()
        logger.info(f"Found {len(kalshi_markets)} markets on Kalshi.")
        markets.extend(kalshi_markets)
    except Exception as e:
        logger.error(f"Kalshi finder failed: {e}")
        
    # Mock fallback if no markets found (useful for out-of-the-box local runs/tests)
    if not markets:
        logger.warning("No active markets found on APIs. Generating mock crypto prediction markets for testing/demonstration...")
        now = datetime.now(timezone.utc)
        markets = [
            PredictionMarket(
                id="mock-btc-up-5m",
                platform="polymarket",
                asset="BTC",
                question="Will Bitcoin close above current price in 30 minutes?",
                expiry=now + timedelta(minutes=30),
                implied_prob_yes=0.52,
                url="https://polymarket.com/market/mock-btc-up",
                ticker="MOCK-BTC-UP"
            ),
            PredictionMarket(
                id="mock-eth-down-5m",
                platform="kalshi",
                asset="ETH",
                question="Will Ethereum close below current price in 30 minutes?",
                expiry=now + timedelta(minutes=30),
                implied_prob_yes=0.48, # YES probability (meaning NO probability is 0.52)
                url="https://kalshi.com/markets/mock-eth-down",
                ticker="MOCK-ETH-DOWN"
            )
        ]
        
    # 3. Initialize fetchers and forecaster
    binance_fetcher = BinanceFetcher()
    gecko_fetcher = CoinGeckoFetcher()
    
    forecaster = KronosForecaster(
        model_name=config.kronos_model,
        tokenizer_name=config.kronos_tokenizer
    )
    
    decisions = []
    
    # 4. Run forecasting and risk sizing for each market
    for market in markets:
        logger.info(f"Processing market {market.id} ({market.asset}) on {market.platform}...")
        
        # Pull historical OHLCV data
        ohlcv_df = None
        try:
            logger.info(f"Fetching historical data for {market.asset} from Binance...")
            ohlcv_df = await binance_fetcher.fetch_history(asset=market.asset, limit=config.lookback_bars)
        except Exception as e:
            logger.warning(f"Binance fetch failed for {market.asset}: {e}. Trying CoinGecko fallback...")
            try:
                ohlcv_df = await gecko_fetcher.fetch_history(asset=market.asset, limit=config.lookback_bars)
            except Exception as ex:
                logger.error(f"CoinGecko fallback also failed for {market.asset}: {ex}. Skipping market.")
                continue
                
        if ohlcv_df is None or ohlcv_df.empty:
            logger.warning(f"No OHLCV history could be fetched for {market.asset}. Skipping market.")
            continue
            
        # Run Kronos forecast
        try:
            forecast_res = forecaster.forecast(ohlcv_df, horizon=config.forecast_horizon)
        except Exception as e:
            logger.error(f"Forecasting failed for {market.asset} on market {market.id}: {e}", exc_info=True)
            continue
            
        # Sizing position with Kelly Criterion
        # If model predicts "up" (YES option): model win prob is forecast_res.probability, market win prob is implied_prob_yes
        # If model predicts "down" (NO option): model win prob is forecast_res.probability, market win prob is (1 - implied_prob_yes)
        if forecast_res.direction == "up":
            market_prob = market.implied_prob_yes
        else:
            market_prob = 1.0 - market.implied_prob_yes
            
        try:
            kelly_frac = kelly_fraction(
                model_prob=forecast_res.probability,
                market_implied_prob=market_prob,
                max_fraction=config.kelly_max_fraction
            )
        except Exception as e:
            logger.error(f"Kelly sizing calculation failed for market {market.id}: {e}")
            kelly_frac = 0.0
            
        # Log structured decision
        # We pass extra parameters to the structured logging JSON record
        logger.info(
            f"Decision for {market.asset} on {market.platform}: "
            f"Forecast={forecast_res.direction} (prob={forecast_res.probability:.2f}), "
            f"Market prob={market_prob:.2f}, Kelly fraction={kelly_frac:.4f}",
            extra={
                "decision": forecast_res.direction,
                "confidence": forecast_res.probability,
                "kelly_fraction": kelly_frac,
                "asset": market.asset,
                "market_id": market.id,
                "platform": market.platform
            }
        )
        
        # 5. Record the prediction in the feedback loop database
        record = PredictionRecord(
            id=None,
            timestamp=datetime.now(timezone.utc),
            asset=market.asset,
            market_id=market.id,
            platform=market.platform,
            direction_predicted=forecast_res.direction,
            model_prob=forecast_res.probability,
            market_implied_prob=market_prob,
            kelly_fraction=kelly_frac,
            expiry=market.expiry
        )
        
        try:
            await recorder.record_prediction(record)
        except Exception as e:
            logger.error(f"Failed to record prediction in SQLite DB: {e}")
            
        decisions.append({
            "market_id": market.id,
            "asset": market.asset,
            "platform": market.platform,
            "question": market.question,
            "direction": forecast_res.direction,
            "confidence": forecast_res.probability,
            "market_prob": market_prob,
            "kelly_fraction": kelly_frac,
            "expiry": market.expiry.isoformat()
        })
        
    # Get current resolution stats
    stats = await recorder.get_stats()
    
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decisions_made": len(decisions),
        "decisions": decisions,
        "resolved_past_count": resolved_count,
        "stats": stats
    }
    
    logger.info(
        f"Cycle complete. Decisions made: {len(decisions)}, Resolved: {resolved_count}, "
        f"Historical Accuracy: {stats['accuracy']*100:.1f}% (PnL: {stats['total_pnl']:.4f})"
    )
    
    return summary
