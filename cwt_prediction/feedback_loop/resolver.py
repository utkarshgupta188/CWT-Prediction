import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional
from .recorder import PredictionRecorder
from ..data_fetcher.binance_provider import BinanceFetcher

logger = logging.getLogger("cwt_prediction.feedback_loop.resolver")

class PredictionResolver:
    def __init__(self, recorder: PredictionRecorder, fetcher: Optional[BinanceFetcher] = None):
        self.recorder = recorder
        self.fetcher = fetcher or BinanceFetcher()

    async def resolve_all(self) -> int:
        """
        Fetches all unresolved, expired predictions and resolves them.
        Returns the number of successfully resolved predictions.
        """
        unresolved = await self.recorder.get_unresolved()
        if not unresolved:
            logger.info("No unresolved predictions found to resolve.")
            return 0

        logger.info(f"Attempting to resolve {len(unresolved)} predictions...")
        resolved_count = 0

        for r in unresolved:
            try:
                # 1. Fetch historical prices around the prediction window
                # We need prices at creation time (timestamp) and expiry time
                # Fetch a window starting 10 min before creation and ending 10 min after expiry
                start_window = r.timestamp - timedelta(minutes=10)
                end_window = r.expiry + timedelta(minutes=10)

                # Fetch klines
                df = await self.fetcher.fetch_history(
                    asset=r.asset,
                    limit=100,
                    start_time=start_window,
                    end_time=end_window
                )

                if df.empty:
                    logger.warning(f"No price data found for prediction ID {r.id} between {start_window} and {end_window}")
                    continue

                # 2. Find closest rows to creation time and expiry time
                # We do this by finding the row with minimum absolute time difference
                df['dt_diff_creation'] = (df['timestamps'] - r.timestamp).abs()
                df['dt_diff_expiry'] = (df['timestamps'] - r.expiry).abs()

                row_creation = df.loc[df['dt_diff_creation'].idxmin()]
                row_expiry = df.loc[df['dt_diff_expiry'].idxmin()]

                # Make sure the closest bar is within a reasonable limit (e.g. 10 minutes)
                max_diff = timedelta(minutes=10)
                if row_creation['dt_diff_creation'] > max_diff or row_expiry['dt_diff_expiry'] > max_diff:
                    logger.warning(
                        f"Price data timestamps are too far from prediction times for ID {r.id}. "
                        f"Creation diff: {row_creation['dt_diff_creation']}, Expiry diff: {row_expiry['dt_diff_expiry']}"
                    )
                    continue

                price_creation = float(row_creation['close'])
                price_expiry = float(row_expiry['close'])

                # 3. Determine actual direction
                actual_direction = "up" if price_expiry > price_creation else "down"

                # 4. Compute PnL
                # For YES options:
                # If predicted YES/UP and it went UP, we win!
                # If predicted NO/DOWN and it went DOWN, we win!
                # If we win, PnL = kelly_fraction * net_odds
                # If we lose, PnL = -kelly_fraction
                is_win = (r.direction_predicted == actual_direction)

                if is_win:
                    # net payout ratio = (1 - m) / m
                    m = r.market_implied_prob
                    # Avoid division by zero
                    if m <= 0.0:
                        m = 0.01
                    elif m >= 1.0:
                        m = 0.99
                    net_odds = (1.0 - m) / m
                    pnl = r.kelly_fraction * net_odds
                else:
                    pnl = -r.kelly_fraction

                # 5. Save to DB
                await self.recorder.resolve_prediction(
                    record_id=r.id,
                    actual_direction=actual_direction,
                    pnl=pnl
                )
                
                logger.info(
                    f"Prediction ID {r.id} resolved: asset={r.asset}, "
                    f"creation_price={price_creation:.2f}, expiry_price={price_expiry:.2f}, "
                    f"predicted={r.direction_predicted}, actual={actual_direction}, win={is_win}, pnl={pnl:.4f}"
                )
                resolved_count += 1

            except Exception as e:
                logger.error(f"Failed to resolve prediction ID {r.id}: {e}", exc_info=True)
                continue

        return resolved_count
