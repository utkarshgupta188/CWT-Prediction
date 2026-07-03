import aiosqlite
import logging
import os
from datetime import datetime, timezone
from typing import List
from .models import PredictionRecord

logger = logging.getLogger("cwt_prediction.feedback_loop.recorder")

class PredictionRecorder:
    def __init__(self, db_path: str = "data/predictions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

    async def init_db(self):
        """Initializes the SQLite database and creates the predictions table if it doesn't exist."""
        logger.info(f"Initializing database at: {self.db_path}")
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    direction_predicted TEXT NOT NULL,
                    model_prob REAL NOT NULL,
                    market_implied_prob REAL NOT NULL,
                    kelly_fraction REAL NOT NULL,
                    expiry TEXT NOT NULL,
                    resolved INTEGER NOT NULL DEFAULT 0,
                    actual_direction TEXT,
                    pnl_if_bet REAL,
                    resolve_timestamp TEXT
                )
            """)
            await db.commit()

    async def record_prediction(self, r: PredictionRecord) -> int:
        """Inserts a new prediction record into the database."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO predictions (
                    timestamp, asset, market_id, platform, direction_predicted,
                    model_prob, market_implied_prob, kelly_fraction, expiry, resolved
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r.timestamp.isoformat(),
                r.asset,
                r.market_id,
                r.platform,
                r.direction_predicted,
                r.model_prob,
                r.market_implied_prob,
                r.kelly_fraction,
                r.expiry.isoformat(),
                1 if r.resolved else 0
            ))
            await db.commit()
            record_id = cursor.lastrowid
            logger.info(f"Recorded prediction ID {record_id} for {r.asset} on {r.platform}")
            return record_id

    async def get_unresolved(self) -> List[PredictionRecord]:
        """Fetches all unresolved predictions that have passed their expiry time."""
        now_str = datetime.now(timezone.utc).isoformat()
        unresolved = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # We fetch where resolved = 0 and expiry <= current time
            async with db.execute(
                "SELECT * FROM predictions WHERE resolved = 0 AND expiry <= ?", (now_str,)
            ) as cursor:
                async for row in cursor:
                    unresolved.append(PredictionRecord(
                        id=row["id"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        asset=row["asset"],
                        market_id=row["market_id"],
                        platform=row["platform"],
                        direction_predicted=row["direction_predicted"],
                        model_prob=row["model_prob"],
                        market_implied_prob=row["market_implied_prob"],
                        kelly_fraction=row["kelly_fraction"],
                        expiry=datetime.fromisoformat(row["expiry"]),
                        resolved=bool(row["resolved"]),
                        actual_direction=row["actual_direction"],
                        pnl_if_bet=row["pnl_if_bet"],
                        resolve_timestamp=datetime.fromisoformat(row["resolve_timestamp"]) if row["resolve_timestamp"] else None
                    ))
        return unresolved

    async def resolve_prediction(self, record_id: int, actual_direction: str, pnl: float):
        """Updates a prediction record as resolved with actual outcome and PnL."""
        now_str = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE predictions 
                SET resolved = 1, actual_direction = ?, pnl_if_bet = ?, resolve_timestamp = ?
                WHERE id = ?
            """, (actual_direction, pnl, now_str, record_id))
            await db.commit()
            logger.info(f"Resolved prediction ID {record_id}: actual={actual_direction}, pnl={pnl:.4f}")
            
    async def get_stats(self) -> dict:
        """Computes feedback/accuracy statistics of resolved predictions."""
        stats = {
            "total_resolved": 0,
            "correct_predictions": 0,
            "accuracy": 0.0,
            "total_pnl": 0.0
        }
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM predictions WHERE resolved = 1") as cursor:
                async for row in cursor:
                    stats["total_resolved"] += 1
                    if row["direction_predicted"] == row["actual_direction"]:
                        stats["correct_predictions"] += 1
                    stats["total_pnl"] += row["pnl_if_bet"] or 0.0

        if stats["total_resolved"] > 0:
            stats["accuracy"] = stats["correct_predictions"] / stats["total_resolved"]
            
        return stats
