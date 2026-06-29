import asyncio
import json
import time
from typing import List
from loguru import logger
from tools.registry import registry

ALLOWED_TOOLS = ["search_polymarket", "search_kalshi"]

class HermesSearchAgent:
    """Hermes Search Agent - restricted to Polymarket and Kalshi search tools."""

    @property
    def allowed_tools(self) -> List[str]:
        return list(ALLOWED_TOOLS)

    async def execute(self, limit_per_platform: int = 10, per_call_timeout: float = 25.0) -> List[dict]:
        logger.info("HermesSearchAgent: Started")
        start = time.time()
        tasks = []
        try:
            loop = asyncio.get_running_loop()
            async def _dispatch(platform: str):
                return await asyncio.wait_for(
                    loop.run_in_executor(None, registry.dispatch, platform, {"limit": limit_per_platform}),
                    timeout=per_call_timeout,
                )

            poly_task = asyncio.create_task(_dispatch("search_polymarket"))
            kalshi_task = asyncio.create_task(_dispatch("search_kalshi"))
            tasks = [poly_task, kalshi_task]

            poly_result_str, kalshi_result_str = await asyncio.gather(
                poly_task, kalshi_task, return_exceptions=True,
            )

            markets = []
            for name, result_str in [("polymarket", poly_result_str), ("kalshi", kalshi_result_str)]:
                if isinstance(result_str, Exception):
                    logger.warning(f"HermesSearchAgent: {name} failed: {result_str}")
                    continue
                try:
                    result = json.loads(result_str) if isinstance(result_str, str) else {}
                    if "error" not in result:
                        markets.extend(result.get("markets", []))
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"HermesSearchAgent: {name} returned unparseable result")

            elapsed = time.time() - start
            logger.info(f"HermesSearchAgent: Finished in {elapsed:.2f}s, found {len(markets)} markets total")
            return markets
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"HermesSearchAgent: Error after {elapsed:.2f}s: {e}")
            raise
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
