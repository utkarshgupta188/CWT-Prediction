import asyncio
import sys
import json
import logging
from cwt_prediction.config import load_config
from cwt_prediction.logging_setup import setup_logging
from cwt_prediction.orchestrator.pipeline import run_cycle

logger = logging.getLogger("cwt_prediction.main")

async def main():
    # 1. Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 2. Setup logging
    setup_logging(log_file=config.log_file, log_level=logging.INFO)
    logger.info("CWT Prediction System CLI started.")
    
    # 3. Check for OpenRouter Key configuration warning
    if not config.openrouter_api_key:
        logger.warning(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Pipeline will run with fallback rules and mock inputs if APIs are limited."
        )
        
    # 4. Run prediction cycle
    try:
        summary = await run_cycle(config)
        # Print summary as pretty printed JSON to stdout for the user/Hermes tool caller to parse easily
        print("\n--- CYCLE SUMMARY ---")
        print(json.dumps(summary, indent=2))
        print("---------------------\n")
    except Exception as e:
        logger.critical(f"Pipeline crashed during execution: {e}", exc_info=True)
        sys.exit(1)
        
    logger.info("Cycle finished successfully.")

if __name__ == "__main__":
    # Use standard selector event loop on Windows to avoid issues with some async subprocesses/sockets
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
