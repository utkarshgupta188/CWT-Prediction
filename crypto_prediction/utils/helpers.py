import asyncio
import httpx
from loguru import logger
from crypto_prediction.schemas.config import settings

setup_logger = logger.bind(name="helpers")

async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    **kwargs
) -> httpx.Response:
    """
    Perform an HTTP request with exponential backoff for:
    - Status codes: 429, 500, 502, 503
    - Network timeouts and ConnectionErrors
    """
    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.request(method, url, **kwargs)
            if response.status_code in [429, 500, 502, 503]:
                logger.warning(
                    f"HTTP {response.status_code} received from {url}. Attempt {attempt}/{max_retries}."
                )
                if attempt == max_retries:
                    response.raise_for_status()
                # If 429, check Retry-After header
                sleep_time = backoff
                if response.status_code == 429 and "Retry-After" in response.headers:
                    try:
                        sleep_time = float(response.headers["Retry-After"])
                    except ValueError:
                        pass
                await asyncio.sleep(sleep_time)
                backoff *= 2
                continue
            
            response.raise_for_status()
            return response
            
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            logger.warning(
                f"Network error/timeout on {url}: {str(e)}. Attempt {attempt}/{max_retries}."
            )
            if attempt == max_retries:
                raise e
            await asyncio.sleep(backoff)
            backoff *= 2
