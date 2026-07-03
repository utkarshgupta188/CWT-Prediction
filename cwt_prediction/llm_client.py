import logging
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger("cwt_prediction.llm_client")

class OpenRouterClient:
    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.1-8b-instruct:free"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    async def get_decision(self, prompt: str) -> Optional[str]:
        """
        Sends a prompt to OpenRouter and returns the content.
        Gracefully returns None if API is missing or fails.
        """
        if not self.api_key:
            logger.warning("OpenRouter API key is not configured. Skipping LLM call.")
            return None
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/nousresearch/hermes-agent",
            "X-Title": "CWT Prediction System"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(self.api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    logger.warning(
                        f"OpenRouter API returned error {response.status_code}: {response.text}"
                    )
                    return None
        except Exception as e:
            logger.error(f"OpenRouter API exception: {e}", exc_info=True)
            return None
