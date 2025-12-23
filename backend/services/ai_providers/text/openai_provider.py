"""
OpenAI SDK implementation for text generation
"""
import logging
from openai import OpenAI
from .base import TextProvider
from config import get_config

logger = logging.getLogger(__name__)


class OpenAITextProvider(TextProvider):
    """Text generation using OpenAI SDK (compatible with Gemini via proxy)"""
    
    def __init__(self, api_key: str, api_base: str = None, model: str = "gemini-3-flash-preview"):
        """
        Initialize OpenAI text provider
        
        Args:
            api_key: API key
            api_base: API base URL (e.g., https://aihubmix.com/v1)
            model: Model name to use
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=get_config().OPENAI_TIMEOUT,  # set timeout from config
            max_retries=get_config().OPENAI_MAX_RETRIES  # set max retries from config
        )
        self.model = model
    
    def generate_text(self, prompt: str, thinking_budget: int = 1000) -> str:
        """
        Generate text using OpenAI SDK

        Args:
            prompt: The input prompt
            thinking_budget: Not used in OpenAI format, kept for interface compatibility

        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # 处理标准 OpenAI 格式
        if response.choices:
            return response.choices[0].message.content

        # 处理 Gemini REST API 格式（某些代理返回的格式）
        # 格式: {"response": {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}}
        raw_response = response.model_dump()
        if "response" in raw_response and "candidates" in raw_response["response"]:
            candidates = raw_response["response"]["candidates"]
            if candidates and candidates[0].get("content", {}).get("parts"):
                return candidates[0]["content"]["parts"][0]["text"]

        # 处理直接 candidates 格式
        if "candidates" in raw_response:
            candidates = raw_response["candidates"]
            if candidates and candidates[0].get("content", {}).get("parts"):
                return candidates[0]["content"]["parts"][0]["text"]

        logger.error(f"Unknown response format: {raw_response}")
        raise ValueError(f"无法解析响应格式: {raw_response}")
