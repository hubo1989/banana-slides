"""
OpenAI SDK implementation for image generation
"""
import logging
import base64
import re
import requests
from io import BytesIO
from typing import Optional, List
from openai import OpenAI
from PIL import Image
from .base import ImageProvider
from config import get_config

logger = logging.getLogger(__name__)


class OpenAIImageProvider(ImageProvider):
    """Image generation using OpenAI SDK (compatible with Gemini via proxy)"""
    
    def __init__(self, api_key: str, api_base: str = None, model: str = "gemini-3-pro-image-preview"):
        """
        Initialize OpenAI image provider
        
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
    
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """
        Encode PIL Image to base64 string
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()
        # Convert to RGB if necessary (e.g., RGBA images)
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate_image(
        self,
        prompt: str,
        ref_images: Optional[List[Image.Image]] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "2K"
    ) -> Optional[Image.Image]:
        """
        Generate image using OpenAI SDK
        
        Note: OpenAI format does NOT support 4K images, defaults to 1K
        
        Args:
            prompt: The image generation prompt
            ref_images: Optional list of reference images
            aspect_ratio: Image aspect ratio
            resolution: Image resolution (only 1K supported, parameter ignored)
            
        Returns:
            Generated PIL Image object, or None if failed
        """
        try:
            # Build message content
            content = []
            
            # Add reference images first (if any)
            if ref_images:
                for ref_img in ref_images:
                    base64_image = self._encode_image_to_base64(ref_img)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
            
            # Add text prompt
            content.append({"type": "text", "text": prompt})
            
            logger.debug(f"Calling OpenAI API for image generation with {len(ref_images) if ref_images else 0} reference images...")
            logger.debug(f"Config - aspect_ratio: {aspect_ratio} (resolution ignored, OpenAI format only supports 1K)")
            
            # Note: resolution is not supported in OpenAI format, only aspect_ratio via system message
            logger.info(f"[IMAGE_GEN] Calling OpenAI API - model: {self.model}, api_base: {self.client.base_url}")
            logger.info(f"[IMAGE_GEN] Request prompt length: {len(prompt)} chars, ref_images: {len(ref_images) if ref_images else 0}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"aspect_ratio={aspect_ratio}"},
                    {"role": "user", "content": content},
                ],
                modalities=["text", "image"]
            )

            logger.info(f"[IMAGE_GEN] OpenAI API call completed, response type: {type(response).__name__}")

            # Extract image from response - handle different response formats

            # 首先处理 Gemini REST API 格式（某些代理返回的格式）
            # response.choices 可能是 None，需要从原始响应中提取
            raw_response = response.model_dump() if hasattr(response, 'model_dump') else {}
            if raw_response is None:
                raw_response = {}

            # ===== DEBUG: 打印完整原始响应 =====
            logger.info(f"[IMAGE_GEN] ===== RAW RESPONSE START =====")
            logger.info(f"[IMAGE_GEN] Response dump keys: {list(raw_response.keys()) if isinstance(raw_response, dict) else 'N/A'}")
            logger.info(f"[IMAGE_GEN] Raw response (truncated): {str(raw_response)[:2000]}")
            if 'choices' in raw_response:
                choices = raw_response.get('choices') or []
                logger.info(f"[IMAGE_GEN] Choices count: {len(choices)}")
                if choices:
                    first_choice = choices[0]
                    logger.info(f"[IMAGE_GEN] First choice keys: {list(first_choice.keys()) if isinstance(first_choice, dict) else 'N/A'}")
                    if 'message' in first_choice:
                        logger.info(f"[IMAGE_GEN] Message keys: {list(first_choice['message'].keys()) if isinstance(first_choice['message'], dict) else 'N/A'}")
            logger.info(f"[IMAGE_GEN] ===== RAW RESPONSE END =====")
            # ===== DEBUG END =====

            # 处理 Gemini REST API 格式: {"response": {"candidates": [{"content": {"parts": [...]}}]}}
            if "response" in raw_response and "candidates" in raw_response["response"]:
                candidates = raw_response["response"]["candidates"]
                if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                    for part in candidates[0]["content"]["parts"]:
                        if "inline_data" in part:
                            image_data = base64.b64decode(part["inline_data"]["data"])
                            image = Image.open(BytesIO(image_data))
                            logger.info(f"[IMAGE_GEN] Successfully extracted image from Gemini inline_data: {image.size}, {image.mode}")
                            return image
                        if "text" in part:
                            text = part["text"]
                            logger.info(f"[IMAGE_GEN] Gemini response text: {text[:100] if len(text) > 100 else text}")

                            # Check if text contains Markdown image with base64 data URL
                            markdown_data_pattern = r'!\[.*?\]\((data:image/[^;]+;base64,[A-Za-z0-9+/=]+)\)'
                            markdown_data_matches = re.findall(markdown_data_pattern, text)
                            if markdown_data_matches:
                                data_url = markdown_data_matches[0]
                                logger.info(f"[IMAGE_GEN] Found Markdown data URL in Gemini response, length: {len(data_url)}")
                                try:
                                    base64_data = data_url.split(',', 1)[1]
                                    image_data = base64.b64decode(base64_data)
                                    image = Image.open(BytesIO(image_data))
                                    logger.info(f"[IMAGE_GEN] Successfully extracted base64 image from Gemini Markdown: {image.size}, {image.mode}")
                                    return image
                                except Exception as decode_error:
                                    logger.warning(f"[IMAGE_GEN] Failed to decode base64 image from Gemini Markdown: {decode_error}")

            # 处理标准 OpenAI 格式
            if not response.choices:
                logger.error(f"[IMAGE_GEN] No choices in response! Raw response: {raw_response}")
                raise ValueError(f"OpenAI API returned no choices. Raw response: {raw_response}")

            message = response.choices[0].message

            # Debug: log available attributes
            logger.info(f"[IMAGE_GEN] Message attributes: {dir(message)}")
            logger.info(f"[IMAGE_GEN] Message.content type: {type(message.content)}, value: {str(message.content)[:500] if message.content else 'None'}")
            
            # Try multi_mod_content first (custom format from some proxies)
            if hasattr(message, 'multi_mod_content') and message.multi_mod_content:
                parts = message.multi_mod_content
                for part in parts:
                    if "text" in part:
                        logger.debug(f"Response text: {part['text'][:100] if len(part['text']) > 100 else part['text']}")
                    if "inline_data" in part:
                        image_data = base64.b64decode(part["inline_data"]["data"])
                        image = Image.open(BytesIO(image_data))
                        logger.debug(f"Successfully extracted image: {image.size}, {image.mode}")
                        return image
            
            # Try standard OpenAI content format (list of content parts)
            if hasattr(message, 'content') and message.content:
                # If content is a list (multimodal response)
                if isinstance(message.content, list):
                    for part in message.content:
                        if isinstance(part, dict):
                            # Handle image_url type
                            if part.get('type') == 'image_url':
                                image_url = part.get('image_url', {}).get('url', '')
                                if image_url.startswith('data:image'):
                                    # Extract base64 data from data URL
                                    base64_data = image_url.split(',', 1)[1]
                                    image_data = base64.b64decode(base64_data)
                                    image = Image.open(BytesIO(image_data))
                                    logger.debug(f"Successfully extracted image from content: {image.size}, {image.mode}")
                                    return image
                            # Handle text type
                            elif part.get('type') == 'text':
                                text = part.get('text', '')
                                if text:
                                    logger.debug(f"Response text: {text[:100] if len(text) > 100 else text}")
                        elif hasattr(part, 'type'):
                            # Handle as object with attributes
                            if part.type == 'image_url':
                                image_url = getattr(part, 'image_url', {})
                                if isinstance(image_url, dict):
                                    url = image_url.get('url', '')
                                else:
                                    url = getattr(image_url, 'url', '')
                                if url.startswith('data:image'):
                                    base64_data = url.split(',', 1)[1]
                                    image_data = base64.b64decode(base64_data)
                                    image = Image.open(BytesIO(image_data))
                                    logger.debug(f"Successfully extracted image from content object: {image.size}, {image.mode}")
                                    return image
                # If content is a string, try to extract image from it
                elif isinstance(message.content, str):
                    content_str = message.content
                    logger.debug(f"Response content (string): {content_str[:200] if len(content_str) > 200 else content_str}")

                    # Try to extract Markdown image URL with data:image prefix (e.g., ![...](data:image/jpeg;base64,...))
                    markdown_data_pattern = r'!\[.*?\]\((data:image/[^;]+;base64,[A-Za-z0-9+/=]+)\)'
                    markdown_data_matches = re.findall(markdown_data_pattern, content_str)
                    if markdown_data_matches:
                        data_url = markdown_data_matches[0]
                        logger.info(f"[IMAGE_GEN] Found Markdown data URL, length: {len(data_url)}")
                        try:
                            # Extract base64 data from data URL
                            base64_data = data_url.split(',', 1)[1]
                            image_data = base64.b64decode(base64_data)
                            image = Image.open(BytesIO(image_data))
                            logger.info(f"[IMAGE_GEN] Successfully extracted base64 image from Markdown: {image.size}, {image.mode}")
                            return image
                        except Exception as decode_error:
                            logger.warning(f"Failed to decode base64 image from Markdown: {decode_error}")

                    # Try to extract Markdown image URL with http/https prefix: ![...](url)
                    markdown_http_pattern = r'!\[.*?\]\((https?://[^\s\)]+)\)'
                    markdown_http_matches = re.findall(markdown_http_pattern, content_str)
                    if markdown_http_matches:
                        image_url = markdown_http_matches[0]  # Use the first image URL found
                        logger.debug(f"Found Markdown http image URL: {image_url}")
                        try:
                            response = requests.get(image_url, timeout=30, stream=True)
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content))
                            image.load()  # Ensure image is fully loaded
                            logger.debug(f"Successfully downloaded image from Markdown URL: {image.size}, {image.mode}")
                            return image
                        except Exception as download_error:
                            logger.warning(f"Failed to download image from Markdown URL: {download_error}")
                    
                    # Try to extract plain URL (not in Markdown format)
                    url_pattern = r'(https?://[^\s\)\]]+\.(?:png|jpg|jpeg|gif|webp|bmp)(?:\?[^\s\)\]]*)?)'
                    url_matches = re.findall(url_pattern, content_str, re.IGNORECASE)
                    if url_matches:
                        image_url = url_matches[0]
                        logger.debug(f"Found plain image URL: {image_url}")
                        try:
                            response = requests.get(image_url, timeout=30, stream=True)
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content))
                            image.load()
                            logger.debug(f"Successfully downloaded image from plain URL: {image.size}, {image.mode}")
                            return image
                        except Exception as download_error:
                            logger.warning(f"Failed to download image from plain URL: {download_error}")
                    
                    # Try to extract base64 data URL from string
                    base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                    base64_matches = re.findall(base64_pattern, content_str)
                    if base64_matches:
                        base64_data = base64_matches[0]
                        logger.debug(f"Found base64 image data in string")
                        try:
                            image_data = base64.b64decode(base64_data)
                            image = Image.open(BytesIO(image_data))
                            logger.debug(f"Successfully extracted base64 image from string: {image.size}, {image.mode}")
                            return image
                        except Exception as decode_error:
                            logger.warning(f"Failed to decode base64 image from string: {decode_error}")
            
            # Log raw response for debugging
            logger.error(f"[IMAGE_GEN] ===== IMAGE EXTRACTION FAILED =====")
            logger.error(f"[IMAGE_GEN] Unable to extract image. Raw message type: {type(message)}")
            logger.error(f"[IMAGE_GEN] Message content type: {type(getattr(message, 'content', None))}")
            logger.error(f"[IMAGE_GEN] Message content: {getattr(message, 'content', 'N/A')}")
            logger.error(f"[IMAGE_GEN] Full raw_response: {raw_response}")
            logger.error(f"[IMAGE_GEN] ===== END DEBUG INFO =====")

            raise ValueError("No valid multimodal response received from OpenAI API")

        except Exception as e:
            error_detail = f"Error generating image with OpenAI (model={self.model}): {type(e).__name__}: {str(e)}"
            logger.error(f"[IMAGE_GEN] {error_detail}", exc_info=True)
            raise Exception(error_detail) from e
