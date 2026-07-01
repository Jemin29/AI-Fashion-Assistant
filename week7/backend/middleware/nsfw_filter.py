from __future__ import annotations

import base64
import io
import json
import logging
from PIL import Image
from fastapi import Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from week7.backend.configs.config import get_settings

logger = logging.getLogger("nsfw_filter")


class NSFWFilter:
    """Detects unsafe/NSFW content using the Falconsai/nsfw_image_detection model or mock fallback."""

    def __init__(self, mock_mode: bool = False) -> None:
        self.mock_mode = mock_mode
        self._classifier = None

    def scan_image(self, img_b64: str) -> bool:
        """Scan a base64 encoded image. Returns True if SAFE, False if UNSAFE (NSFW)."""
        if not img_b64:
            return True

        # Mock mode checks for trigger signatures in base64 string
        if self.mock_mode:
            if "unsafe" in img_b64.lower() or "nsfw" in img_b64.lower() or "nude" in img_b64.lower():
                logger.warning("Mock NSFW Filter: Flagged unsafe image signature.")
                return False
            return True

        try:
            # Strip base64 metadata header if present
            if "," in img_b64:
                img_b64 = img_b64.split(",")[1]
            img_data = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")

            # Lazily initialize Hugging Face pipeline
            if self._classifier is None:
                from transformers import pipeline
                logger.info("Loading Hugging Face 'Falconsai/nsfw_image_detection' model...")
                self._classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

            results = self._classifier(img)
            # Inspect output classifications: e.g., [{'label': 'nsfw', 'score': 0.95}, ...]
            for res in results:
                if res.get("label") == "nsfw" and res.get("score", 0.0) > 0.5:
                    logger.warning(f"NSFW Content Detected! Model Score: {res.get('score'):.4f}")
                    return False
            return True
        except Exception as exc:
            logger.error(f"Failed to scan image with NSFW classifier: {str(exc)}. Falling back to SAFE.")
            return True


class NSFWFilterMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware automatically scanning outgoing JSON responses for generated image payloads."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        content_type = response.headers.get("content-type", "")

        # Only inspect JSON responses
        if "application/json" not in content_type:
            return response

        # Read the streaming body safely
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        try:
            data = json.loads(response_body.decode("utf-8"))
            image_b64 = None

            # Detect generated images in JSON structures
            if isinstance(data, dict):
                if "image" in data and isinstance(data["image"], str):
                    image_b64 = data["image"]
                elif "result" in data and isinstance(data["result"], dict) and "image" in data["result"]:
                    image_b64 = data["result"]["image"]

            # If image payload is found, run NSFW safety validation
            if image_b64:
                from week7.backend.api.dependencies import get_nsfw_filter
                nsfw_filter = get_nsfw_filter()
                if not nsfw_filter.scan_image(image_b64):
                    logger.warning(f"Blocking HTTP response: NSFW content detected in response payload.")
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={
                            "success": False,
                            "error": {
                                "code": "UNSAFE_CONTENT_DETECTED",
                                "message": "Generated image was flagged by our safety pipeline as containing unsafe content."
                            }
                        }
                    )
        except Exception as exc:
            logger.error(f"Error executing NSFW filter middleware: {str(exc)}")

        # Re-create response with reconstituted body
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
