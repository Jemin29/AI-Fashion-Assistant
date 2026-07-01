from __future__ import annotations

import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger("rate_limit")

def get_client_ip(request: Request) -> str:
    """Extracts client IP address respecting reverse proxy headers."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    if request.client:
        return request.client.host
    return "127.0.0.1"

# Central Limiter instance utilizing client IP addresses as key identifiers
limiter = Limiter(
    key_func=get_client_ip,
    default_limits=["100/minute"]  # Generous global default limit
)


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Custom exception handler returning standardized error JSON for 429 Too Many Requests."""
    logger.warning(f"Rate limit exceeded on endpoint {request.url.path} from client {request.client.host if request.client else 'unknown'}")
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": f"Rate limit exceeded: {exc.detail}. Please slow down."
            }
        }
    )
