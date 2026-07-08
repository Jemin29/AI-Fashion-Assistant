from __future__ import annotations

import os
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
from week7.backend.configs.config import get_settings

settings = get_settings()

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing standard secure HTTP headers for security compliance."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Clickjacking defense
        response.headers["X-Frame-Options"] = "DENY"
        
        # MIME-sniffing defense
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Legacy XSS mitigation
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Content Security Policy (allows Swagger UI assets to load from trusted CDNs)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
            "img-src 'self' data: fastly.jsdelivr.net; "
            "connect-src 'self';"
        )
        
        # Referrer disclosure restriction
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # HTTP Strict Transport Security (HSTS)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class ProductionSafetyMiddleware(BaseHTTPMiddleware):
    """Middleware validating request protocols and logging CORS/HTTPS warnings in production."""
    async def dispatch(self, request: Request, call_next):
        if os.getenv("ENVIRONMENT") == "production":
            # HTTPS check (respecting proxy headers)
            proto = request.headers.get("x-forwarded-proto", "http").lower()
            if proto != "https" and not request.url.path.startswith("/api/v1/health") and not request.url.path.startswith("/health"):
                logger.warning(
                    f"[SECURITY WARNING] Insecure request protocol '{proto}' detected "
                    f"on production endpoint: {request.url.path} from client {request.client.host if request.client else 'unknown'}"
                )
        return await call_next(request)


def validate_production_environment():
    """Runs structural security audit on settings loaded for production environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    if env != "production":
        return

    logger.info("Executing production environment security audit...")

    # 1. Debug settings verification
    if settings.server.debug:
        raise ValueError(
            "CRITICAL SECURITY FAILURE: SERVER__DEBUG cannot be set to True "
            "in a production environment."
        )
    if settings.server.reload:
        raise ValueError(
            "CRITICAL SECURITY FAILURE: SERVER__RELOAD must be disabled in production."
        )

    # 2. Secret key verification
    jwt_key = settings.security.jwt_secret_key
    default_keys = [
        "production-secret-signing-key-placeholder-32-chars",
        "local-dev-secret-key-32-characters-minimum"
    ]
    if jwt_key in default_keys:
        raise ValueError(
            "CRITICAL SECURITY FAILURE: SECURITY__JWT_SECRET_KEY is using a default "
            "placeholder value. You must generate a unique, cryptographically random secret."
        )
    if len(jwt_key) < 32:
        raise ValueError(
            "CRITICAL SECURITY FAILURE: SECURITY__JWT_SECRET_KEY must be at least "
            "32 characters long."
        )

    # 3. CORS configuration validation
    if "*" in settings.server.allowed_origins:
        raise ValueError(
            "CRITICAL SECURITY FAILURE: SERVER__ALLOWED_ORIGINS cannot contain wildcard '*' "
            "in a production environment."
        )

    # 4. Redis password validation
    if settings.redis.enabled and not settings.redis.mock_fallback:
        if not settings.redis.password:
            logger.warning(
                "[SECURITY WARNING] Redis is enabled in production but no access "
                "password has been configured."
            )

    logger.info("Production security audit completed with no fatal issues.")
