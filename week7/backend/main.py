from __future__ import annotations

import sys
import os
# Move current working directory to the end of sys.path to prevent 'datasets' folder shading
cwd = os.getcwd()
if cwd in sys.path:
    sys.path.remove(cwd)
    sys.path.append(cwd)
if '' in sys.path:
    sys.path.remove('')
    sys.path.append('')

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

from week7.backend.configs.config import get_settings
from week7.backend.logging_config import configure_logging

# Load System Configurations
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager handling startup initialization and shutdown cleanups."""
    # Startup Initialization
    log_level = "DEBUG" if settings.server.debug else "INFO"
    configure_logging(log_level=log_level, log_to_file=True)
    logger.info("Initializing production FastAPI web backend...")
    
    # Run production environment safety validation checks
    from week7.backend.middleware.security import validate_production_environment
    validate_production_environment()
    
    logger.info(f"Configuration Loaded - Environment Mock Mode: {settings.model.global_mock}")
    
    yield
    
    # Shutdown Cleanups
    logger.info("Cleaning up backend resources...")
    logger.info("FastAPI web backend shut down successfully.")


# Initialize FastAPI App
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise-grade REST API backend for the AI Fashion Assistant.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Production Security Middlewares (Secure Headers & HTTPS Checks)
from week7.backend.middleware.security import SecurityHeadersMiddleware, ProductionSafetyMiddleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(ProductionSafetyMiddleware)

# Configure Centralized Error Handler Middleware
from week7.backend.middleware.error_handler import ErrorHandlerMiddleware
app.add_middleware(ErrorHandlerMiddleware)

# Configure NSFW Image Safety Filter Middleware
from week7.backend.middleware.nsfw_filter import NSFWFilterMiddleware
app.add_middleware(NSFWFilterMiddleware)

# Configure slowapi Rate Limiting
from slowapi.errors import RateLimitExceeded
from week7.backend.configs.rate_limit import limiter, rate_limit_exceeded_handler

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)




# =============================================================================
# ── Custom Middlewares
# =============================================================================

@app.middleware("http")
async def log_requests_context(request: Request, call_next):
    """Middleware capturing request parameters and execution performance latency."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Incoming Request: {request.method} {request.url.path} | Client IP: {client_ip}")
    
    try:
        response = await call_next(request)
        latency = (time.time() - start_time) * 1000
        logger.info(
            f"Outgoing Response: {request.method} {request.url.path} | "
            f"Status: {response.status_code} | Latency: {latency:.2f}ms"
        )
        response.headers["X-Process-Time-Ms"] = f"{latency:.2f}"
        return response
    except Exception as exc:
        latency = (time.time() - start_time) * 1000
        logger.error(
            f"Request Exception: {request.method} {request.url.path} | "
            f"Latency: {latency:.2f}ms | Error: {str(exc)}"
        )
        raise exc


# =============================================================================
# ── Global Exception Handlers
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Global handler converting pydantic/fastapi validation failures to unified schemas."""
    logger.warning(f"Validation failure on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "The request body or query parameters failed structural validation.",
                "details": exc.errors(),
            }
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Global handler intercepting HTTPExceptions."""
    logger.warning(f"HTTPException triggered on {request.url.path} [{exc.status_code}]: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": f"HTTP_ERROR_{exc.status_code}",
                "message": exc.detail,
            }
        }
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Global fallback interceptor converting runtime uncaught errors to safe JSON structures."""
    logger.error(f"Unhandled system error on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected system error occurred on our servers.",
            }
        }
    )


# =============================================================================
# ── API Endpoint Routers Inclusion
# =============================================================================

from week7.backend.routers import (
    health,
    generation,
    controlnet,
    lora,
    rag,
    recommendations,
    history,
    generation_week2,
    controlnet_week3,
    lora_week4,
    assistant_week5,
    tasks,
)
from week7.backend.routers import auth_router
from week7.backend import health as health_endpoints

app.include_router(health.router, prefix="/api/v1")
app.include_router(generation.router, prefix="/api/v1")
app.include_router(controlnet.router, prefix="/api/v1")
app.include_router(lora.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
app.include_router(recommendations.router, prefix="/api/v1")
app.include_router(history.router, prefix="/api/v1")
app.include_router(generation_week2.router, prefix="")
app.include_router(controlnet_week3.router, prefix="")
app.include_router(lora_week4.router, prefix="")
app.include_router(assistant_week5.router, prefix="")
app.include_router(tasks.router, prefix="")
app.include_router(auth_router.router, prefix="")

# Expose root-level health, readiness, and liveness endpoints
app.include_router(health_endpoints.router)






