from __future__ import annotations

import time
import traceback
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from week7.backend.logging_config import log_error, log_api_request


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global system error handling middleware catching uncaught exceptions and formatting standard JSON responses."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        try:
            response = await call_next(request)
            
            # Record execution latency
            process_time = (time.perf_counter() - start_time) * 1000
            
            # Skip logging requests context middleware logs since it logs them separately,
            # but record the API request through centralized helper loggers
            log_api_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=process_time,
                client_ip=client_ip
            )
            return response
        except Exception as exc:
            process_time = (time.perf_counter() - start_time) * 1000
            
            # Log traceback context
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            log_error(
                message=f"Unhandled system exception: {request.method} {request.url.path} - {str(exc)}",
                exc=exc,
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": client_ip,
                    "latency_ms": process_time,
                    "traceback": tb
                }
            )

            # Return proper HTTP response
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An unhandled server error occurred on our servers."
                    }
                }
            )
