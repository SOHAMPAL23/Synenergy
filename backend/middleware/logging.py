"""
EnerVision AI - Request Logging Middleware
Logs method, path, status, and duration for every request.
"""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("enervision.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Structured request/response logging with correlation IDs."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "req_id=%s method=%s path=%s status=%d elapsed_ms=%.1f ip=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request.client.host if request.client else "-",
        )
        response.headers["X-Request-ID"] = request_id
        return response
