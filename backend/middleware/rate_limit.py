"""
EnerVision AI - Rate Limiting Middleware
Sliding window rate limiter using in-memory counter (production: use Redis).
"""

import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-IP sliding window rate limiter.
    Limits: RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW_SECONDS.
    Excluded paths: /health
    """

    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self._window: int = settings.RATE_LIMIT_WINDOW_SECONDS
        self._max_requests: int = settings.RATE_LIMIT_REQUESTS
        self._clients: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health check
        if request.url.path.rstrip("/") in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_start = now - self._window

        # Clean old timestamps
        timestamps = self._clients[client_ip]
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= self._max_requests:
            retry_after = int(self._window - (now - timestamps[0]))
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded.",
                    "detail": f"Max {self._max_requests} requests per {self._window}s.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._max_requests)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self._max_requests - len(timestamps))
        )
        response.headers["X-RateLimit-Window"] = str(self._window)
        return response
