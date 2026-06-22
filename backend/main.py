"""
EnerVision AI - Main FastAPI Application
Production-ready FastAPI app with all routes, middleware, and lifecycle events.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import api_router
from backend.core.config import settings
from backend.database.session import create_all_tables
from backend.middleware import RateLimitMiddleware, RequestLoggingMiddleware

# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("enervision")


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  EnerVision AI Backend — Starting up")
    logger.info("  Environment : %s", settings.ENVIRONMENT)
    logger.info("  Version     : %s", settings.APP_VERSION)
    logger.info("=" * 60)

    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.ML_OUTPUTS_DIR, exist_ok=True)

    # Create database tables if they don't exist
    try:
        await create_all_tables()
        logger.info("Database tables verified/created.")
    except Exception as e:
        logger.error("Database initialization failed: %s", e)

    # Add ML root to sys.path for lazy imports in services
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("EnerVision AI Backend — Shutting down.")


# ─── App Factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="EnerVision AI",
        description=(
            "## EnerVision AI — Energy Forecasting & Optimization API\n\n"
            "### Features\n"
            "- **Authentication**: JWT with RBAC (admin / analyst / viewer)\n"
            "- **Data Upload**: CSV energy data ingestion\n"
            "- **ML Training**: AutoML model selection (LinearRegression, RandomForest, XGBoost, ARIMA, SARIMA, SARIMAX)\n"
            "- **Forecasting**: 24h / 7d / 30d energy forecasts\n"
            "- **Anomaly Detection**: Z-Score, IQR, IsolationForest, LOF, One-Class SVM\n"
            "- **Recommendations**: Rule-based energy optimization advice\n"
            "- **Explainability**: SHAP feature importance\n"
            "- **Dashboard**: Comprehensive analytics view\n\n"
            "### Authentication\n"
            "All endpoints (except `/auth/register`, `/auth/login`, `/health`) require a Bearer JWT token.\n"
            "Obtain one via `POST /auth/login`."
        ),
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Custom Middleware ─────────────────────────────────────────────────
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # ── Routes ───────────────────────────────────────────────────────────
    app.include_router(api_router, prefix="/api/v1")

    # ── Root redirect ─────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse(
            content={
                "message": "EnerVision AI Backend",
                "version": settings.APP_VERSION,
                "docs": "/docs",
                "health": "/api/v1/health",
            }
        )

    # ── Global Exception Handler ──────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        
        # Ensure CORS headers are attached to 500 error responses
        headers = {}
        origin = request.headers.get("origin")
        if origin and ("*" in settings.ALLOWED_ORIGINS or origin in settings.ALLOWED_ORIGINS):
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
            
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error.",
                "detail": str(exc) if settings.DEBUG else "Contact support.",
            },
            headers=headers,
        )

    return app


app = create_app()
