"""
EnerVision AI - Dashboard & Health Routes
GET /dashboard, GET /health
"""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from backend.api.deps import get_current_user_id
from backend.core.config import settings
from backend.database.session import get_db
from backend.schemas.schemas import DashboardResponse, HealthResponse
from backend.services import MLService

router = APIRouter(tags=["Dashboard & Health"])


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    summary="Get comprehensive dashboard data",
    description=(
        "Returns a complete dashboard view including user profile, "
        "consumption statistics, recent forecasts, anomalies, and top recommendations."
    ),
)
async def get_dashboard(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Aggregated dashboard data for the authenticated user."""
    try:
        svc = MLService(db, user_id)
        return await svc.get_dashboard()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dashboard error: {e}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API, database, and ML model availability.",
)
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Returns the health status of:
    - API server
    - Database connectivity
    - ML model artifacts availability
    """
    import os

    # DB check
    db_status = "unknown"
    try:
        await db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"error: {e}"

    # ML models check
    ml_available = os.path.isdir(settings.ML_MODELS_DIR)

    return HealthResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        database=db_status,
        ml_models_available=ml_available,
        timestamp=datetime.now(timezone.utc),
    )
