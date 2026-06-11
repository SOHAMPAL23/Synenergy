"""EnerVision AI - API Routes package."""
from fastapi import APIRouter

from backend.api.routes.auth import router as auth_router
from backend.api.routes.upload import router as upload_router
from backend.api.routes.ml import router as ml_router
from backend.api.routes.dashboard import router as dashboard_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(upload_router)
api_router.include_router(ml_router)
api_router.include_router(dashboard_router)

__all__ = ["api_router"]
