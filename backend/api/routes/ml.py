"""
EnerVision AI - ML Routes
POST /train, GET /forecast, GET /anomalies, GET /recommendations, GET /explanations
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user_id, require_analyst_or_above
from backend.database.session import get_db
from backend.schemas.schemas import (
    TrainRequest, TrainResponse,
    ForecastsResponse,
    AnomaliesResponse,
    RecommendationsResponse,
    ExplanationResponse,
    PredictionRequest, PredictionResponse,
)
from backend.services.services import MLService, run_ml_pipeline_background

router = APIRouter(tags=["ML Pipeline"])


def _ml_service(user_id: str, db: AsyncSession) -> MLService:
    return MLService(db, user_id)


@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Train ML forecasting models",
    description=(
        "Runs the full EnerVision training pipeline: cleaning → feature engineering → "
        "model selection → forecasting → anomaly detection → recommendations. "
        "Results are persisted to the database."
    ),
)
async def train(
    background_tasks: BackgroundTasks,
    req: TrainRequest = TrainRequest(),
    user_id: str = Depends(get_current_user_id),
):
    """
    Trigger ML training pipeline on uploaded energy data in the background.

    Requires at least one CSV upload via `POST /upload`.
    """
    try:
        req_dict = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        background_tasks.add_task(run_ml_pipeline_background, user_id, req_dict)
        return TrainResponse(
            status="processing",
            message="Training pipeline started in the background. Check dashboard for updates.",
            best_model="Processing...",
            metrics={},
            training_time_seconds=0.0,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {e}",
        )


@router.get(
    "/forecast",
    response_model=ForecastsResponse,
    summary="Get energy forecasts",
    description="Retrieve 24h, 7d, and 30d energy forecasts from the most recent training run.",
)
async def get_forecasts(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return multi-horizon energy forecasts (24h, 7d, 30d)."""
    try:
        return await _ml_service(user_id, db).get_forecasts()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))



@router.get(
    "/recommendations",
    response_model=RecommendationsResponse,
    summary="Get energy optimization recommendations",
    description="Retrieve rule-based energy optimization recommendations.",
)
async def get_recommendations(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return prioritized energy optimization recommendations."""
    return await _ml_service(user_id, db).get_recommendations()


@router.get(
    "/anomalies",
    response_model=AnomaliesResponse,
    summary="Get anomaly detection results",
    description="Run multi-method anomaly detection (Z-Score, IQR, Isolation Forest, LOF, One-Class SVM) on user energy data.",
)
async def get_anomalies(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return anomaly points with severity scores and per-method breakdown."""
    try:
        return await _ml_service(user_id, db).get_anomalies()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {e}",
        )


@router.get(
    "/explanations",
    response_model=ExplanationResponse,
    summary="Get SHAP feature importance",
    description="Compute and return SHAP feature importance values for the fitted model.",
)
async def get_explanations(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return SHAP-based feature importance explanations."""
    try:
        return await _ml_service(user_id, db).get_explanations()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {e}",
        )


@router.get(
    "/forecast/live",
    response_model=ForecastsResponse,
    summary="Get live energy forecasts",
    description="Loads the serialized model and generates a forecast dynamically using the latest database data.",
)
async def get_live_forecast(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return live multi-horizon energy forecasts (24h, 7d, 30d) computed on the fly."""
    try:
        return await _ml_service(user_id, db).predict_live()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Live forecasting failed: {e}",
        )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Run raw model inference",
    description="Submit a list of feature dictionaries to get model predictions.",
)
async def predict(
    req: PredictionRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return model predictions for custom input features."""
    try:
        service = _ml_service(user_id, db)
        predictions = await service.predict_raw(req.features)
        
        # Load metadata to find the model name
        from ml.models.serializer import ModelSerializer
        from ml.utils.config_loader import config as ml_cfg
        ser = ModelSerializer(cfg=ml_cfg)
        metadata = ser.load_metadata(name=f"metadata_{user_id}")
        model_name = metadata.get("best_model", "Model")
        
        return PredictionResponse(predictions=predictions, model_name=model_name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}",
        )
