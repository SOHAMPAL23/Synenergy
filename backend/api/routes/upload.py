"""
EnerVision AI - Upload Route
POST /upload — CSV energy data upload with validation.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user_id
from backend.core.config import settings
from backend.database.session import get_db
from backend.schemas.schemas import UploadResponse
from backend.services import UploadService

router = APIRouter(tags=["Data Upload"])

MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload energy CSV data",
    description=(
        "Upload a CSV file containing energy consumption data. "
        "The file must contain a `DE_load_actual_entsoe_transparency` column "
        "and a timestamp index or `utc_timestamp` column."
    ),
)
async def upload_csv(
    file: UploadFile = File(..., description="CSV file with energy data"),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a CSV energy dataset.

    - Validates file type (CSV only)
    - Parses timestamps and target column
    - Persists rows to `energy_records` table
    - Returns summary with rows loaded, rejected, time range
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported.",
        )

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    try:
        svc = UploadService(db, user_id)
        return await svc.process(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {e}",
        )
