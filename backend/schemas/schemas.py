"""
EnerVision AI - Pydantic Schemas
Request / Response schemas for all API endpoints.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, field_validator, model_validator


# ─── Base ─────────────────────────────────────────────────────────────────────

class OrmBase(BaseModel):
    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════════════════════════════════════════
# Auth Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str = "viewer"

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        if v not in ("admin", "analyst", "viewer"):
            raise ValueError("Role must be admin, analyst, or viewer.")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(OrmBase):
    id: uuid.UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Site Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class SiteCreate(BaseModel):
    name: str
    location: Optional[str] = None
    timezone: str = "UTC"
    description: Optional[str] = None


class SiteResponse(OrmBase):
    id: uuid.UUID
    name: str
    location: Optional[str]
    timezone: str
    description: Optional[str]
    is_active: bool
    created_at: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Meter Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class MeterCreate(BaseModel):
    meter_id: str
    meter_type: str = "electricity"
    unit: str = "kWh"
    description: Optional[str] = None


class MeterResponse(OrmBase):
    id: uuid.UUID
    site_id: uuid.UUID
    meter_id: str
    meter_type: str
    unit: str
    is_active: bool
    created_at: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Upload Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    rows_loaded: int
    rows_valid: int
    rows_rejected: int
    columns: List[str]
    time_range: Dict[str, str]
    warnings: List[str] = []
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# Training Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class TrainRequest(BaseModel):
    upload_id: Optional[str] = None
    skip_statistical_models: bool = False
    site_id: Optional[uuid.UUID] = None


class ModelMetrics(BaseModel):
    rmse: float
    mae: float
    mape: float


class TrainResponse(BaseModel):
    status: str
    best_model: str
    metrics: Dict[str, ModelMetrics]
    training_time_seconds: float
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# Forecast Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class ForecastPoint(BaseModel):
    timestamp: str
    forecast: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    horizon: str
    model_name: str
    points: List[ForecastPoint]
    generated_at: datetime


class ForecastsResponse(BaseModel):
    forecasts: Dict[str, ForecastResponse]
    best_model: str



# ═══════════════════════════════════════════════════════════════════════════════
# Recommendation Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class RecommendationItem(BaseModel):
    id: Optional[uuid.UUID] = None
    category: str
    priority: str
    title: str
    description: str
    estimated_saving_pct: float
    action_items: List[str]


class RecommendationsResponse(BaseModel):
    total: int
    high_priority: int
    medium_priority: int
    low_priority: int
    recommendations: List[RecommendationItem]


# ═══════════════════════════════════════════════════════════════════════════════
# Explanation Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureImportanceItem(BaseModel):
    feature: str
    mean_abs_shap: float
    rank: int


class ExplanationResponse(BaseModel):
    model_name: str
    explainer_type: str
    feature_importances: List[FeatureImportanceItem]
    top_features: List[str]
    generated_at: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Anomaly Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class AnomalyPoint(BaseModel):
    timestamp: str
    value: float
    is_anomaly: bool
    anomaly_score: float  # 0.0–1.0 fraction of methods that flagged it
    severity: str         # "low" | "medium" | "high"


class MethodBreakdown(BaseModel):
    method: str
    count: int


class AnomaliesResponse(BaseModel):
    total_records: int
    anomaly_count: int
    anomaly_rate_pct: float
    points: List[AnomalyPoint]
    method_breakdown: List[MethodBreakdown]
    generated_at: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard Schema
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardStats(BaseModel):
    total_records: int
    date_range_start: Optional[str]
    date_range_end: Optional[str]
    avg_consumption_mw: float
    max_consumption_mw: float
    min_consumption_mw: float
    best_model: Optional[str]
    forecast_horizons_available: List[str]
    recommendations_count: int
    high_priority_recommendations: int


class DashboardResponse(BaseModel):
    user: UserResponse
    stats: DashboardStats
    recent_forecasts: Optional[List[ForecastPoint]]
    top_recommendations: List[RecommendationItem]


# ═══════════════════════════════════════════════════════════════════════════════
# Health Schema
# ═══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    database: str
    ml_models_available: bool
    timestamp: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Error Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Live Prediction Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionRequest(BaseModel):
    features: List[Dict[str, float]]


class PredictionResponse(BaseModel):
    predictions: List[float]
    model_name: str
