"""
EnerVision AI - Business Services
Thin orchestration layer between routes and repositories + ML pipeline.
"""

import io
import os
import uuid
import time
import logging
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.security import hash_password, verify_password, create_access_token, create_refresh_token
from backend.repositories import (
    UserRepository, EnergyRecordRepository, ForecastRepository,
    RecommendationRepository,
)
from backend.schemas.schemas import (
    RegisterRequest, LoginRequest, TokenResponse,
    UploadResponse, TrainResponse, TrainRequest,
    ForecastsResponse, ForecastResponse, ForecastPoint,
    RecommendationsResponse, RecommendationItem,
    ExplanationResponse, FeatureImportanceItem,
    DashboardResponse, DashboardStats,
)

logger = logging.getLogger(__name__)


# ─── Auth Service ─────────────────────────────────────────────────────────────

class AuthService:

    def __init__(self, db: AsyncSession) -> None:
        self._repo = UserRepository(db)

    async def register(self, req: RegisterRequest):
        from backend.models.orm import User
        if req.role == "admin":
            raise ValueError("Direct registration as an administrator is not permitted.")
        existing = await self._repo.get_by_email(req.email)
        if existing:
            raise ValueError("Email already registered.")
        user = User(
            email=req.email,
            hashed_password=hash_password(req.password),
            full_name=req.full_name,
            role=req.role,
        )
        return await self._repo.create(user)

    async def login(self, req: LoginRequest) -> TokenResponse:
        user = await self._repo.get_by_email(req.email)
        if not user or not verify_password(req.password, user.hashed_password):
            raise ValueError("Invalid email or password.")
        if not user.is_active:
            raise ValueError("Account is deactivated.")
        await self._repo.update(user.id, {"last_login": datetime.now(timezone.utc)})
        return TokenResponse(
            access_token=create_access_token(str(user.id), {"role": user.role}),
            refresh_token=create_refresh_token(str(user.id)),
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    async def refresh(self, user_id: str) -> TokenResponse:
        user = await self._repo.get(uuid.UUID(user_id))
        if not user or not user.is_active:
            raise ValueError("User not found or inactive.")
        return TokenResponse(
            access_token=create_access_token(str(user.id), {"role": user.role}),
            refresh_token=create_refresh_token(str(user.id)),
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    async def get_user(self, user_id: str):
        return await self._repo.get(uuid.UUID(user_id))


# ─── Upload Service ───────────────────────────────────────────────────────────

class UploadService:
    """Parse, validate and persist uploaded CSV energy data."""

    REQUIRED_COLS = {"DE_load_actual_entsoe_transparency"}
    TIMESTAMP_COLS = ["utc_timestamp", "timestamp", "datetime", "date", "time"]

    def __init__(self, db: AsyncSession, user_id: str) -> None:
        self._db = db
        self._user_id = uuid.UUID(user_id)
        self._energy_repo = EnergyRecordRepository(db)

    async def process(self, filename: str, content: bytes) -> UploadResponse:
        t0 = time.perf_counter()
        upload_id = str(uuid.uuid4())
        warnings: List[str] = []

        # ── Parse ────────────────────────────────────────────────────────
        try:
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
        except Exception as e:
            raise ValueError(f"CSV parse error: {e}")

        # ── Timestamp column ─────────────────────────────────────────────
        ts_col = None
        for c in self.TIMESTAMP_COLS:
            if c in df.columns:
                ts_col = c
                break

        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df.dropna(subset=[ts_col])
            df = df.set_index(ts_col)
        else:
            # Try parsing the index directly
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                warnings.append("No timestamp column found; using row index.")

        # ── Target column check ──────────────────────────────────────────
        target_col = "DE_load_actual_entsoe_transparency"
        if target_col not in df.columns:
            # Try to be lenient — check for a numeric column to use
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                df = df.rename(columns={numeric_cols[0]: target_col})
                warnings.append(
                    f"Target column not found; using '{numeric_cols[0]}' as target."
                )
            else:
                raise ValueError(
                    f"Required column '{target_col}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

        rows_raw = len(df)

        # ── Persist to DB ────────────────────────────────────────────────
        from backend.models.orm import EnergyRecord
        records_to_insert = []
        rows_rejected = 0

        for ts, row in df.iterrows():
            val = row.get(target_col)
            if pd.isna(val):
                rows_rejected += 1
                continue
            records_to_insert.append(
                EnergyRecord(
                    user_id=self._user_id,
                    timestamp=ts,
                    value=float(val),
                    unit="MW",
                    source_file=filename,
                )
            )

        # Batch insert
        for rec in records_to_insert:
            self._db.add(rec)
        await self._db.flush()

        # ── Determine time range ─────────────────────────────────────────
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            time_range = {
                "start": str(df.index.min()),
                "end": str(df.index.max()),
            }
        else:
            time_range = {"start": "unknown", "end": "unknown"}

        elapsed = time.perf_counter() - t0
        logger.info(
            "Upload %s: %d rows processed in %.2fs for user %s",
            upload_id, len(records_to_insert), elapsed, self._user_id,
        )

        return UploadResponse(
            upload_id=upload_id,
            filename=filename,
            rows_loaded=rows_raw,
            rows_valid=len(records_to_insert),
            rows_rejected=rows_rejected,
            columns=list(df.columns),
            time_range=time_range,
            warnings=warnings,
            message=f"Successfully uploaded {len(records_to_insert)} energy records.",
        )


# ─── ML Service ───────────────────────────────────────────────────────────────

class MLService:
    """
    Runs the EnerVision ML pipeline stages and persists results to DB.
    Loads data from the DB (energy_records for the user).
    """

    def __init__(self, db: AsyncSession, user_id: str) -> None:
        self._db = db
        self._user_id = uuid.UUID(user_id)
        self._energy_repo = EnergyRecordRepository(db)
        self._forecast_repo = ForecastRepository(db)
        self._rec_repo = RecommendationRepository(db)

    async def _load_dataframe(self) -> pd.DataFrame:
        """Load user's energy records from DB into a DataFrame."""
        records = await self._energy_repo.get_by_user(self._user_id, limit=50_000)
        if not records:
            raise ValueError("No energy data found. Upload a CSV first.")
        data = [
            {"timestamp": r.timestamp, "DE_load_actual_entsoe_transparency": r.value}
            for r in records
        ]
        df = pd.DataFrame(data).set_index("timestamp").sort_index()
        if not df.index.tzinfo:
            df.index = df.index.tz_localize("UTC")
        return df

    async def train(self, req: TrainRequest) -> TrainResponse:
        t0 = time.perf_counter()
        df = await self._load_dataframe()
        logger.info("Training on %d rows for user %s", len(df), self._user_id)

        # ── Import ML modules (lazy to avoid slow startup) ───────────────
        import sys, os
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if root not in sys.path:
            sys.path.insert(0, root)

        from ml.preprocessing.cleaner import DataCleaner
        from ml.feature_engineering.feature_pipeline import FeatureEngineer
        from ml.forecasting.model_selector import ModelSelector
        from ml.utils.helpers import time_split
        from ml.utils.config_loader import config as ml_cfg

        def _run_ml_math():
            cleaner = DataCleaner(cfg=ml_cfg)
            df_clean = cleaner.clean(df)

            fe = FeatureEngineer(cfg=ml_cfg)
            df_feat = fe.transform(df_clean)
            feat_cols = fe.get_feature_columns(df_feat)
            X = df_feat[feat_cols]
            y = df_feat[ml_cfg.data.target_column]

            X_train, X_test = time_split(X, test_size=0.2)
            y_train = y.loc[X_train.index]
            y_test = y.loc[X_test.index]

            selector = ModelSelector(cfg=ml_cfg)
            best_model, metrics = selector.run(X_train, y_train, X_test, y_test)
            
            from ml.forecasting.forecast_generator import ForecastGenerator
            fg = ForecastGenerator(best_model, fe, cfg=ml_cfg)
            forecasts = fg.generate(df_clean)
            
            from ml.anomaly_detection.anomaly_detector import AnomalyDetector
            detector = AnomalyDetector(cfg=ml_cfg)
            anomaly_df = detector.detect(df_clean)
            
            from ml.recommendation_engine.recommender import RecommendationEngine
            engine_rec = RecommendationEngine(cfg=ml_cfg)
            recs = engine_rec.generate(
                forecast_df=forecasts.get("24h"),
                history_df=df_clean,
                anomaly_df=anomaly_df,
            )
            
            # Serialize best model, feature engineer, and metadata
            from ml.models.serializer import ModelSerializer
            ser = ModelSerializer(cfg=ml_cfg)
            ser.save_model(best_model, f"best_model_{self._user_id}")
            ser.save_model(fe, f"feature_engineer_{self._user_id}")
            
            metadata_dict = {
                "user_id": str(self._user_id),
                "best_model": best_model.name,
                "metrics": {k: {m: round(v, 4) for m, v in vals.items()} for k, vals in metrics.items()},
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }
            ser.save_metadata(metadata_dict, name=f"metadata_{self._user_id}")
            
            return best_model.name, metrics, forecasts, recs
            
        best_model_name, metrics, forecasts, recs = await asyncio.to_thread(_run_ml_math)

        # ── Persist forecasts ────────────────────────────────────────────
        await self._forecast_repo.mark_all_old(self._user_id)
        from backend.models.orm import Forecast

        for horizon, fc_df in forecasts.items():
            fc_data = {
                str(ts): {
                    "forecast": float(row["forecast"]),
                    "lower_bound": float(row["lower_bound"]),
                    "upper_bound": float(row["upper_bound"]),
                }
                for ts, row in fc_df.iterrows()
            }
            best_metrics = metrics.get(best_model.name, {})
            fc_orm = Forecast(
                user_id=self._user_id,
                model_name=best_model.name,
                horizon=horizon,
                forecast_data=fc_data,
                metrics={k: {m: round(v, 4) for m, v in vals.items()} for k, vals in metrics.items()},
                rmse=best_metrics.get("rmse"),
                mae=best_metrics.get("mae"),
                mape=best_metrics.get("mape"),
                is_latest=True,
            )
            self._db.add(fc_orm)

        # ── Persist recommendations ───────────────────────────────────────
        from backend.models.orm import Recommendation as RecOrm

        await self._rec_repo.deactivate_all_for_user(self._user_id)
        for r in recs:
            self._db.add(RecOrm(
                user_id=self._user_id,
                category=r.category,
                priority=r.priority,
                title=r.title,
                description=r.description,
                estimated_saving_pct=r.estimated_saving_pct,
                action_items=r.action_items,
            ))

        await self._db.flush()
        elapsed = time.perf_counter() - t0

        from backend.schemas.schemas import ModelMetrics
        metrics_out = {
            name: ModelMetrics(rmse=m["rmse"], mae=m["mae"], mape=m["mape"])
            for name, m in metrics.items()
        }

        return TrainResponse(
            status="success",
            best_model=best_model_name,
            metrics=metrics_out,
            training_time_seconds=round(elapsed, 2),
            message=f"Training complete. Best model: {best_model_name}",
        )

    async def get_forecasts(self) -> ForecastsResponse:
        fc_list = await self._forecast_repo.get_latest_by_user(self._user_id)
        if not fc_list:
            raise ValueError("No forecasts found. Run /train first.")

        forecasts_dict: Dict[str, ForecastResponse] = {}
        best_model = fc_list[0].model_name

        for fc in fc_list:
            points = []
            for ts_str, vals in fc.forecast_data.items():
                points.append(ForecastPoint(
                    timestamp=ts_str,
                    forecast=vals["forecast"],
                    lower_bound=vals["lower_bound"],
                    upper_bound=vals["upper_bound"],
                ))
            forecasts_dict[fc.horizon] = ForecastResponse(
                horizon=fc.horizon,
                model_name=fc.model_name,
                points=sorted(points, key=lambda p: p.timestamp),
                generated_at=fc.created_at,
            )

        return ForecastsResponse(forecasts=forecasts_dict, best_model=best_model)

    async def get_recommendations(self) -> RecommendationsResponse:
        recs = await self._rec_repo.get_active_by_user(self._user_id)
        items = [
            RecommendationItem(
                id=r.id,
                category=r.category,
                priority=r.priority,
                title=r.title,
                description=r.description,
                estimated_saving_pct=r.estimated_saving_pct,
                action_items=r.action_items or [],
            )
            for r in recs
        ]
        return RecommendationsResponse(
            total=len(items),
            high_priority=sum(1 for i in items if i.priority == "HIGH"),
            medium_priority=sum(1 for i in items if i.priority == "MEDIUM"),
            low_priority=sum(1 for i in items if i.priority == "LOW"),
            recommendations=items,
        )

    async def get_anomalies(self):
        """Run AnomalyDetector on user's energy records and return AnomaliesResponse."""
        from backend.schemas.schemas import AnomaliesResponse, AnomalyPoint, MethodBreakdown
        df = await self._load_dataframe()

        import sys, os
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if root not in sys.path:
            sys.path.insert(0, root)

        from ml.preprocessing.cleaner import DataCleaner
        from ml.anomaly_detection.anomaly_detector import AnomalyDetector
        from ml.utils.config_loader import config as ml_cfg

        cleaner = DataCleaner(cfg=ml_cfg)
        df_clean = cleaner.clean(df)

        detector = AnomalyDetector(cfg=ml_cfg)
        result_df = detector.detect(df_clean)

        target_col = ml_cfg.data.target_column
        flag_methods = [c for c in result_df.columns if c.startswith("anomaly_") and c != "anomaly_score"]

        def _severity(score: float) -> str:
            if score >= 0.8:
                return "high"
            elif score >= 0.5:
                return "medium"
            return "low"

        # Build per-point list (limit to 2000 points for API response size)
        sample = result_df.tail(2000) if len(result_df) > 2000 else result_df
        points = []
        for ts, row in sample.iterrows():
            val = float(row.get(target_col, 0.0))
            score = float(row.get("anomaly_score", 0.0))
            is_anom = bool(row.get("is_anomaly", 0))
            points.append(AnomalyPoint(
                timestamp=str(ts),
                value=val,
                is_anomaly=is_anom,
                anomaly_score=round(score, 4),
                severity=_severity(score) if is_anom else "none",
            ))

        method_breakdown = [
            MethodBreakdown(
                method=col.replace("anomaly_", ""),
                count=int(result_df[col].sum()),
            )
            for col in flag_methods
        ]

        total = len(result_df)
        anom_count = int(result_df.get("is_anomaly", result_df.get(flag_methods[0] if flag_methods else [], 0)).sum()) if "is_anomaly" in result_df.columns else 0

        return AnomaliesResponse(
            total_records=total,
            anomaly_count=anom_count,
            anomaly_rate_pct=round(100 * anom_count / max(total, 1), 2),
            points=points,
            method_breakdown=method_breakdown,
            generated_at=datetime.now(timezone.utc),
        )


    async def get_explanations(self) -> ExplanationResponse:
        """Run SHAP on the user's data and return feature importance using the best serialized model if possible."""
        df = await self._load_dataframe()

        import sys, os
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if root not in sys.path:
            sys.path.insert(0, root)

        from ml.preprocessing.cleaner import DataCleaner
        from ml.feature_engineering.feature_pipeline import FeatureEngineer
        from ml.forecasting.linear_model import LinearRegressionModel
        from ml.explainability.shap_explainer import SHAPExplainer
        from ml.utils.config_loader import config as ml_cfg
        from ml.models.serializer import ModelSerializer

        cleaner = DataCleaner(cfg=ml_cfg)
        df_clean = cleaner.clean(df)
        
        ser = ModelSerializer(cfg=ml_cfg)
        model_name = "LinearRegression"
        explainer_type = "LinearExplainer"
        
        # Check if saved model exists and is not statistical
        if ser.model_exists(f"best_model_{self._user_id}"):
            best_model = ser.load_model(f"best_model_{self._user_id}")
            stat_models = {"ARIMA", "SARIMA", "SARIMAX"}
            if best_model.name not in stat_models:
                model = best_model
                fe = ser.load_model(f"feature_engineer_{self._user_id}")
                df_feat = fe.transform(df_clean)
                feat_cols = fe.get_feature_columns(df_feat)
                X = df_feat[feat_cols]
                model_name = best_model.name
                if model_name in ("XGBoost", "RandomForest"):
                    explainer_type = "TreeExplainer"
                else:
                    explainer_type = "KernelExplainer"
            else:
                # Fallback to LinearRegression for explainability
                fe = FeatureEngineer(cfg=ml_cfg)
                df_feat = fe.transform(df_clean)
                feat_cols = fe.get_feature_columns(df_feat)
                X = df_feat[feat_cols]
                y = df_feat[ml_cfg.data.target_column]
                model = LinearRegressionModel()
                model.fit(X, y)
        else:
            # Fallback if no saved model
            fe = FeatureEngineer(cfg=ml_cfg)
            df_feat = fe.transform(df_clean)
            feat_cols = fe.get_feature_columns(df_feat)
            X = df_feat[feat_cols]
            y = df_feat[ml_cfg.data.target_column]
            model = LinearRegressionModel()
            model.fit(X, y)

        explainer = SHAPExplainer(model, feat_cols, cfg=ml_cfg)
        bg = min(100, len(X))
        explainer.fit(X.iloc[:bg])
        explainer.explain(X.iloc[:min(200, len(X))])
        importance_df = explainer.feature_importance()

        items = [
            FeatureImportanceItem(
                feature=row["feature"],
                mean_abs_shap=float(row["mean_abs_shap"]),
                rank=i + 1,
            )
            for i, (_, row) in enumerate(importance_df.iterrows())
        ]

        return ExplanationResponse(
            model_name=model_name,
            explainer_type=explainer_type,
            feature_importances=items,
            top_features=[i.feature for i in items[:5]],
            generated_at=datetime.now(timezone.utc),
        )

    async def get_dashboard(self) -> DashboardResponse:
        from backend.schemas.schemas import UserResponse

        user_repo = UserRepository(self._db)
        user = await user_repo.get(self._user_id)

        total_recs = await self._energy_repo.count(user_id=self._user_id)
        records = await self._energy_repo.get_by_user(self._user_id, limit=1000)
        fc_list = await self._forecast_repo.get_latest_by_user(self._user_id)
        recs = await self._rec_repo.get_active_by_user(self._user_id)

        values = [r.value for r in records]
        timestamps = [r.timestamp for r in records]

        stats = DashboardStats(
            total_records=total_recs,
            date_range_start=str(min(timestamps)) if timestamps else None,
            date_range_end=str(max(timestamps)) if timestamps else None,
            avg_consumption_mw=round(sum(values) / max(len(values), 1), 2),
            max_consumption_mw=max(values) if values else 0.0,
            min_consumption_mw=min(values) if values else 0.0,
            best_model=fc_list[0].model_name if fc_list else None,
            forecast_horizons_available=list({fc.horizon for fc in fc_list}),
            recommendations_count=len(recs),
            high_priority_recommendations=sum(1 for r in recs if r.priority == "HIGH"),
        )

        recent_forecast = None
        if fc_list:
            recent_fc = fc_list[0]
            recent_forecast = [
                ForecastPoint(
                    timestamp=ts_str,
                    forecast=vals["forecast"],
                    lower_bound=vals["lower_bound"],
                    upper_bound=vals["upper_bound"],
                )
                for ts_str, vals in list(recent_fc.forecast_data.items())[:24]
            ]

        top_recs = [
            RecommendationItem(
                id=r.id,
                category=r.category,
                priority=r.priority,
                title=r.title,
                description=r.description,
                estimated_saving_pct=r.estimated_saving_pct,
                action_items=r.action_items or [],
            )
            for r in recs[:5]
        ]

        return DashboardResponse(
            user=UserResponse.model_validate(user),
            stats=stats,
            recent_forecasts=recent_forecast,
            top_recommendations=top_recs,
        )

    async def predict_live(self) -> ForecastsResponse:
        """Loads the serialized user model and feature pipeline, then generates a live forecast from the latest DB data."""
        from ml.models.serializer import ModelSerializer
        from ml.utils.config_loader import config as ml_cfg
        from ml.forecasting.forecast_generator import ForecastGenerator
        from ml.preprocessing.cleaner import DataCleaner

        ser = ModelSerializer(cfg=ml_cfg)
        if not ser.model_exists(f"best_model_{self._user_id}"):
            raise ValueError("No trained model found for this user. Train a model first.")

        best_model = ser.load_model(f"best_model_{self._user_id}")
        fe = ser.load_model(f"feature_engineer_{self._user_id}")

        df = await self._load_dataframe()
        cleaner = DataCleaner(cfg=ml_cfg)
        df_clean = cleaner.clean(df)

        fg = ForecastGenerator(best_model, fe, cfg=ml_cfg)
        forecasts = fg.generate(df_clean)

        forecasts_dict: Dict[str, ForecastResponse] = {}
        for horizon, fc_df in forecasts.items():
            points = []
            for ts, row in fc_df.iterrows():
                points.append(ForecastPoint(
                    timestamp=str(ts),
                    forecast=float(row["forecast"]),
                    lower_bound=float(row["lower_bound"]),
                    upper_bound=float(row["upper_bound"]),
                ))
            forecasts_dict[horizon] = ForecastResponse(
                horizon=horizon,
                model_name=best_model.name,
                points=sorted(points, key=lambda p: p.timestamp),
                generated_at=datetime.now(timezone.utc),
            )

        return ForecastsResponse(forecasts=forecasts_dict, best_model=best_model.name)

    async def predict_raw(self, features: List[Dict[str, float]]) -> List[float]:
        """Runs raw prediction using the serialized best model on a batch of pre-calculated feature dictionaries."""
        from ml.models.serializer import ModelSerializer
        from ml.utils.config_loader import config as ml_cfg

        ser = ModelSerializer(cfg=ml_cfg)
        if not ser.model_exists(f"best_model_{self._user_id}"):
            raise ValueError("No trained model found for this user. Train a model first.")

        best_model = ser.load_model(f"best_model_{self._user_id}")
        
        stat_models = {"ARIMA", "SARIMA", "SARIMAX"}
        if best_model.name in stat_models:
            raise ValueError(f"Raw feature prediction is not supported for statistical model '{best_model.name}'. Use live forecast instead.")

        df_input = pd.DataFrame(features)

        # Retrieve feature names the model was trained on
        feature_names = getattr(best_model, "_feature_names", None)
        if not feature_names:
            fe = ser.load_model(f"feature_engineer_{self._user_id}")
            # Make a dummy 200-row DataFrame with a valid DatetimeIndex to get features
            dummy_idx = pd.date_range("2020-01-01", periods=200, freq="h")
            dummy_df = pd.DataFrame({ml_cfg.data.target_column: 1.0}, index=dummy_idx)
            dummy_feat = fe.transform(dummy_df)
            feature_names = fe.get_feature_columns(dummy_feat)

        if feature_names:
            # Add missing columns filled with 0.0 and order them
            for col in feature_names:
                if col not in df_input.columns:
                    df_input[col] = 0.0
            df_input = df_input[feature_names]

        preds = best_model.predict(df_input)
        return [float(p) for p in preds]

async def run_ml_pipeline_background(user_id: str, req_dict: dict):
    from backend.database.session import AsyncSessionLocal
    from backend.schemas.schemas import TrainRequest
    import traceback
    
    logger.info(f"Background ML training started for user {user_id}")
    try:
        async with AsyncSessionLocal() as db:
            service = MLService(db, user_id)
            await service.train(TrainRequest(**req_dict))
            await db.commit()
            logger.info(f"Background ML training completed for user {user_id}")
    except Exception as e:
        logger.error(f"Background ML training failed for user {user_id}: {e}")
        logger.error(traceback.format_exc())
