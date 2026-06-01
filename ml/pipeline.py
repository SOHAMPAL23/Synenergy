"""
EnerVision AI - Master Pipeline Orchestrator
Wires every stage together into a single callable pipeline class.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ml.ingestion.data_loader import DataLoader
from ml.ingestion.schema_validator import SchemaValidator
from ml.preprocessing.cleaner import DataCleaner
from ml.feature_engineering.feature_pipeline import FeatureEngineer
from ml.forecasting.model_selector import ModelSelector
from ml.forecasting.forecast_generator import ForecastGenerator
from ml.anomaly_detection.anomaly_detector import AnomalyDetector
from ml.explainability.shap_explainer import SHAPExplainer
from ml.recommendation_engine.recommender import RecommendationEngine
from ml.models.serializer import ModelSerializer
from ml.utils.config_loader import config
from ml.utils.helpers import time_split, ensure_dir
from ml.utils.logger import get_logger, PipelineLogger
from ml.utils.visualizer import Visualizer

logger = get_logger(__name__)


class EnerVisionPipeline:
    """
    End-to-end EnerVision AI pipeline.

    Stages
    ------
    1.  Data Ingestion   → load CSV, validate schema
    2.  Preprocessing    → clean, deduplicate, cap outliers
    3.  Feature Eng.     → time/lag/rolling features
    4.  Model Selection  → train 6 models, pick best by RMSE
    5.  Forecasting      → 24h / 7d / 30d horizon forecasts
    6.  Anomaly Detect.  → Z-Score, IQR, IsolationForest, LOF, OC-SVM
    7.  Explainability   → SHAP values + plots
    8.  Recommendations  → rule-based optimization advice
    9.  Serialization    → save all artifacts to disk

    Usage::

        pipeline = EnerVisionPipeline()
        results = pipeline.run()
    """

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or config
        self._serializer = ModelSerializer(cfg=self._cfg)
        # Ensure output directories exist
        for d in [
            self._cfg.data.processed_dir,
            self._cfg.data.models_dir,
            self._cfg.data.reports_dir,
            self._cfg.data.plots_dir,
            self._cfg.logging.log_dir,
        ]:
            ensure_dir(d)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, skip_training: bool = False) -> Dict:
        """
        Execute the full pipeline.

        Args:
            skip_training: If True and a saved model exists, load it
                           instead of retraining (for inference-only runs).

        Returns:
            results dict with keys: forecasts, anomaly_summary,
            shap_importance, recommendations, metrics.
        """
        t_start = time.perf_counter()
        logger.info("=" * 70)
        logger.info("  EnerVision AI Pipeline — Starting")
        logger.info("=" * 70)

        # ── Stage 1 & 2: Ingest + Validate ─────────────────────────────
        df_raw = self._stage_ingest()

        # ── Stage 3: Preprocess ────────────────────────────────────────
        df_clean = self._stage_preprocess(df_raw)

        # ── Stage 4: Feature Engineering ───────────────────────────────
        fe = FeatureEngineer(cfg=self._cfg)
        df_features = self._stage_features(df_clean, fe)

        target_col = self._cfg.data.target_column
        feature_cols = fe.get_feature_columns(df_features)
        X = df_features[feature_cols]
        y = df_features[target_col]

        # ── Stage 5: Model Selection / Training ────────────────────────
        best_model, metrics = self._stage_train(
            X, y, skip_training, fe
        )

        # ── Stage 6: Forecasting ───────────────────────────────────────
        forecasts = self._stage_forecast(df_clean, best_model, fe)

        # ── Stage 7: Anomaly Detection ─────────────────────────────────
        anomaly_df, anomaly_summary = self._stage_anomaly(df_clean)

        # ── Stage 8: Explainability ────────────────────────────────────
        shap_importance = self._stage_explain(best_model, X, feature_cols)

        # ── Stage 9: Recommendations ───────────────────────────────────
        recommendations = self._stage_recommend(
            forecasts, df_clean, anomaly_df, shap_importance
        )
        # ── Stage 10: Visualizations ───────────────────────────────────
        self._stage_visualize(df_clean, forecasts, anomaly_df, shap_importance, metrics)

        # ── Serialize all outputs ──────────────────────────────────────
        self._stage_serialize(
            best_model, fe, forecasts, anomaly_df,
            shap_importance, recommendations, metrics
        )

        elapsed = time.perf_counter() - t_start
        logger.info("=" * 70)
        logger.info("  EnerVision AI Pipeline — COMPLETE in %.1fs", elapsed)
        logger.info("=" * 70)

        def _jsonify_forecast(df):
            """Convert DataFrame with Timestamp index to JSON-safe dict."""
            return {str(k): v for k, v in df.to_dict(orient="index").items()}

        return {
            "best_model": best_model.name,
            "metrics": metrics,
            "forecasts": {k: _jsonify_forecast(v) for k, v in forecasts.items()},
            "anomaly_summary": anomaly_summary,
            "shap_importance": shap_importance.to_dict(orient="records") if shap_importance is not None else [],
            "recommendations": self._rec_engine_instance.to_dict(recommendations),
        }

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _stage_ingest(self) -> pd.DataFrame:
        with PipelineLogger(logger, "Stage 1-2: Ingest + Validate"):
            loader = DataLoader(cfg=self._cfg)
            df_raw = loader.load()
            validator = SchemaValidator(cfg=self._cfg)
            validator.validate(df_raw)
            return df_raw

    def _stage_preprocess(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        with PipelineLogger(logger, "Stage 3: Preprocess"):
            cleaner = DataCleaner(cfg=self._cfg)
            return cleaner.clean(df_raw)

    def _stage_features(self, df_clean: pd.DataFrame, fe: FeatureEngineer) -> pd.DataFrame:
        with PipelineLogger(logger, "Stage 4: Feature Engineering"):
            df_features = fe.transform(df_clean)
            logger.info(
                "Feature matrix: %d rows × %d cols",
                *df_features.shape,
            )
            # Save processed features
            processed_path = os.path.join(
                self._cfg.data.processed_dir, "features.parquet"
            )
            try:
                df_features.to_parquet(processed_path)
                logger.info("Processed features saved → %s", processed_path)
            except Exception as e:
                logger.warning("Could not save parquet: %s", e)
            return df_features

    def _stage_train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        skip_training: bool,
        fe: FeatureEngineer,
    ) -> Tuple:
        with PipelineLogger(logger, "Stage 5: Model Selection"):
            if skip_training and self._serializer.model_exists("best_model"):
                logger.info("Loading pre-trained model from disk.")
                best_model = self._serializer.load_model("best_model")
                metadata = self._serializer.load_metadata()
                metrics = metadata.get("metrics", {})
                return best_model, metrics

            # Chronological train/test split
            fc_cfg = self._cfg.forecasting
            X_train, X_test = time_split(
                X,
                test_size=float(fc_cfg.test_size),
            )
            y_train = y.loc[X_train.index]
            y_test = y.loc[X_test.index]

            selector = ModelSelector(cfg=self._cfg)
            best_model, metrics = selector.run(X_train, y_train, X_test, y_test)
            return best_model, metrics

    def _stage_forecast(
        self,
        df_clean: pd.DataFrame,
        best_model,
        fe: FeatureEngineer,
    ) -> Dict:
        with PipelineLogger(logger, "Stage 6: Forecasting"):
            fg = ForecastGenerator(best_model, fe, cfg=self._cfg)
            return fg.generate(df_clean)

    def _stage_anomaly(self, df_clean: pd.DataFrame) -> Tuple:
        with PipelineLogger(logger, "Stage 7: Anomaly Detection"):
            detector = AnomalyDetector(cfg=self._cfg)
            anomaly_df = detector.detect(df_clean)
            summary = detector.summary(anomaly_df)
            logger.info("Anomaly summary: %s", summary)
            return anomaly_df, summary

    def _stage_explain(
        self, best_model, X: pd.DataFrame, feature_cols: List[str]
    ) -> Optional[pd.DataFrame]:
        with PipelineLogger(logger, "Stage 8: SHAP Explainability"):
            stat_models = {"ARIMA", "SARIMA", "SARIMAX"}
            if best_model.name in stat_models:
                logger.info(
                    "Skipping SHAP for statistical model '%s'.", best_model.name
                )
                return None
            try:
                exp_cfg = self._cfg.explainability
                bg_samples = int(exp_cfg.get("background_samples", 100))
                max_samples = int(exp_cfg.get("max_samples", 500))

                explainer = SHAPExplainer(best_model, feature_cols, cfg=self._cfg)
                X_bg = X.iloc[:bg_samples]
                X_sample = X.iloc[:max_samples]

                explainer.fit(X_bg)
                explainer.explain(X_sample)
                importance_df = explainer.feature_importance()
                explainer.save_plots()
                return importance_df
            except Exception as exc:
                logger.error("SHAP stage failed: %s", exc, exc_info=True)
                return None

    def _stage_recommend(
        self,
        forecasts: Dict,
        df_clean: pd.DataFrame,
        anomaly_df: pd.DataFrame,
        shap_importance: Optional[pd.DataFrame],
    ) -> list:
        with PipelineLogger(logger, "Stage 9: Recommendations"):
            self._rec_engine_instance = RecommendationEngine(cfg=self._cfg)
            # Use 24h forecast for recommendation rules
            fc_24h = forecasts.get("24h")
            return self._rec_engine_instance.generate(
                forecast_df=fc_24h,
                history_df=df_clean,
                anomaly_df=anomaly_df,
                shap_importance_df=shap_importance,
            )

    def _stage_serialize(
        self,
        best_model,
        fe: FeatureEngineer,
        forecasts: Dict,
        anomaly_df: pd.DataFrame,
        shap_importance: Optional[pd.DataFrame],
        recommendations: list,
        metrics: Dict,
    ) -> None:
        with PipelineLogger(logger, "Stage 10: Serialize Outputs"):
            ser = self._serializer
            ser.save_model(best_model, "best_model")
            ser.save_model(fe, "feature_engineer")

            for label, fc_df in forecasts.items():
                ser.save_forecast(fc_df, label)

            ser.save_anomalies(anomaly_df)

            if shap_importance is not None:
                ser.save_shap_importance(shap_importance)

            rec_engine = self._rec_engine_instance
            ser.save_recommendations(rec_engine.to_dict(recommendations))

            # Build metadata
            metadata = {
                "best_model": best_model.name,
                "metrics": {
                    k: {m: round(v, 4) for m, v in vals.items()}
                    for k, vals in metrics.items()
                },
                "target_column": self._cfg.data.target_column,
                "n_recommendations": len(recommendations),
                "forecast_horizons": list(forecasts.keys()),
            }
            ser.save_metadata(metadata)

    def _stage_visualize(
        self,
        df_clean: pd.DataFrame,
        forecasts: Dict,
        anomaly_df: pd.DataFrame,
        shap_importance: Optional[pd.DataFrame],
        metrics: Dict,
    ) -> None:
        with PipelineLogger(logger, "Stage 10: Visualizations"):
            visualizer = Visualizer(cfg=self._cfg)
            target_col = self._cfg.data.target_column
            
            # 1. Model comparison
            if metrics:
                visualizer.plot_model_comparison(metrics)
                
            # 2. Forecasts vs actuals
            for horizon, fc_df in forecasts.items():
                visualizer.plot_forecast(df_clean, fc_df, horizon, target_col)
                
            # 3. Anomalies
            if not anomaly_df.empty:
                visualizer.plot_anomalies(anomaly_df, target_col)
                
            # 4. Feature importance
            if shap_importance is not None:
                visualizer.plot_feature_importance(shap_importance)

