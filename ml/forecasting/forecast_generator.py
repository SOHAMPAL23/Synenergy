"""
EnerVision AI - Forecast Generator
Generates 24-hour, 7-day, and 30-day forecasts from a fitted model.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ml.feature_engineering.feature_pipeline import FeatureEngineer
from ml.forecasting.base_model import BaseModel
from ml.utils.config_loader import config
from ml.utils.helpers import clip_predictions
from ml.utils.logger import get_logger, PipelineLogger

logger = get_logger(__name__)


class ForecastGenerator:
    """
    Uses a trained best model to produce multi-horizon point forecasts.

    Strategy
    --------
    * ML models: iterative one-step-ahead forecasting where the prediction
      from step t is fed back as the lag feature for step t+1.
    * Statistical models: direct n-step forecast via the fitted SARIMAX result.

    Usage::

        fg = ForecastGenerator(best_model, feature_engineer)
        forecasts = fg.generate(history_df)
        # forecasts["24h"], forecasts["7d"], forecasts["30d"]
    """

    _HORIZON_MAP = {
        "24h": "short",
        "7d": "medium",
        "30d": "long",
    }

    def __init__(
        self,
        model: BaseModel,
        feature_engineer: Optional[FeatureEngineer] = None,
        cfg=None,
    ) -> None:
        self._model = model
        self._fe = feature_engineer or FeatureEngineer()
        self._cfg = cfg or config
        fc_cfg = self._cfg.forecasting.forecast_horizons
        self._horizons: Dict[str, int] = {
            "24h": int(fc_cfg.get("short", 24)),
            "7d": int(fc_cfg.get("medium", 168)),
            "30d": int(fc_cfg.get("long", 720)),
        }
        self._target_col: str = self._cfg.data.target_column

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, history_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for all configured horizons.

        Args:
            history_df: Clean historical DataFrame (DatetimeIndex, target + exog cols).

        Returns:
            Dict mapping horizon label → DataFrame with columns
            [timestamp, forecast, lower_bound, upper_bound].
        """
        with PipelineLogger(logger, "ForecastGenerator.generate"):
            forecasts = {}
            for label, horizon in self._horizons.items():
                logger.info("Generating %s (%d-hour) forecast…", label, horizon)
                fc_df = self._forecast_horizon(history_df, horizon)
                forecasts[label] = fc_df
                logger.info(
                    "  %s forecast: min=%.0f | max=%.0f | mean=%.0f",
                    label,
                    fc_df["forecast"].min(),
                    fc_df["forecast"].max(),
                    fc_df["forecast"].mean(),
                )
            return forecasts

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _forecast_horizon(self, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Generate future timestamps and predict energy load."""
        stat_models = {"ARIMA", "SARIMA", "SARIMAX"}

        if self._model.name in stat_models:
            return self._statistical_forecast(history_df, horizon)
        else:
            return self._ml_forecast(history_df, horizon)

    def _ml_forecast(self, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Iterative recursive forecast for ML models.
        Uses the last known data point and rolls forward step by step.
        """
        # Build feature-engineered history
        df_feat = self._fe.transform(history_df.copy())
        feature_cols = self._fe.get_feature_columns(df_feat)

        # Start from the last row of history
        # We'll use a sliding window simulation approach
        sim_df = history_df.copy()

        last_ts = sim_df.index[-1]
        freq = pd.infer_freq(sim_df.index[-10:]) or "H"

        preds_list = []
        timestamps = pd.date_range(
            start=last_ts + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )

        for i, ts in enumerate(timestamps):
            # Add a placeholder row for the next timestamp
            new_row = pd.DataFrame(
                {self._target_col: [np.nan]}, index=[ts]
            )
            # Include exog columns from last known row
            for col in sim_df.columns:
                if col != self._target_col:
                    new_row[col] = sim_df[col].iloc[-1]

            sim_df = pd.concat([sim_df, new_row])

            # Rebuild features on growing window
            try:
                df_feat = self._fe.transform(sim_df.iloc[-max(200, horizon * 2):].copy())
                row = df_feat[feature_cols].iloc[[-1]]
                pred = float(self._model.predict(row)[0])
                pred = max(0.0, pred)  # physical lower bound
            except Exception as exc:
                logger.warning("Step %d prediction failed: %s. Using last known.", i, exc)
                pred = float(sim_df[self._target_col].dropna().iloc[-1])

            # Feed prediction back into history
            sim_df.at[ts, self._target_col] = pred
            preds_list.append(pred)

        preds = np.array(preds_list)
        return self._build_forecast_df(timestamps, preds)

    def _statistical_forecast(self, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Direct multi-step forecast from statistical model."""
        last_ts = history_df.index[-1]
        freq = pd.infer_freq(history_df.index[-10:]) or "H"
        timestamps = pd.date_range(
            start=last_ts + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )

        # For SARIMAX we need future exog values — use last known values
        exog_cols = [c for c in self._cfg.data.exog_columns if c in history_df.columns]
        if exog_cols and self._model.name == "SARIMAX":
            future_exog = pd.DataFrame(
                {col: [history_df[col].iloc[-1]] * horizon for col in exog_cols},
                index=timestamps,
            )
            preds = self._model.predict(future_exog)
        else:
            # Create empty DataFrame with correct length for step count
            dummy = pd.DataFrame(index=timestamps)
            preds = self._model.predict(dummy)

        preds = np.clip(np.asarray(preds), 0, None)
        return self._build_forecast_df(timestamps, preds)

    def _build_forecast_df(self, timestamps, preds: np.ndarray) -> pd.DataFrame:
        """Package forecasts with naive confidence bounds (±10%)."""
        return pd.DataFrame({
            "timestamp": timestamps,
            "forecast": preds,
            "lower_bound": preds * 0.90,
            "upper_bound": preds * 1.10,
        }).set_index("timestamp")
