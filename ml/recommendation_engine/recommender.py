"""
EnerVision AI - Recommendation Engine
Rule-based optimization recommendations from forecast and anomaly data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ml.utils.config_loader import config
from ml.utils.logger import get_logger, PipelineLogger

logger = get_logger(__name__)


@dataclass
class Recommendation:
    category: str
    priority: str          # HIGH / MEDIUM / LOW
    title: str
    description: str
    estimated_saving_pct: float = 0.0
    action_items: List[str] = field(default_factory=list)


class RecommendationEngine:
    """
    Generates rule-based energy optimization recommendations from:
    - Forecast data (peak hours, pattern analysis)
    - Anomaly detection results
    - SHAP feature importance
    - Historical consumption statistics

    Usage::

        engine = RecommendationEngine()
        recommendations = engine.generate(forecast_df, history_df, anomaly_df, shap_importance_df)
    """

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or config
        rec_cfg = self._cfg.recommendation
        self._peak_hours: List[int] = list(rec_cfg.peak_hours)
        self._hvac_threshold: float = float(rec_cfg.hvac_contribution_threshold)
        self._high_pct: float = float(rec_cfg.high_consumption_percentile)
        self._low_pct: float = float(rec_cfg.low_consumption_percentile)
        self._load_shift_threshold: float = float(rec_cfg.load_shift_threshold_mw)
        self._target_col: str = self._cfg.data.target_column

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        forecast_df: Optional[pd.DataFrame] = None,
        history_df: Optional[pd.DataFrame] = None,
        anomaly_df: Optional[pd.DataFrame] = None,
        shap_importance_df: Optional[pd.DataFrame] = None,
    ) -> List[Recommendation]:
        with PipelineLogger(logger, "RecommendationEngine.generate"):
            recs: List[Recommendation] = []

            if forecast_df is not None:
                recs += self._peak_load_rules(forecast_df)
                recs += self._off_peak_opportunity(forecast_df)

            if history_df is not None:
                recs += self._consumption_trend_rules(history_df)
                recs += self._weekend_vs_weekday(history_df)

            if anomaly_df is not None:
                recs += self._anomaly_rules(anomaly_df)

            if shap_importance_df is not None:
                recs += self._shap_driven_rules(shap_importance_df)

            recs += self._general_efficiency_rules()

            recs = self._deduplicate(recs)
            recs = sorted(recs, key=lambda r: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[r.priority])
            logger.info("Generated %d recommendations.", len(recs))
            for r in recs:
                logger.info("  [%s] %s: %s", r.priority, r.category, r.title)
            return recs

    def to_dict(self, recs: List[Recommendation]) -> List[dict]:
        return [
            {
                "category": r.category,
                "priority": r.priority,
                "title": r.title,
                "description": r.description,
                "estimated_saving_pct": r.estimated_saving_pct,
                "action_items": r.action_items,
            }
            for r in recs
        ]

    # ------------------------------------------------------------------
    # Rule groups
    # ------------------------------------------------------------------

    def _peak_load_rules(self, forecast_df: pd.DataFrame) -> List[Recommendation]:
        recs = []
        if "forecast" not in forecast_df.columns:
            return recs

        fc = forecast_df.copy()
        if not isinstance(fc.index, pd.DatetimeIndex):
            return recs

        fc["hour"] = fc.index.hour
        peak_mask = fc["hour"].isin(self._peak_hours)
        peak_avg = fc.loc[peak_mask, "forecast"].mean() if peak_mask.any() else 0
        off_peak_avg = fc.loc[~peak_mask, "forecast"].mean() if (~peak_mask).any() else 0

        if peak_avg > self._load_shift_threshold and off_peak_avg > 0:
            ratio = peak_avg / off_peak_avg
            if ratio > 1.15:
                recs.append(Recommendation(
                    category="Load Shifting",
                    priority="HIGH",
                    title="Shift Loads Away from Peak Hours (3 PM – 6 PM)",
                    description=(
                        f"Forecasted peak consumption ({peak_avg:,.0f} MW) is "
                        f"{ratio:.1f}x higher than off-peak ({off_peak_avg:,.0f} MW). "
                        "Shifting flexible loads reduces peak demand charges."
                    ),
                    estimated_saving_pct=min(15.0, (ratio - 1) * 10),
                    action_items=[
                        "Schedule industrial batch jobs between 10 PM – 6 AM.",
                        "Pre-cool or pre-heat buildings before 3 PM.",
                        "Enable demand-response programs with grid operator.",
                        "Install smart meters with time-of-use tariff plans.",
                    ],
                ))
        return recs

    def _off_peak_opportunity(self, forecast_df: pd.DataFrame) -> List[Recommendation]:
        recs = []
        if "forecast" not in forecast_df.columns:
            return recs
        fc = forecast_df.copy()
        if isinstance(fc.index, pd.DatetimeIndex):
            fc["hour"] = fc.index.hour
            night_mask = fc["hour"].between(0, 5)
            if night_mask.any():
                night_avg = fc.loc[night_mask, "forecast"].mean()
                overall_avg = fc["forecast"].mean()
                if night_avg < overall_avg * 0.75:
                    recs.append(Recommendation(
                        category="Energy Storage",
                        priority="MEDIUM",
                        title="Utilize Off-Peak Hours for Battery Charging",
                        description=(
                            f"Night-time load ({night_avg:,.0f} MW) is "
                            f"{100*(1-night_avg/overall_avg):.0f}% below daily average. "
                            "Charge storage systems during off-peak for daytime discharge."
                        ),
                        estimated_saving_pct=8.0,
                        action_items=[
                            "Install grid-scale battery storage.",
                            "Configure EV fleet charging for 12 AM – 5 AM window.",
                            "Enable automated smart charging algorithms.",
                        ],
                    ))
        return recs

    def _consumption_trend_rules(self, history_df: pd.DataFrame) -> List[Recommendation]:
        recs = []
        if self._target_col not in history_df.columns:
            return recs
        series = history_df[self._target_col].dropna()
        if len(series) < 168:
            return recs

        # Compare last week to previous week
        last_week = series.iloc[-168:].mean()
        prev_week = series.iloc[-336:-168].mean() if len(series) >= 336 else series.mean()
        pct_change = (last_week - prev_week) / (prev_week + 1e-8) * 100

        if pct_change > 10:
            recs.append(Recommendation(
                category="Consumption Monitoring",
                priority="HIGH",
                title=f"Consumption Increased {pct_change:.1f}% Week-over-Week",
                description=(
                    f"Last week average: {last_week:,.0f} MW vs "
                    f"previous week: {prev_week:,.0f} MW. "
                    "Investigate root cause immediately."
                ),
                estimated_saving_pct=pct_change * 0.5,
                action_items=[
                    "Audit new high-consumption equipment activated last week.",
                    "Check HVAC set-points for unintended changes.",
                    "Review occupancy patterns and scheduling.",
                ],
            ))

        high_threshold = np.percentile(series, self._high_pct)
        recent = series.iloc[-24:]
        if recent.mean() > high_threshold:
            recs.append(Recommendation(
                category="Peak Reduction",
                priority="HIGH",
                title="Current Consumption in Top 10% of Historical Range",
                description=(
                    f"Recent 24-hour average ({recent.mean():,.0f} MW) exceeds "
                    f"the {self._high_pct}th percentile ({high_threshold:,.0f} MW)."
                ),
                estimated_saving_pct=12.0,
                action_items=[
                    "Activate demand curtailment programs.",
                    "Reduce non-critical system loads temporarily.",
                    "Alert facility managers for immediate action.",
                ],
            ))
        return recs

    def _weekend_vs_weekday(self, history_df: pd.DataFrame) -> List[Recommendation]:
        recs = []
        if self._target_col not in history_df.columns:
            return recs
        if not isinstance(history_df.index, pd.DatetimeIndex):
            return recs
        df = history_df[[self._target_col]].copy()
        df["is_weekend"] = df.index.dayofweek >= 5
        wd_avg = df.loc[~df["is_weekend"], self._target_col].mean()
        we_avg = df.loc[df["is_weekend"], self._target_col].mean()
        if wd_avg > 0 and we_avg / wd_avg < 0.6:
            recs.append(Recommendation(
                category="Scheduling",
                priority="LOW",
                title="Weekend Load is 40%+ Lower — Optimize Scheduled Maintenance",
                description=(
                    f"Weekday avg: {wd_avg:,.0f} MW | Weekend avg: {we_avg:,.0f} MW. "
                    "Schedule high-energy maintenance tasks on weekends."
                ),
                estimated_saving_pct=5.0,
                action_items=[
                    "Move planned maintenance windows to Saturday/Sunday.",
                    "Schedule data center backups and batch jobs for weekends.",
                ],
            ))
        return recs

    def _anomaly_rules(self, anomaly_df: pd.DataFrame) -> List[Recommendation]:
        recs = []
        if "is_anomaly" not in anomaly_df.columns:
            return recs
        total = len(anomaly_df)
        n_anomalies = int(anomaly_df["is_anomaly"].sum())
        pct = 100 * n_anomalies / (total + 1e-8)
        if pct > 5:
            recs.append(Recommendation(
                category="Anomaly Management",
                priority="HIGH" if pct > 10 else "MEDIUM",
                title=f"{n_anomalies} Anomalous Consumption Events Detected ({pct:.1f}%)",
                description=(
                    "Anomalous readings may indicate equipment malfunction, "
                    "data quality issues, or unauthorized energy use."
                ),
                estimated_saving_pct=min(20.0, pct),
                action_items=[
                    "Investigate timestamps flagged as anomalous.",
                    "Check sensor calibration and data pipelines.",
                    "Verify no unauthorized high-draw equipment is connected.",
                    "Install power quality monitoring systems.",
                ],
            ))
        return recs

    def _shap_driven_rules(self, shap_df: pd.DataFrame) -> List[Recommendation]:
        recs = []
        if "feature" not in shap_df.columns:
            return recs
        top_features = shap_df.head(5)["feature"].tolist()

        if any("rolling" in f for f in top_features):
            recs.append(Recommendation(
                category="Forecasting",
                priority="LOW",
                title="Historical Rolling Averages Strongly Drive Predictions",
                description=(
                    "Rolling mean features are top SHAP contributors, "
                    "indicating strong autocorrelation. Consider ARIMA-based approaches."
                ),
                estimated_saving_pct=0.0,
                action_items=[
                    "Leverage rolling forecast models for operational planning.",
                    "Build automated alerts when rolling average deviates significantly.",
                ],
            ))

        if any("hour" in f for f in top_features):
            recs.append(Recommendation(
                category="Time-of-Use",
                priority="MEDIUM",
                title="Hour-of-Day is a Key Driver — Implement Time-of-Use Tariffs",
                description=(
                    "Hour features rank highly in SHAP importance, "
                    "indicating strong intraday patterns suitable for TOU pricing."
                ),
                estimated_saving_pct=7.0,
                action_items=[
                    "Negotiate time-of-use tariff with utility provider.",
                    "Automate energy-intensive processes to run in cheapest hours.",
                    "Display real-time pricing to building occupants.",
                ],
            ))

        if any("is_holiday" in f or "is_weekend" in f for f in top_features):
            recs.append(Recommendation(
                category="Calendar Optimization",
                priority="LOW",
                title="Calendar Effects Are Significant — Plan Accordingly",
                description=(
                    "Holiday and weekend flags are strong predictors. "
                    "Pre-positioning resources around calendar events reduces waste."
                ),
                estimated_saving_pct=4.0,
                action_items=[
                    "Pre-schedule HVAC setbacks before holidays.",
                    "Coordinate production shutdowns with public holidays.",
                ],
            ))
        return recs

    def _general_efficiency_rules(self) -> List[Recommendation]:
        return [
            Recommendation(
                category="HVAC Optimization",
                priority="MEDIUM",
                title="HVAC Systems: Implement Smart Scheduling",
                description=(
                    "HVAC typically accounts for 30–50% of commercial building energy. "
                    "Smart scheduling and setbacks can deliver significant savings."
                ),
                estimated_saving_pct=15.0,
                action_items=[
                    "Install programmable thermostats with occupancy sensors.",
                    "Set heating/cooling setbacks during unoccupied hours.",
                    "Schedule preventive HVAC maintenance quarterly.",
                    "Consider variable frequency drives (VFDs) for HVAC motors.",
                ],
            ),
            Recommendation(
                category="Renewable Integration",
                priority="MEDIUM",
                title="Increase Renewable Energy Consumption During Solar/Wind Peaks",
                description=(
                    "Grid data shows significant solar and wind generation. "
                    "Aligning high-consumption tasks with renewable peaks reduces carbon footprint."
                ),
                estimated_saving_pct=10.0,
                action_items=[
                    "Monitor real-time renewable generation API.",
                    "Shift flexible loads to coincide with high solar/wind output.",
                    "Consider on-site renewable generation (rooftop solar / wind).",
                ],
            ),
        ]

    def _deduplicate(self, recs: List[Recommendation]) -> List[Recommendation]:
        seen_titles = set()
        unique = []
        for r in recs:
            if r.title not in seen_titles:
                seen_titles.add(r.title)
                unique.append(r)
        return unique
