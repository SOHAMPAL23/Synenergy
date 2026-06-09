"""
EnerVision AI - Pipeline Visualizer
Generates matplotlib plots for each pipeline stage:
  - Model comparison bar chart
  - Forecast vs actuals line chart
  - Anomaly scatter plot
  - SHAP feature importance bar chart

All plots are saved to the configured plots directory.
"""

import logging
import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from ml.utils.config_loader import config as _default_config
from ml.utils.helpers import ensure_dir

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Generates and saves pipeline visualization plots.

    Usage::

        viz = Visualizer(cfg=config)
        viz.plot_model_comparison(metrics)
        viz.plot_forecast(df_clean, fc_df, "24h", "DE_load_actual_entsoe_transparency")
    """

    PALETTE = {
        "primary": "#3b82f6",
        "secondary": "#22d3ee",
        "accent": "#8b5cf6",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "success": "#22c55e",
        "bg": "#0f1117",
        "surface": "#1e2340",
        "text": "#e2e8f0",
        "muted": "#64748b",
    }

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or _default_config
        self._plots_dir: str = self._cfg.data.plots_dir
        ensure_dir(self._plots_dir)
        self._apply_dark_theme()

    # ──────────────────────────────────────────────────────────────────────
    # Theme
    # ──────────────────────────────────────────────────────────────────────

    def _apply_dark_theme(self) -> None:
        """Apply a consistent dark theme to all matplotlib plots."""
        plt.rcParams.update({
            "figure.facecolor":  self.PALETTE["bg"],
            "axes.facecolor":    self.PALETTE["surface"],
            "axes.edgecolor":    "#2d3748",
            "axes.labelcolor":   self.PALETTE["text"],
            "xtick.color":       self.PALETTE["muted"],
            "ytick.color":       self.PALETTE["muted"],
            "text.color":        self.PALETTE["text"],
            "grid.color":        "#2d3748",
            "grid.linestyle":    "--",
            "grid.alpha":        0.5,
            "legend.facecolor":  self.PALETTE["surface"],
            "legend.edgecolor":  "#2d3748",
            "font.family":       "sans-serif",
            "font.size":         10,
        })

    # ──────────────────────────────────────────────────────────────────────
    # 1. Model Comparison
    # ──────────────────────────────────────────────────────────────────────

    def plot_model_comparison(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """
        Bar chart comparing RMSE across all trained models.

        Args:
            metrics: {model_name: {rmse, mae, mape}} dict from ModelSelector.

        Returns:
            Absolute path to saved PNG.
        """
        if not metrics:
            logger.warning("No metrics provided; skipping model comparison plot.")
            return ""

        try:
            models = list(metrics.keys())
            rmse_vals = [metrics[m].get("rmse", 0.0) for m in models]
            mae_vals = [metrics[m].get("mae", 0.0) for m in models]

            x = range(len(models))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 5))
            bars1 = ax.bar([i - width / 2 for i in x], rmse_vals, width,
                           label="RMSE", color=self.PALETTE["primary"], alpha=0.85)
            bars2 = ax.bar([i + width / 2 for i in x], mae_vals, width,
                           label="MAE", color=self.PALETTE["secondary"], alpha=0.85)

            ax.set_xticks(list(x))
            ax.set_xticklabels(models, rotation=15, ha="right")
            ax.set_ylabel("Error (MW)")
            ax.set_title("Model Performance Comparison — RMSE & MAE")
            ax.legend()
            ax.grid(axis="y")

            # Annotate best model
            best_idx = rmse_vals.index(min(rmse_vals))
            bars1[best_idx].set_edgecolor(self.PALETTE["warning"])
            bars1[best_idx].set_linewidth(2)

            fig.tight_layout()
            path = os.path.join(self._plots_dir, "model_comparison.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved model comparison plot → %s", path)
            return path

        except Exception as exc:
            logger.error("plot_model_comparison failed: %s", exc, exc_info=True)
            return ""

    # ──────────────────────────────────────────────────────────────────────
    # 2. Forecast vs Actuals
    # ──────────────────────────────────────────────────────────────────────

    def plot_forecast(
        self,
        df_actual: pd.DataFrame,
        fc_df: pd.DataFrame,
        horizon: str,
        target_col: str,
    ) -> str:
        """
        Line chart of actual energy consumption vs forecast with CI band.

        Args:
            df_actual: Cleaned historical DataFrame with DatetimeIndex.
            fc_df:     Forecast DataFrame with columns [forecast, lower_bound, upper_bound].
            horizon:   Label string (e.g. "24h", "7d", "30d").
            target_col: Column name of the target variable.

        Returns:
            Absolute path to saved PNG.
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 5))

            # Plot last N actuals for context
            n_context = {"24h": 48, "7d": 168, "30d": 336}.get(horizon, 48)
            actuals = df_actual[target_col].iloc[-n_context:]
            ax.plot(
                actuals.index, actuals.values,
                color=self.PALETTE["muted"], linewidth=1.2,
                label="Actual", alpha=0.8,
            )

            # Forecast line
            ax.plot(
                fc_df.index, fc_df["forecast"],
                color=self.PALETTE["primary"], linewidth=2,
                label=f"Forecast ({horizon})",
            )

            # Confidence interval
            ax.fill_between(
                fc_df.index,
                fc_df["lower_bound"],
                fc_df["upper_bound"],
                color=self.PALETTE["secondary"], alpha=0.15,
                label="95% CI",
            )

            ax.set_xlabel("Time")
            ax.set_ylabel(f"{target_col} (MW)")
            ax.set_title(f"Energy Forecast — {horizon} Horizon")
            ax.legend(loc="upper left")
            ax.grid(True)

            if hasattr(fc_df.index, "strftime"):
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
                fig.autofmt_xdate()

            fig.tight_layout()
            path = os.path.join(self._plots_dir, f"forecast_{horizon}.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved forecast plot (%s) → %s", horizon, path)
            return path

        except Exception as exc:
            logger.error("plot_forecast(%s) failed: %s", horizon, exc, exc_info=True)
            return ""

    # ──────────────────────────────────────────────────────────────────────
    # 3. Anomaly Detection
    # ──────────────────────────────────────────────────────────────────────

    def plot_anomalies(
        self,
        anomaly_df: pd.DataFrame,
        target_col: str,
    ) -> str:
        """
        Scatter plot of energy timeseries with anomaly points highlighted.

        Args:
            anomaly_df: DataFrame with columns [target_col, is_anomaly, anomaly_score].
            target_col: Column name of the target variable.

        Returns:
            Absolute path to saved PNG.
        """
        try:
            if "is_anomaly" not in anomaly_df.columns:
                logger.warning("No 'is_anomaly' column; skipping anomaly plot.")
                return ""

            normal = anomaly_df[anomaly_df["is_anomaly"] == 0]
            anomalous = anomaly_df[anomaly_df["is_anomaly"] == 1]

            fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

            # Top: timeseries + anomalies
            ax = axes[0]
            ax.plot(
                normal.index, normal[target_col],
                color=self.PALETTE["muted"], linewidth=0.8,
                label="Normal", alpha=0.7,
            )
            if not anomalous.empty:
                ax.scatter(
                    anomalous.index, anomalous[target_col],
                    color=self.PALETTE["danger"], s=25, zorder=5,
                    label=f"Anomaly ({len(anomalous)})", alpha=0.9,
                )
            ax.set_ylabel(f"{target_col} (MW)")
            ax.set_title(f"Anomaly Detection — {len(anomalous)} Anomalies Detected")
            ax.legend(loc="upper left")
            ax.grid(True)

            # Bottom: anomaly score
            ax2 = axes[1]
            if "anomaly_score" in anomaly_df.columns:
                ax2.fill_between(
                    anomaly_df.index,
                    anomaly_df["anomaly_score"],
                    color=self.PALETTE["danger"], alpha=0.4,
                )
                ax2.axhline(0.5, color=self.PALETTE["warning"], linestyle="--", linewidth=1)
                ax2.set_ylabel("Anomaly Score")
                ax2.set_ylim(0, 1)
                ax2.grid(True)

            if hasattr(anomaly_df.index, "strftime"):
                for ax_ in axes:
                    ax_.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
                fig.autofmt_xdate()

            fig.tight_layout()
            path = os.path.join(self._plots_dir, "anomaly_detection.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved anomaly plot → %s", path)
            return path

        except Exception as exc:
            logger.error("plot_anomalies failed: %s", exc, exc_info=True)
            return ""

    # ──────────────────────────────────────────────────────────────────────
    # 4. SHAP Feature Importance
    # ──────────────────────────────────────────────────────────────────────

    def plot_feature_importance(
        self,
        shap_importance: pd.DataFrame,
        top_n: int = 15,
    ) -> str:
        """
        Horizontal bar chart of SHAP feature importance.

        Args:
            shap_importance: DataFrame with columns [feature, mean_abs_shap].
            top_n:           Number of top features to display.

        Returns:
            Absolute path to saved PNG.
        """
        try:
            if shap_importance is None or shap_importance.empty:
                logger.warning("Empty SHAP importance; skipping plot.")
                return ""

            df = shap_importance.head(top_n).sort_values("mean_abs_shap")
            colors = [
                self.PALETTE["primary"] if i >= len(df) - 3
                else self.PALETTE["secondary"]
                for i in range(len(df))
            ]

            fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
            bars = ax.barh(df["feature"], df["mean_abs_shap"], color=colors, alpha=0.85)

            ax.set_xlabel("Mean |SHAP| Value")
            ax.set_title(f"SHAP Feature Importance — Top {len(df)} Features")
            ax.grid(axis="x")

            # Annotate values
            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{width:.1f}", va="center", fontsize=8,
                    color=self.PALETTE["text"],
                )

            fig.tight_layout()
            path = os.path.join(self._plots_dir, "shap_importance.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved SHAP importance plot → %s", path)
            return path

        except Exception as exc:
            logger.error("plot_feature_importance failed: %s", exc, exc_info=True)
            return ""
