"""
EnerVision AI - SHAP Explainability
Feature importance, waterfall, summary, and local explanation plots.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.forecasting.base_model import BaseModel
from ml.utils.config_loader import config
from ml.utils.helpers import ensure_dir
from ml.utils.logger import get_logger, PipelineLogger

logger = get_logger(__name__)


class SHAPExplainer:
    """
    Wraps SHAP to produce global and local explanations for fitted ML models.

    Supported model types
    ---------------------
    * XGBoost / Random Forest  → TreeExplainer (fast)
    * Linear models            → LinearExplainer
    * Any other                → KernelExplainer (slow, sample-based)

    Usage::

        exp = SHAPExplainer(best_model, feature_cols)
        exp.fit(X_background)
        exp.explain(X_sample)
        exp.save_plots()
    """

    def __init__(
        self,
        model: BaseModel,
        feature_cols: List[str],
        cfg=None,
    ) -> None:
        self._model = model
        self._feature_cols = feature_cols
        self._cfg = cfg or config
        exp_cfg = self._cfg.explainability
        self._max_samples: int = int(exp_cfg.get("max_samples", 500))
        self._background_samples: int = int(exp_cfg.get("background_samples", 100))
        self._top_n: int = int(exp_cfg.get("plot_top_features", 20))
        self._save_plots: bool = bool(exp_cfg.get("save_plots", True))
        self._plots_dir: str = self._cfg.data.plots_dir
        self._explainer = None
        self._shap_values: Optional[np.ndarray] = None
        self._X_explained: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_background: pd.DataFrame) -> "SHAPExplainer":
        """Build the SHAP explainer using background data for expectation."""
        with PipelineLogger(logger, "SHAPExplainer.fit"):
            X_bg = X_background[self._feature_cols].iloc[:self._background_samples]
            model_name = self._model.name

            if model_name in ("XGBoost", "RandomForest"):
                inner = getattr(self._model, "_model", None)
                try:
                    self._explainer = shap.TreeExplainer(inner)
                    logger.info("Using TreeExplainer for '%s'.", model_name)
                except Exception as tree_err:
                    logger.warning(
                        "TreeExplainer failed (%s); falling back to KernelExplainer.", tree_err
                    )
                    predict_fn = lambda x: self._model.predict(
                        pd.DataFrame(x, columns=self._feature_cols)
                    )
                    self._explainer = shap.KernelExplainer(predict_fn, X_bg)
                    logger.info("Using KernelExplainer (fallback) for '%s'.", model_name)
            elif model_name == "LinearRegression":
                inner = getattr(self._model, "_model", None)
                self._explainer = shap.LinearExplainer(inner, X_bg)
                logger.info("Using LinearExplainer for '%s'.", model_name)
            else:
                predict_fn = lambda x: self._model.predict(pd.DataFrame(x, columns=self._feature_cols))
                self._explainer = shap.KernelExplainer(predict_fn, X_bg)
                logger.info("Using KernelExplainer for '%s'.", model_name)

        return self

    def explain(self, X_sample: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for a sample set; stores internally."""
        if self._explainer is None:
            raise RuntimeError("Call fit() before explain().")

        X = X_sample[self._feature_cols].iloc[:self._max_samples]
        logger.info("Computing SHAP values for %d samples…", len(X))

        sv = self._explainer(X)
        if hasattr(sv, "values"):
            self._shap_values = sv.values
            self._shap_obj = sv
        else:
            self._shap_values = sv
            self._shap_obj = None
        self._X_explained = X
        logger.info("SHAP values computed. Shape: %s", self._shap_values.shape)
        return self._shap_values

    def feature_importance(self) -> pd.DataFrame:
        """Return mean |SHAP value| per feature, sorted descending."""
        self._check_explained()
        mean_abs = np.abs(self._shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self._feature_cols,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return df

    def local_explanation(self, idx: int) -> dict:
        """Return SHAP values for a single sample row."""
        self._check_explained()
        row = self._X_explained.iloc[idx]
        shap_row = self._shap_values[idx]
        return dict(zip(self._feature_cols, shap_row))

    def save_plots(self) -> None:
        """Save all SHAP plots to the plots directory."""
        self._check_explained()
        ensure_dir(self._plots_dir)

        self._save_summary_plot()
        self._save_bar_plot()
        self._save_waterfall_plot()
        logger.info("All SHAP plots saved to '%s'.", self._plots_dir)

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _save_summary_plot(self) -> None:
        path = os.path.join(self._plots_dir, "shap_summary_plot.png")
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self._shap_values,
                self._X_explained,
                feature_names=self._feature_cols,
                max_display=self._top_n,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Summary plot saved: %s", path)
        except Exception as e:
            logger.warning("Summary plot failed: %s", e)

    def _save_bar_plot(self) -> None:
        path = os.path.join(self._plots_dir, "shap_feature_importance.png")
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                self._shap_values,
                self._X_explained,
                feature_names=self._feature_cols,
                plot_type="bar",
                max_display=self._top_n,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Feature importance bar plot saved: %s", path)
        except Exception as e:
            logger.warning("Bar plot failed: %s", e)

    def _save_waterfall_plot(self) -> None:
        if self._shap_obj is None:
            logger.warning("Waterfall plot requires Explanation object (TreeExplainer/LinearExplainer).")
            return
        path = os.path.join(self._plots_dir, "shap_waterfall_plot.png")
        try:
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(self._shap_obj[0], show=False)
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Waterfall plot saved: %s", path)
        except Exception as e:
            logger.warning("Waterfall plot failed: %s", e)

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _check_explained(self) -> None:
        if self._shap_values is None:
            raise RuntimeError("Call explain() before accessing SHAP results.")
