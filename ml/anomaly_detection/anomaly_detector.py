"""
EnerVision AI - Anomaly Detection
Five methods: Z-Score, IQR, Isolation Forest, LOF, One-Class SVM.
"""

from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from ml.utils.config_loader import config
from ml.utils.logger import get_logger, PipelineLogger

logger = get_logger(__name__)


class AnomalyDetector:
    """
    Detects anomalies using five independent methods with ensemble consensus.
    Flags: anomaly_<method>=1 if anomaly, anomaly_score=fraction of methods,
    is_anomaly=1 if majority vote.
    """

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or config
        ad_cfg = self._cfg.anomaly_detection
        self._methods: List[str] = list(ad_cfg.methods)
        self._zscore_threshold: float = float(ad_cfg.zscore_threshold)
        self._iqr_mult: float = float(ad_cfg.iqr_multiplier)
        self._target_col: str = self._cfg.data.target_column
        iso = ad_cfg.isolation_forest
        self._iso_contamination = float(iso.get("contamination", 0.05))
        self._iso_n_estimators = int(iso.get("n_estimators", 100))
        self._iso_seed = int(iso.get("random_state", 42))
        lof = ad_cfg.lof
        self._lof_neighbors = int(lof.get("n_neighbors", 20))
        self._lof_contamination = float(lof.get("contamination", 0.05))
        svm = ad_cfg.one_class_svm
        self._svm_nu = float(svm.get("nu", 0.05))
        self._svm_kernel = str(svm.get("kernel", "rbf"))
        self._svm_gamma = str(svm.get("gamma", "scale"))

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        with PipelineLogger(logger, "AnomalyDetector.detect"):
            result = df.copy()
            series = df[self._target_col].dropna()
            X_2d = series.values.reshape(-1, 1)
            flag_cols: List[str] = []
            for method in self._methods:
                col_name = f"anomaly_{method}"
                try:
                    flags = self._run_method(method, series, X_2d)
                    flag_series = pd.Series(0, index=df.index, name=col_name)
                    flag_series[series.index] = flags
                    result[col_name] = flag_series
                    flag_cols.append(col_name)
                    logger.info("  %-20s → %d anomalies (%.1f%%)",
                                method, int(flags.sum()), 100*flags.sum()/len(series))
                except Exception as exc:
                    logger.error("Method '%s' failed: %s", method, exc, exc_info=True)
            if flag_cols:
                result["anomaly_score"] = result[flag_cols].mean(axis=1)
                result["is_anomaly"] = (result["anomaly_score"] >= 0.5).astype(int)
                total = int(result["is_anomaly"].sum())
                logger.info("Ensemble anomalies: %d / %d (%.1f%%)",
                            total, len(result), 100*total/len(result))
            return result

    def summary(self, result_df: pd.DataFrame) -> Dict:
        return {col: int(result_df[col].sum())
                for col in result_df.columns
                if col.startswith("anomaly_") and col != "anomaly_score"}

    def _run_method(self, method: str, series: pd.Series, X: np.ndarray) -> np.ndarray:
        dispatch = {
            "zscore": lambda: self._zscore(series),
            "iqr": lambda: self._iqr(series),
            "isolation_forest": lambda: self._isolation_forest(X),
            "lof": lambda: self._lof(X),
            "one_class_svm": lambda: self._one_class_svm(X),
        }
        fn = dispatch.get(method.lower())
        if fn is None:
            raise ValueError(f"Unknown anomaly method: '{method}'")
        return fn()

    def _zscore(self, series: pd.Series) -> np.ndarray:
        z = np.abs((series.values - series.mean()) / (series.std() + 1e-8))
        return (z > self._zscore_threshold).astype(int)

    def _iqr(self, series: pd.Series) -> np.ndarray:
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        return ((series < Q1 - self._iqr_mult*IQR) | (series > Q3 + self._iqr_mult*IQR)).astype(int).values

    def _isolation_forest(self, X: np.ndarray) -> np.ndarray:
        clf = IsolationForest(n_estimators=self._iso_n_estimators,
                              contamination=self._iso_contamination,
                              random_state=self._iso_seed)
        return (clf.fit_predict(X) == -1).astype(int)

    def _lof(self, X: np.ndarray) -> np.ndarray:
        clf = LocalOutlierFactor(n_neighbors=self._lof_neighbors,
                                 contamination=self._lof_contamination)
        return (clf.fit_predict(X) == -1).astype(int)

    def _one_class_svm(self, X: np.ndarray) -> np.ndarray:
        X_sc = StandardScaler().fit_transform(X)
        clf = OneClassSVM(nu=self._svm_nu, kernel=self._svm_kernel, gamma=self._svm_gamma)
        return (clf.fit_predict(X_sc) == -1).astype(int)
