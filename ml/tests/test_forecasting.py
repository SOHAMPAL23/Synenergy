"""
EnerVision AI - Unit Tests: Forecasting Models
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ml.feature_engineering.feature_pipeline import FeatureEngineer
from ml.forecasting.linear_model import LinearRegressionModel
from ml.forecasting.random_forest_model import RandomForestModel
from ml.forecasting.xgboost_model import XGBoostModel
from ml.utils.helpers import compute_metrics


def _make_feature_df(n: int = 600) -> tuple:
    """Return (X_train, y_train, X_test, y_test) with engineered features."""
    idx = pd.date_range("2019-01-01", periods=n, freq="h", tz="UTC")
    target = (
        50000
        + 10000 * np.sin(2 * np.pi * np.arange(n) / 24)   # daily cycle
        + np.random.normal(0, 1000, n)
    )
    solar = np.random.uniform(0, 15000, n)
    wind = np.random.uniform(5000, 35000, n)
    raw_df = pd.DataFrame(
        {
            "DE_load_actual_entsoe_transparency": target,
            "DE_solar_generation_actual": solar,
            "DE_wind_generation_actual": wind,
        },
        index=idx,
    )
    fe = FeatureEngineer()
    df = fe.transform(raw_df)
    feat_cols = fe.get_feature_columns(df)
    X = df[feat_cols]
    y = df["DE_load_actual_entsoe_transparency"]
    split = int(len(X) * 0.8)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


class TestLinearRegressionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X_tr, cls.y_tr, cls.X_te, cls.y_te = _make_feature_df()
        cls.model = LinearRegressionModel()

    def test_fit_returns_self(self):
        result = self.model.fit(self.X_tr, self.y_tr)
        self.assertIs(result, self.model)

    def test_is_fitted_after_fit(self):
        self.model.fit(self.X_tr, self.y_tr)
        self.assertTrue(self.model._is_fitted)

    def test_predict_shape(self):
        self.model.fit(self.X_tr, self.y_tr)
        preds = self.model.predict(self.X_te)
        self.assertEqual(preds.shape[0], len(self.X_te))

    def test_predict_positive_values(self):
        self.model.fit(self.X_tr, self.y_tr)
        preds = self.model.predict(self.X_te)
        # Most energy predictions should be positive
        self.assertGreater((preds > 0).mean(), 0.8)

    def test_evaluate_returns_metrics(self):
        self.model.fit(self.X_tr, self.y_tr)
        metrics = self.model.evaluate(self.X_te, self.y_te)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("mape", metrics)
        self.assertGreater(metrics["rmse"], 0)

    def test_name(self):
        self.assertEqual(self.model.name, "LinearRegression")


class TestRandomForestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X_tr, cls.y_tr, cls.X_te, cls.y_te = _make_feature_df()
        cls.model = RandomForestModel()

    def test_fit_predict_cycle(self):
        self.model.fit(self.X_tr, self.y_tr)
        preds = self.model.predict(self.X_te)
        self.assertEqual(len(preds), len(self.X_te))

    def test_feature_importances_shape(self):
        self.model.fit(self.X_tr, self.y_tr)
        imp = self.model.feature_importances_
        self.assertEqual(len(imp), self.X_tr.shape[1])

    def test_feature_importances_sum_to_one(self):
        self.model.fit(self.X_tr, self.y_tr)
        imp = self.model.feature_importances_
        self.assertAlmostEqual(imp.sum(), 1.0, places=5)

    def test_name(self):
        self.assertEqual(self.model.name, "RandomForest")


class TestXGBoostModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X_tr, cls.y_tr, cls.X_te, cls.y_te = _make_feature_df()
        cls.model = XGBoostModel()

    def test_fit_predict_cycle(self):
        self.model.fit(self.X_tr, self.y_tr)
        preds = self.model.predict(self.X_te)
        self.assertEqual(len(preds), len(self.X_te))

    def test_xgboost_beats_naive(self):
        self.model.fit(self.X_tr, self.y_tr)
        preds = self.model.predict(self.X_te)
        xgb_rmse = compute_metrics(self.y_te.values, preds)["rmse"]
        naive_rmse = compute_metrics(
            self.y_te.values,
            np.full(len(self.y_te), self.y_tr.mean())
        )["rmse"]
        self.assertLess(xgb_rmse, naive_rmse,
                        "XGBoost should beat the naive mean predictor")

    def test_name(self):
        self.assertEqual(self.model.name, "XGBoost")


class TestMetricsHelpers(unittest.TestCase):

    def test_rmse_zero_for_perfect(self):
        from ml.utils.helpers import rmse
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(rmse(y, y), 0.0)

    def test_mae_zero_for_perfect(self):
        from ml.utils.helpers import mae
        y = np.array([10.0, 20.0, 30.0])
        self.assertAlmostEqual(mae(y, y), 0.0)

    def test_mape_zero_for_perfect(self):
        from ml.utils.helpers import mape
        y = np.array([100.0, 200.0, 300.0])
        self.assertAlmostEqual(mape(y, y), 0.0, places=3)

    def test_rmse_known_value(self):
        from ml.utils.helpers import rmse
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        expected = np.sqrt((9 + 16) / 2)
        self.assertAlmostEqual(rmse(y_true, y_pred), expected, places=5)


if __name__ == "__main__":
    unittest.main()
