"""
EnerVision AI - Unit Tests: Feature Engineering
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ml.feature_engineering.feature_pipeline import FeatureEngineer


def _make_clean_df(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="H", tz="UTC")
    target = np.random.uniform(30000, 70000, n)
    solar = np.random.uniform(0, 20000, n)
    wind = np.random.uniform(5000, 40000, n)
    return pd.DataFrame(
        {
            "DE_load_actual_entsoe_transparency": target,
            "DE_solar_generation_actual": solar,
            "DE_wind_generation_actual": wind,
        },
        index=idx,
    )


class TestFeatureEngineer(unittest.TestCase):

    def setUp(self):
        self.fe = FeatureEngineer()
        self.df = _make_clean_df()

    def test_transform_returns_dataframe(self):
        result = self.fe.transform(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_time_features_present(self):
        result = self.fe.transform(self.df)
        for col in ["hour", "day", "month", "week", "quarter", "season",
                    "is_weekend", "is_holiday"]:
            self.assertIn(col, result.columns, f"Missing time feature: {col}")

    def test_lag_features_present(self):
        result = self.fe.transform(self.df)
        for lag in [1, 24, 168]:
            col = f"load_t_{lag}"
            self.assertIn(col, result.columns, f"Missing lag feature: {col}")

    def test_rolling_features_present(self):
        result = self.fe.transform(self.df)
        for window in [7, 30]:
            self.assertIn(f"rolling_mean_{window}", result.columns)
            self.assertIn(f"rolling_std_{window}", result.columns)

    def test_cyclical_features_present(self):
        result = self.fe.transform(self.df)
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            self.assertIn(col, result.columns)

    def test_no_nan_after_transform(self):
        result = self.fe.transform(self.df)
        self.assertEqual(result.isnull().sum().sum(), 0,
                         "NaN values remain after transform+dropna")

    def test_hour_range(self):
        result = self.fe.transform(self.df)
        self.assertTrue((result["hour"] >= 0).all())
        self.assertTrue((result["hour"] <= 23).all())

    def test_season_range(self):
        result = self.fe.transform(self.df)
        self.assertTrue(result["season"].isin([0, 1, 2, 3]).all())

    def test_is_weekend_binary(self):
        result = self.fe.transform(self.df)
        self.assertTrue(result["is_weekend"].isin([0, 1]).all())

    def test_get_feature_columns_excludes_target(self):
        result = self.fe.transform(self.df)
        feat_cols = self.fe.get_feature_columns(result)
        self.assertNotIn("DE_load_actual_entsoe_transparency", feat_cols)
        self.assertGreater(len(feat_cols), 0)

    def test_row_count_reduced_by_lags(self):
        result = self.fe.transform(self.df)
        # 168-hour lag means at least 168 rows are dropped
        self.assertLess(len(result), len(self.df))


if __name__ == "__main__":
    unittest.main()
