"""
EnerVision AI - Unit Tests for Inference and Feature Enhancements
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ml.models.serializer import ModelSerializer
from ml.feature_engineering.feature_pipeline import FeatureEngineer


class TestModelSerializer(unittest.TestCase):

    def setUp(self):
        # Create a temp directory for outputs config
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Build mock config loader
        class MockConfig:
            class data:
                models_dir = os.path.join(self.temp_dir.name, "models")
                processed_dir = os.path.join(self.temp_dir.name, "processed")
        
        self.cfg = MockConfig()
        self.ser = ModelSerializer(cfg=self.cfg)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_load_model(self):
        model = LinearRegression()
        model.coef_ = np.array([1.5, 2.5])
        model.intercept_ = 0.5
        
        model_name = "test_model_123"
        self.assertFalse(self.ser.model_exists(model_name))
        
        self.ser.save_model(model, model_name)
        self.assertTrue(self.ser.model_exists(model_name))
        
        loaded = self.ser.load_model(model_name)
        self.assertIsInstance(loaded, LinearRegression)
        self.assertEqual(loaded.intercept_, 0.5)
        np.testing.assert_array_equal(loaded.coef_, model.coef_)

    def test_save_load_metadata(self):
        meta = {"best_model": "XGBoost", "rmse": 45.2}
        meta_name = "test_metadata"
        
        self.ser.save_metadata(meta, name=meta_name)
        loaded = self.ser.load_metadata(name=meta_name)
        
        self.assertEqual(loaded["best_model"], "XGBoost")
        self.assertEqual(loaded["rmse"], 45.2)


class TestFeatureEngineerEnhancements(unittest.TestCase):

    def test_predictive_features_generated(self):
        # Create a sequential time index
        idx = pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")
        df = pd.DataFrame(
            {"DE_load_actual_entsoe_transparency": np.sin(np.arange(200)) * 10 + 50},
            index=idx
        )
        
        fe = FeatureEngineer()
        df_feat = fe.transform(df)
        
        # Test new lag features
        self.assertIn("load_t_2", df_feat.columns)
        self.assertIn("load_t_48", df_feat.columns)
        self.assertIn("load_t_168", df_feat.columns)
        
        # Test new difference features
        self.assertIn("load_diff_1", df_feat.columns)
        self.assertIn("load_diff_24", df_feat.columns)
        self.assertIn("load_diff_168", df_feat.columns)
        
        # Test weekend-hour interaction
        self.assertIn("hour_is_weekend", df_feat.columns)
        
        # Test rolling window additions (24 and 168 should be included)
        self.assertIn("rolling_mean_24", df_feat.columns)
        self.assertIn("rolling_mean_168", df_feat.columns)
        
        # Non-null assertions
        self.assertEqual(df_feat.isnull().sum().sum(), 0)
        # Dropped maximum lag rows (169 rows, leaving 31 rows)
        self.assertEqual(len(df_feat), 200 - 169)


if __name__ == "__main__":
    unittest.main()
