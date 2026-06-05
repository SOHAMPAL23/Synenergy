"""
EnerVision AI - Unit Tests: Ingestion
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ml.ingestion.schema_validator import SchemaValidator, SchemaValidationError


def _make_df(n=200, null_target=False) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    target = np.random.uniform(30000, 70000, n)
    if null_target:
        target[:] = np.nan
    return pd.DataFrame(
        {"DE_load_actual_entsoe_transparency": target}, index=idx
    )


class TestSchemaValidator(unittest.TestCase):

    def setUp(self):
        self.validator = SchemaValidator()

    def test_valid_dataframe_passes(self):
        df = _make_df()
        self.assertIsNone(self.validator.validate(df))  # should not raise

    def test_non_datetime_index_fails(self):
        df = _make_df()
        df.index = range(len(df))
        with self.assertRaises(SchemaValidationError):
            self.validator.validate(df)

    def test_missing_required_column_fails(self):
        df = _make_df()
        df = df.drop(columns=["DE_load_actual_entsoe_transparency"])
        with self.assertRaises(SchemaValidationError):
            self.validator.validate(df)

    def test_too_few_rows_fails(self):
        df = _make_df(n=5)
        with self.assertRaises(SchemaValidationError):
            self.validator.validate(df)

    def test_all_null_target_fails(self):
        df = _make_df(null_target=True)
        with self.assertRaises(SchemaValidationError):
            self.validator.validate(df)

    def test_report_returns_dict(self):
        df = _make_df()
        report = self.validator.report(df)
        self.assertIn("total_rows", report)
        self.assertIn("missing_per_column", report)
        self.assertEqual(report["total_rows"], 200)


class TestDataCleaner(unittest.TestCase):

    def _make_dirty_df(self) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=300, freq="h", tz="UTC")
        data = np.random.uniform(30000, 70000, 300)
        # Introduce missing values
        data[10:15] = np.nan
        # Introduce outliers
        data[50] = 999999.0
        data[100] = -99999.0
        df = pd.DataFrame({"DE_load_actual_entsoe_transparency": data}, index=idx)
        # Introduce a duplicate index
        df = pd.concat([df, df.iloc[[0]]])
        return df

    def test_clean_removes_duplicates(self):
        from ml.preprocessing.cleaner import DataCleaner
        cleaner = DataCleaner()
        df_dirty = self._make_dirty_df()
        df_clean = cleaner.clean(df_dirty)
        self.assertFalse(df_clean.index.duplicated().any())

    def test_clean_fills_missing(self):
        from ml.preprocessing.cleaner import DataCleaner
        cleaner = DataCleaner()
        df_dirty = self._make_dirty_df()
        df_clean = cleaner.clean(df_dirty)
        self.assertEqual(df_clean["DE_load_actual_entsoe_transparency"].isnull().sum(), 0)

    def test_clean_caps_outliers(self):
        from ml.preprocessing.cleaner import DataCleaner
        cleaner = DataCleaner()
        df_dirty = self._make_dirty_df()
        df_clean = cleaner.clean(df_dirty)
        # After IQR capping outlier of 999999 should be reduced
        self.assertLess(df_clean["DE_load_actual_entsoe_transparency"].max(), 999999.0)


if __name__ == "__main__":
    unittest.main()
