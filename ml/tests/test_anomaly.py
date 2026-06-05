"""
EnerVision AI - Unit Tests: Anomaly Detection
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ml.anomaly_detection.anomaly_detector import AnomalyDetector


def _make_series_with_anomalies(n: int = 500) -> pd.DataFrame:
    """Return a DataFrame with known anomaly spikes injected."""
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    signal = np.random.normal(50000, 3000, n)
    # Inject obvious anomalies at known positions
    anomaly_positions = [50, 150, 300, 400]
    for pos in anomaly_positions:
        signal[pos] = 200000.0   # extreme high spike
    df = pd.DataFrame(
        {"DE_load_actual_entsoe_transparency": signal},
        index=idx,
    )
    return df, anomaly_positions


class TestAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.detector = AnomalyDetector()
        self.df, self.known_anomalies = _make_series_with_anomalies()

    def test_detect_returns_dataframe(self):
        result = self.detector.detect(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_anomaly_columns_added(self):
        result = self.detector.detect(self.df)
        self.assertIn("is_anomaly", result.columns)
        self.assertIn("anomaly_score", result.columns)

    def test_method_flag_columns_added(self):
        result = self.detector.detect(self.df)
        for method in ["zscore", "iqr", "isolation_forest", "lof", "one_class_svm"]:
            self.assertIn(f"anomaly_{method}", result.columns,
                          f"Flag column for method '{method}' missing")

    def test_detects_known_spikes(self):
        result = self.detector.detect(self.df)
        # Z-Score should catch the 200000 spikes (>> 3-sigma from mean)
        for pos in self.known_anomalies:
            ts = self.df.index[pos]
            self.assertEqual(
                result.loc[ts, "anomaly_zscore"], 1,
                f"Z-Score failed to detect anomaly at position {pos}",
            )

    def test_is_anomaly_is_binary(self):
        result = self.detector.detect(self.df)
        self.assertTrue(result["is_anomaly"].isin([0, 1]).all())

    def test_anomaly_score_between_0_and_1(self):
        result = self.detector.detect(self.df)
        self.assertTrue((result["anomaly_score"] >= 0).all())
        self.assertTrue((result["anomaly_score"] <= 1).all())

    def test_summary_returns_dict(self):
        result = self.detector.detect(self.df)
        summary = self.detector.summary(result)
        self.assertIsInstance(summary, dict)
        self.assertGreater(len(summary), 0)

    def test_output_same_length_as_input(self):
        result = self.detector.detect(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_normal_series_has_low_anomaly_rate(self):
        """A Gaussian series should have an anomaly rate close to contamination %."""
        idx = pd.date_range("2020-01-01", periods=1000, freq="h", tz="UTC")
        normal_df = pd.DataFrame(
            {"DE_load_actual_entsoe_transparency": np.random.normal(50000, 2000, 1000)},
            index=idx,
        )
        result = self.detector.detect(normal_df)
        rate = result["is_anomaly"].mean()
        self.assertLess(rate, 0.3, "Too many anomalies flagged in normal data")

    def test_multivariate_features_scaling_and_alignment(self):
        series = self.df["DE_load_actual_entsoe_transparency"]
        self.df["DE_solar_generation_actual"] = np.random.normal(5000, 1000, len(self.df))
        X_multi = self.detector._get_multivariate_features(self.df, series)
        # Features should include target (1) + cyclical hour (2) + weekend (1) + solar (1) = 5 features
        self.assertEqual(X_multi.shape[1], 5)
        # Features should be standardized
        np.testing.assert_array_almost_equal(X_multi.mean(axis=0), np.zeros(5), decimal=4)
        np.testing.assert_array_almost_equal(X_multi.std(axis=0), np.ones(5), decimal=4)


if __name__ == "__main__":
    unittest.main()
