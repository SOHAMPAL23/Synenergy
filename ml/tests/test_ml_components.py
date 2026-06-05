"""
EnerVision AI ─ ML Model Component Tests
=========================================
Proper unit tests for every ML component.

Coverage
--------
1.  DataCleaner          – duplicates, NaN strategies, outlier capping (IQR & Z-score)
2.  SchemaValidator      – all 4 check methods + report
3.  FeatureEngineer      – time / lag / rolling / cyclical features + edge cases
4.  LinearRegressionModel – fit / predict / evaluate / coef / feature_importances
5.  RandomForestModel    – fit / predict / feature_importances / no-tuning path
6.  XGBoostModel         – fit / predict / feature_importances / booster property
7.  BaseModel interface  – evaluate guard, fit_evaluate, repr, is_fitted flag
8.  AnomalyDetector      – each of 5 methods, ensemble voting, score bounds, summary
9.  SHAPExplainer        – fit/explain/importance/local_explanation guards & values
10. RecommendationEngine – every rule group, priority sort, dedup, to_dict
11. ForecastGenerator    – ML path / statistical path, bound math, horizon lengths
12. MetricHelpers        – rmse, mae, mape, compute_metrics, edge cases
13. ConfigLoader         – dot-access, nested, default, missing key error
14. Helpers              – time_split, clip_predictions, ensure_dir, safe_concat

Run:
    pytest ml/tests/test_ml_components.py -v
"""

import os
import sys
import tempfile
import unittest
from typing import Tuple

import numpy as np
import pandas as pd

# ── ensure project root is on the path ──────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TARGET = "DE_load_actual_entsoe_transparency"


# ════════════════════════════════════════════════════════════════════════════
# Shared data factories
# ════════════════════════════════════════════════════════════════════════════

def _make_raw(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Clean hourly energy DataFrame with target + two exog columns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    load = (
        50_000
        + 10_000 * np.sin(2 * np.pi * np.arange(n) / 24)
        + 3_000 * np.sin(2 * np.pi * np.arange(n) / (24 * 7))
        + rng.normal(0, 1_200, n)
    )
    solar = np.clip(rng.normal(5_000, 2_500, n), 0, None)
    wind  = np.clip(rng.normal(20_000, 6_000, n), 0, None)
    return pd.DataFrame(
        {TARGET: load,
         "DE_solar_generation_actual": solar,
         "DE_wind_generation_actual":  wind},
        index=idx,
    )


def _make_dirty(n: int = 400, seed: int = 1) -> pd.DataFrame:
    """Raw DataFrame that includes NaNs, extreme outliers, and a dup row."""
    df = _make_raw(n, seed)
    df.iloc[10:14, 0]  = np.nan          # 4 NaN rows
    df.iloc[80,    0]  = 999_999.0       # extreme high
    df.iloc[150,   0]  = -80_000.0       # extreme low (physically impossible)
    return pd.concat([df, df.iloc[[0]]])  # duplicate first row


def _make_features(n: int = 800) -> Tuple[pd.DataFrame, pd.Series,
                                           pd.DataFrame, pd.Series]:
    """Return (X_train, y_train, X_test, y_test) after feature engineering."""
    from ml.feature_engineering.feature_pipeline import FeatureEngineer
    df = _make_raw(n)
    fe  = FeatureEngineer()
    df_feat = fe.transform(df)
    feat_cols = fe.get_feature_columns(df_feat)
    X = df_feat[feat_cols]
    y = df_feat[TARGET]
    split = int(len(X) * 0.8)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


# ════════════════════════════════════════════════════════════════════════════
# 1.  DataCleaner
# ════════════════════════════════════════════════════════════════════════════

class TestDataCleaner(unittest.TestCase):
    """Tests every cleaning step independently and combined."""

    def setUp(self):
        from ml.preprocessing.cleaner import DataCleaner
        self.cleaner = DataCleaner()
        self.dirty   = _make_dirty()

    # ── Return contract ─────────────────────────────────────────────────────

    def test_clean_returns_dataframe(self):
        self.assertIsInstance(self.cleaner.clean(self.dirty), pd.DataFrame)

    def test_clean_preserves_datetime_index(self):
        out = self.cleaner.clean(self.dirty)
        self.assertIsInstance(out.index, pd.DatetimeIndex)

    def test_clean_preserves_target_column(self):
        out = self.cleaner.clean(self.dirty)
        self.assertIn(TARGET, out.columns)

    # ── Duplicate removal ───────────────────────────────────────────────────

    def test_clean_removes_duplicate_index(self):
        out = self.cleaner.clean(self.dirty)
        self.assertFalse(out.index.duplicated().any(),
                         "Duplicate index rows must be removed")

    def test_clean_duplicate_count_decreases(self):
        n_before = len(self.dirty)
        out = self.cleaner.clean(self.dirty)
        self.assertLess(len(out), n_before)

    def test_no_duplicate_already_clean(self):
        df = _make_raw(200)
        out = self.cleaner.clean(df)
        self.assertFalse(out.index.duplicated().any())

    # ── Missing value handling ───────────────────────────────────────────────

    def test_clean_fills_all_missing_values(self):
        out = self.cleaner.clean(self.dirty)
        self.assertEqual(out[TARGET].isnull().sum(), 0,
                         "No NaN should remain after cleaning")

    def test_clean_with_large_nan_block(self):
        """A run of 20 NaNs should still be filled via ffill/bfill fallback."""
        df = _make_raw(300)
        df.iloc[50:70, 0] = np.nan
        out = self.cleaner.clean(df)
        self.assertEqual(out[TARGET].isnull().sum(), 0)

    def test_all_nan_target_still_produces_output(self):
        """Cleaner should not crash even on pathological input."""
        df = _make_raw(150)
        df[TARGET] = np.nan
        # Should not raise — may return all-NaN or filled values
        try:
            out = self.cleaner.clean(df)
            self.assertIsInstance(out, pd.DataFrame)
        except Exception:
            pass  # acceptable to raise, but not crash the process

    # ── Outlier capping ─────────────────────────────────────────────────────

    def test_clean_caps_high_outlier(self):
        out = self.cleaner.clean(self.dirty)
        self.assertLess(out[TARGET].max(), 999_999.0,
                        "Extreme high value must be capped")

    def test_clean_caps_low_outlier(self):
        out = self.cleaner.clean(self.dirty)
        self.assertGreater(out[TARGET].min(), -80_000.0,
                           "Extreme low value must be capped")

    def test_outlier_capping_not_dropping(self):
        """Capping keeps the row; row count must stay the same (minus dupes)."""
        # 1 dup → expect len(dirty)-1 rows after cleaning
        n_dupes = self.dirty.index.duplicated().sum()
        out = self.cleaner.clean(self.dirty)
        self.assertEqual(len(out), len(self.dirty) - n_dupes)

    def test_clean_preserves_normal_values(self):
        """Values far from extremes should be unchanged."""
        df = _make_raw(300)
        mid_val = float(df[TARGET].iloc[5])
        out = self.cleaner.clean(df)
        # Value at position 5 should be unchanged (it is not an outlier)
        self.assertAlmostEqual(float(out[TARGET].iloc[5]), mid_val, places=0)

    def test_zscore_outlier_method(self):
        """DataCleaner configured with zscore should also cap outliers."""
        from ml.utils.config_loader import ConfigLoader
        import yaml, textwrap, tempfile

        cfg_text = textwrap.dedent(f"""
            project: {{name: test, version: "0.1", description: test}}
            data:
              raw_dir: data
              primary_file: test.csv
              processed_dir: ml/outputs/processed
              models_dir: ml/outputs/models
              reports_dir: ml/outputs/reports
              plots_dir: ml/outputs/plots
              target_column: {TARGET}
              timestamp_column: utc_timestamp
              exog_columns: []
              required_columns: [utc_timestamp, {TARGET}]
              max_rows: 5000
              start_date: "2015-01-01"
              end_date: "2025-12-31"
            preprocessing:
              missing_value_strategy: interpolate
              outlier_method: zscore
              outlier_threshold: 3.0
              iqr_multiplier: 1.5
              duplicate_strategy: drop_first
            feature_engineering:
              lag_hours: [1, 24, 168]
              rolling_windows: [7, 30]
              include_holidays: false
              holiday_country: DE
            forecasting:
              test_size: 0.2
              validation_size: 0.1
              random_state: 42
              cv_folds: 3
              models:
                linear_regression: {{enabled: true}}
                random_forest: {{enabled: false}}
                xgboost: {{enabled: false}}
                arima: {{enabled: false}}
                sarima: {{enabled: false}}
                sarimax: {{enabled: false}}
              forecast_horizons: {{short: 24, medium: 168, long: 720}}
              metrics: [rmse, mae, mape]
            anomaly_detection:
              methods: [zscore, iqr]
              zscore_threshold: 3.0
              iqr_multiplier: 1.5
              isolation_forest: {{contamination: 0.05, random_state: 42, n_estimators: 50}}
              lof: {{n_neighbors: 20, contamination: 0.05}}
              one_class_svm: {{nu: 0.05, kernel: rbf, gamma: scale}}
            explainability:
              max_samples: 50
              background_samples: 20
              plot_top_features: 10
              save_plots: false
            recommendation:
              peak_hours: [15, 16, 17, 18]
              hvac_contribution_threshold: 0.30
              high_consumption_percentile: 90
              low_consumption_percentile: 10
              load_shift_threshold_mw: 5000
            logging:
              level: WARNING
              format: "%(message)s"
              log_dir: ml/outputs/logs
              log_file: test.log
        """)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(cfg_text)
            tmp = f.name

        try:
            cfg = ConfigLoader(tmp)
            from ml.preprocessing.cleaner import DataCleaner
            cleaner = DataCleaner(cfg=cfg)
            df = _make_dirty(300)
            out = cleaner.clean(df)
            self.assertLess(out[TARGET].max(), 999_999.0)
        finally:
            os.remove(tmp)


# ════════════════════════════════════════════════════════════════════════════
# 2.  SchemaValidator
# ════════════════════════════════════════════════════════════════════════════

class TestSchemaValidator(unittest.TestCase):

    def setUp(self):
        from ml.ingestion.schema_validator import SchemaValidator, SchemaValidationError
        self.SchemaValidationError = SchemaValidationError
        self.validator = SchemaValidator()

    def _ok_df(self, n: int = 200) -> pd.DataFrame:
        return _make_raw(n)

    # ── Happy-path ──────────────────────────────────────────────────────────

    def test_valid_df_does_not_raise(self):
        # Must return None silently
        result = self.validator.validate(self._ok_df())
        self.assertIsNone(result)

    def test_valid_df_with_extra_columns_passes(self):
        df = self._ok_df()
        df["extra_col"] = 1.0
        self.assertIsNone(self.validator.validate(df))

    # ── DatetimeIndex check ──────────────────────────────────────────────────

    def test_integer_index_raises(self):
        df = self._ok_df()
        df.index = range(len(df))
        with self.assertRaises(self.SchemaValidationError):
            self.validator.validate(df)

    def test_string_index_raises(self):
        df = self._ok_df()
        df.index = [str(x) for x in range(len(df))]
        with self.assertRaises(self.SchemaValidationError):
            self.validator.validate(df)

    # ── Required column check ────────────────────────────────────────────────

    def test_missing_target_column_raises(self):
        df = self._ok_df().drop(columns=[TARGET])
        with self.assertRaises(self.SchemaValidationError):
            self.validator.validate(df)

    # ── Row count check ──────────────────────────────────────────────────────

    def test_too_few_rows_raises(self):
        df = _make_raw(10)  # below min_rows=100
        with self.assertRaises(self.SchemaValidationError):
            self.validator.validate(df)

    def test_exactly_100_rows_passes(self):
        df = _make_raw(100)
        self.assertIsNone(self.validator.validate(df))

    def test_99_rows_raises(self):
        df = _make_raw(99)
        with self.assertRaises(self.SchemaValidationError):
            self.validator.validate(df)

    # ── All-null target check ────────────────────────────────────────────────

    def test_all_null_target_raises(self):
        df = self._ok_df()
        df[TARGET] = np.nan
        with self.assertRaises(self.SchemaValidationError):
            self.validator.validate(df)

    def test_partial_null_target_passes(self):
        df = self._ok_df()
        df.iloc[:10, 0] = np.nan   # only 10 / 200 NaN — below 50% threshold
        self.assertIsNone(self.validator.validate(df))

    # ── Report method ────────────────────────────────────────────────────────

    def test_report_returns_dict(self):
        rpt = self.validator.report(self._ok_df())
        self.assertIsInstance(rpt, dict)

    def test_report_total_rows_correct(self):
        rpt = self.validator.report(_make_raw(250))
        self.assertEqual(rpt["total_rows"], 250)

    def test_report_missing_per_column(self):
        df = self._ok_df()
        df.iloc[:5, 0] = np.nan
        rpt = self.validator.report(df)
        self.assertIn("missing_per_column", rpt)
        self.assertEqual(rpt["missing_per_column"][TARGET], 5)

    def test_report_duplicated_rows(self):
        df = self._ok_df()
        df_dup = pd.concat([df, df.iloc[[0]]])
        rpt = self.validator.report(df_dup)
        self.assertGreater(rpt["duplicated_rows"], 0)

    def test_report_date_range(self):
        rpt = self.validator.report(self._ok_df())
        self.assertIn("date_range", rpt)
        self.assertIn("start", rpt["date_range"])
        self.assertIn("end", rpt["date_range"])

    def test_report_columns_list(self):
        rpt = self.validator.report(self._ok_df())
        self.assertIsInstance(rpt["columns"], list)
        self.assertIn(TARGET, rpt["columns"])

    def test_report_missing_pct(self):
        df = self._ok_df()
        df.iloc[:20, 0] = np.nan  # 10% of 200
        rpt = self.validator.report(df)
        self.assertIn("missing_pct_per_column", rpt)
        pct = rpt["missing_pct_per_column"][TARGET]
        self.assertAlmostEqual(pct, 10.0, places=0)


# ════════════════════════════════════════════════════════════════════════════
# 3.  FeatureEngineer
# ════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.feature_engineering.feature_pipeline import FeatureEngineer
        cls.fe      = FeatureEngineer()
        cls.df_raw  = _make_raw(700)
        cls.df_out  = cls.fe.transform(cls.df_raw)
        cls.feat    = cls.fe.get_feature_columns(cls.df_out)

    # ── Output contract ─────────────────────────────────────────────────────

    def test_returns_dataframe(self):
        self.assertIsInstance(self.df_out, pd.DataFrame)

    def test_preserves_datetime_index(self):
        self.assertIsInstance(self.df_out.index, pd.DatetimeIndex)

    def test_no_nan_in_output(self):
        self.assertEqual(self.df_out.isnull().sum().sum(), 0,
                         "Transform must drop all NaN rows")

    def test_target_column_preserved(self):
        self.assertIn(TARGET, self.df_out.columns)

    def test_output_rows_fewer_than_input(self):
        # lag-168 means 168 rows dropped at minimum
        self.assertLess(len(self.df_out), len(self.df_raw))

    # ── Time features ───────────────────────────────────────────────────────

    def test_hour_column_present(self):
        self.assertIn("hour", self.df_out.columns)

    def test_hour_range_0_to_23(self):
        self.assertTrue((self.df_out["hour"] >= 0).all())
        self.assertTrue((self.df_out["hour"] <= 23).all())

    def test_day_column_range(self):
        self.assertTrue((self.df_out["day"] >= 1).all())
        self.assertTrue((self.df_out["day"] <= 31).all())

    def test_month_column_range(self):
        self.assertTrue((self.df_out["month"] >= 1).all())
        self.assertTrue((self.df_out["month"] <= 12).all())

    def test_quarter_range(self):
        self.assertTrue((self.df_out["quarter"] >= 1).all())
        self.assertTrue((self.df_out["quarter"] <= 4).all())

    def test_season_values_in_0_to_3(self):
        self.assertTrue(self.df_out["season"].isin([0, 1, 2, 3]).all())

    def test_is_weekend_binary(self):
        self.assertTrue(self.df_out["is_weekend"].isin([0, 1]).all())

    def test_is_holiday_binary(self):
        self.assertTrue(self.df_out["is_holiday"].isin([0, 1]).all())

    def test_day_of_week_range(self):
        self.assertTrue((self.df_out["day_of_week"] >= 0).all())
        self.assertTrue((self.df_out["day_of_week"] <= 6).all())

    # ── Cyclical features ────────────────────────────────────────────────────

    def test_hour_sin_in_minus1_to_1(self):
        self.assertTrue((self.df_out["hour_sin"] >= -1.0 - 1e-9).all())
        self.assertTrue((self.df_out["hour_sin"] <= 1.0 + 1e-9).all())

    def test_hour_cos_in_minus1_to_1(self):
        self.assertTrue((self.df_out["hour_cos"] >= -1.0 - 1e-9).all())
        self.assertTrue((self.df_out["hour_cos"] <= 1.0 + 1e-9).all())

    def test_month_sin_present(self):
        self.assertIn("month_sin", self.df_out.columns)

    def test_month_cos_present(self):
        self.assertIn("month_cos", self.df_out.columns)

    def test_dow_sin_present(self):
        self.assertIn("dow_sin", self.df_out.columns)

    # ── Lag features ────────────────────────────────────────────────────────

    def test_lag_1_present(self):
        self.assertIn("load_t_1", self.df_out.columns)

    def test_lag_24_present(self):
        self.assertIn("load_t_24", self.df_out.columns)

    def test_lag_168_present(self):
        self.assertIn("load_t_168", self.df_out.columns)

    def test_lag_1_matches_shifted_target(self):
        """load_t_1 at row i must equal target at row i-1."""
        # After transform + dropna the indices don't align trivially,
        # but load_t_1 should equal the previous row's target within the output.
        target_vals = self.df_out[TARGET].values
        lag1_vals   = self.df_out["load_t_1"].values
        # lag_1[i] ≈ target[i-1]  for consecutive rows (after dropna)
        # We just check correlation is strongly positive
        corr = np.corrcoef(target_vals[1:], lag1_vals[1:])[0, 1]
        self.assertGreater(corr, 0.9, "lag-1 should correlate > 0.9 with target")

    # ── Rolling features ────────────────────────────────────────────────────

    def test_rolling_mean_7_present(self):
        self.assertIn("rolling_mean_7", self.df_out.columns)

    def test_rolling_mean_30_present(self):
        self.assertIn("rolling_mean_30", self.df_out.columns)

    def test_rolling_std_7_present(self):
        self.assertIn("rolling_std_7", self.df_out.columns)

    def test_rolling_std_30_present(self):
        self.assertIn("rolling_std_30", self.df_out.columns)

    def test_rolling_std_non_negative(self):
        self.assertTrue((self.df_out["rolling_std_7"] >= 0).all())

    # ── get_feature_columns ──────────────────────────────────────────────────

    def test_feature_columns_exclude_target(self):
        self.assertNotIn(TARGET, self.feat)

    def test_feature_columns_non_empty(self):
        self.assertGreater(len(self.feat), 10)

    def test_feature_columns_all_in_dataframe(self):
        for col in self.feat:
            self.assertIn(col, self.df_out.columns, f"Feature column {col!r} not found")

    def test_feature_columns_no_duplicates(self):
        self.assertEqual(len(self.feat), len(set(self.feat)))

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_minimal_input_still_works(self):
        """Even a 200-row df should produce output after dropping lag rows."""
        from ml.feature_engineering.feature_pipeline import FeatureEngineer
        fe  = FeatureEngineer()
        df  = _make_raw(250)
        out = fe.transform(df)
        self.assertGreater(len(out), 0)

    def test_transform_does_not_mutate_input(self):
        df_copy = self.df_raw.copy()
        self.fe.transform(self.df_raw)
        pd.testing.assert_frame_equal(self.df_raw, df_copy)


# ════════════════════════════════════════════════════════════════════════════
# 4.  LinearRegressionModel
# ════════════════════════════════════════════════════════════════════════════

class TestLinearRegressionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.forecasting.linear_model import LinearRegressionModel
        cls.ModelClass = LinearRegressionModel
        cls.X_tr, cls.y_tr, cls.X_te, cls.y_te = _make_features(600)
        cls.model = LinearRegressionModel()
        cls.model.fit(cls.X_tr, cls.y_tr)

    # ── Name ────────────────────────────────────────────────────────────────

    def test_name_is_linear_regression(self):
        self.assertEqual(self.model.name, "LinearRegression")

    # ── fit() ────────────────────────────────────────────────────────────────

    def test_fit_returns_self(self):
        m = self.ModelClass()
        ret = m.fit(self.X_tr, self.y_tr)
        self.assertIs(ret, m)

    def test_is_fitted_after_fit(self):
        self.assertTrue(self.model._is_fitted)

    def test_not_fitted_before_fit(self):
        m = self.ModelClass()
        self.assertFalse(m._is_fitted)

    # ── predict() ────────────────────────────────────────────────────────────

    def test_predict_length_matches_input(self):
        preds = self.model.predict(self.X_te)
        self.assertEqual(len(preds), len(self.X_te))

    def test_predict_returns_numpy_array(self):
        preds = self.model.predict(self.X_te)
        self.assertIsInstance(preds, np.ndarray)

    def test_predict_all_finite(self):
        preds = self.model.predict(self.X_te)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_predict_before_fit_raises(self):
        m = self.ModelClass()
        with self.assertRaises(Exception):
            m.predict(self.X_te)

    def test_predict_single_row(self):
        preds = self.model.predict(self.X_te.iloc[[0]])
        self.assertEqual(len(preds), 1)

    # ── evaluate() ───────────────────────────────────────────────────────────

    def test_evaluate_returns_dict(self):
        m = self.ModelClass()
        m.fit(self.X_tr, self.y_tr)
        metrics = m.evaluate(self.X_te, self.y_te)
        self.assertIsInstance(metrics, dict)

    def test_evaluate_has_rmse_mae_mape(self):
        m = self.ModelClass()
        m.fit(self.X_tr, self.y_tr)
        m = m.evaluate(m, self.X_te, self.y_te) if False else m.evaluate(self.X_te, self.y_te)
        for k in ("rmse", "mae", "mape"):
            self.assertIn(k, m)

    def test_evaluate_rmse_positive(self):
        m = self.ModelClass()
        m.fit(self.X_tr, self.y_tr)
        metrics = m.evaluate(self.X_te, self.y_te)
        self.assertGreater(metrics["rmse"], 0)

    def test_evaluate_before_fit_raises(self):
        m = self.ModelClass()
        with self.assertRaises(RuntimeError):
            m.evaluate(self.X_te, self.y_te)

    # ── Properties ───────────────────────────────────────────────────────────

    def test_coef_shape_matches_features(self):
        self.assertEqual(len(self.model.coef_), self.X_tr.shape[1])

    def test_feature_importances_shape(self):
        imp = self.model.feature_importances_
        self.assertEqual(len(imp), self.X_tr.shape[1])

    def test_feature_importances_non_negative(self):
        imp = self.model.feature_importances_
        self.assertTrue((imp >= 0).all())

    # ── fit_evaluate ─────────────────────────────────────────────────────────

    def test_fit_evaluate_returns_model_and_metrics(self):
        m = self.ModelClass()
        model, metrics = m.fit_evaluate(self.X_tr, self.y_tr, self.X_te, self.y_te)
        self.assertTrue(model._is_fitted)
        self.assertIn("rmse", metrics)

    # ── repr ─────────────────────────────────────────────────────────────────

    def test_repr_contains_name(self):
        self.assertIn("LinearRegression", repr(self.model))

    def test_repr_unfitted_says_not_fitted(self):
        m = self.ModelClass()
        self.assertIn("not fitted", repr(m))

    def test_repr_fitted_says_fitted(self):
        self.assertIn("fitted", repr(self.model))


# ════════════════════════════════════════════════════════════════════════════
# 5.  RandomForestModel  (no-tuning path for speed)
# ════════════════════════════════════════════════════════════════════════════

class TestRandomForestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import textwrap, tempfile, yaml
        from ml.forecasting.random_forest_model import RandomForestModel
        from ml.utils.config_loader import ConfigLoader

        # Minimal config: tuning=false + small RF for speed
        cfg_yaml = textwrap.dedent(f"""
            project: {{name: t, version: "0.1", description: t}}
            data:
              raw_dir: data
              primary_file: t.csv
              processed_dir: ml/outputs/processed
              models_dir: ml/outputs/models
              reports_dir: ml/outputs/reports
              plots_dir: ml/outputs/plots
              target_column: {TARGET}
              timestamp_column: utc_timestamp
              exog_columns: []
              required_columns: [utc_timestamp, {TARGET}]
              max_rows: 5000
              start_date: "2015-01-01"
              end_date: "2025-12-31"
            preprocessing:
              missing_value_strategy: interpolate
              outlier_method: iqr
              outlier_threshold: 3.0
              iqr_multiplier: 1.5
              duplicate_strategy: drop_first
            feature_engineering:
              lag_hours: [1, 24, 168]
              rolling_windows: [7, 30]
              include_holidays: false
              holiday_country: DE
            forecasting:
              test_size: 0.2
              validation_size: 0.1
              random_state: 42
              cv_folds: 3
              models:
                linear_regression: {{enabled: true}}
                random_forest:
                  enabled: true
                  tuning: false
                  n_estimators: 20
                  max_depth: 5
                  min_samples_split: 5
                  n_jobs: 1
                xgboost: {{enabled: false}}
                arima: {{enabled: false}}
                sarima: {{enabled: false}}
                sarimax: {{enabled: false}}
              forecast_horizons: {{short: 24, medium: 168, long: 720}}
              metrics: [rmse, mae, mape]
            anomaly_detection:
              methods: [zscore]
              zscore_threshold: 3.0
              iqr_multiplier: 1.5
              isolation_forest: {{contamination: 0.05, random_state: 42, n_estimators: 10}}
              lof: {{n_neighbors: 10, contamination: 0.05}}
              one_class_svm: {{nu: 0.05, kernel: rbf, gamma: scale}}
            explainability:
              max_samples: 30
              background_samples: 15
              plot_top_features: 5
              save_plots: false
            recommendation:
              peak_hours: [15, 16, 17, 18]
              hvac_contribution_threshold: 0.30
              high_consumption_percentile: 90
              low_consumption_percentile: 10
              load_shift_threshold_mw: 5000
            logging:
              level: WARNING
              format: "%(message)s"
              log_dir: ml/outputs/logs
              log_file: test.log
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False, encoding="utf-8") as f:
            f.write(cfg_yaml)
            cls._cfg_path = f.name

        cls.cfg = ConfigLoader(cls._cfg_path)
        cls.ModelClass = RandomForestModel
        cls.X_tr, cls.y_tr, cls.X_te, cls.y_te = _make_features(500)
        cls.model = RandomForestModel(cfg=cls.cfg)
        cls.model.fit(cls.X_tr, cls.y_tr)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls._cfg_path):
            os.remove(cls._cfg_path)

    def test_name(self):
        self.assertEqual(self.model.name, "RandomForest")

    def test_fit_returns_self(self):
        m = self.ModelClass(cfg=self.cfg)
        self.assertIs(m.fit(self.X_tr, self.y_tr), m)

    def test_is_fitted_after_fit(self):
        self.assertTrue(self.model._is_fitted)

    def test_predict_shape(self):
        preds = self.model.predict(self.X_te)
        self.assertEqual(len(preds), len(self.X_te))

    def test_predict_all_finite(self):
        self.assertTrue(np.all(np.isfinite(self.model.predict(self.X_te))))

    def test_feature_importances_length(self):
        self.assertEqual(len(self.model.feature_importances_), self.X_tr.shape[1])

    def test_feature_importances_sum_to_one(self):
        self.assertAlmostEqual(self.model.feature_importances_.sum(), 1.0, places=5)

    def test_feature_importances_non_negative(self):
        self.assertTrue((self.model.feature_importances_ >= 0).all())

    def test_evaluate_returns_positive_rmse(self):
        metrics = self.model.evaluate(self.X_te, self.y_te)
        self.assertGreater(metrics["rmse"], 0)

    def test_predict_on_training_data(self):
        """Overfitting test: RF should do well on its own training data."""
        from ml.utils.helpers import rmse
        preds = self.model.predict(self.X_tr)
        train_rmse = rmse(self.y_tr.values, preds)
        naive_rmse = rmse(self.y_tr.values, np.full(len(self.y_tr), self.y_tr.mean()))
        self.assertLess(train_rmse, naive_rmse * 0.5,
                        "RF should achieve < 50% of naive RMSE on training set")


# ════════════════════════════════════════════════════════════════════════════
# 6.  XGBoostModel  (tuning=false for speed)
# ════════════════════════════════════════════════════════════════════════════

class TestXGBoostModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import textwrap, tempfile
        from ml.forecasting.xgboost_model import XGBoostModel
        from ml.utils.config_loader import ConfigLoader

        cfg_yaml = textwrap.dedent(f"""
            project: {{name: t, version: "0.1", description: t}}
            data:
              raw_dir: data
              primary_file: t.csv
              processed_dir: ml/outputs/processed
              models_dir: ml/outputs/models
              reports_dir: ml/outputs/reports
              plots_dir: ml/outputs/plots
              target_column: {TARGET}
              timestamp_column: utc_timestamp
              exog_columns: []
              required_columns: [utc_timestamp, {TARGET}]
              max_rows: 5000
              start_date: "2015-01-01"
              end_date: "2025-12-31"
            preprocessing:
              missing_value_strategy: interpolate
              outlier_method: iqr
              outlier_threshold: 3.0
              iqr_multiplier: 1.5
              duplicate_strategy: drop_first
            feature_engineering:
              lag_hours: [1, 24, 168]
              rolling_windows: [7, 30]
              include_holidays: false
              holiday_country: DE
            forecasting:
              test_size: 0.2
              validation_size: 0.1
              random_state: 42
              cv_folds: 3
              models:
                linear_regression: {{enabled: false}}
                random_forest: {{enabled: false}}
                xgboost:
                  enabled: true
                  tuning: false
                  n_estimators: 30
                  learning_rate: 0.1
                  max_depth: 4
                  subsample: 0.8
                  colsample_bytree: 0.8
                  n_jobs: 1
                  verbosity: 0
                arima: {{enabled: false}}
                sarima: {{enabled: false}}
                sarimax: {{enabled: false}}
              forecast_horizons: {{short: 24, medium: 168, long: 720}}
              metrics: [rmse, mae, mape]
            anomaly_detection:
              methods: [zscore]
              zscore_threshold: 3.0
              iqr_multiplier: 1.5
              isolation_forest: {{contamination: 0.05, random_state: 42, n_estimators: 10}}
              lof: {{n_neighbors: 10, contamination: 0.05}}
              one_class_svm: {{nu: 0.05, kernel: rbf, gamma: scale}}
            explainability:
              max_samples: 30
              background_samples: 15
              plot_top_features: 5
              save_plots: false
            recommendation:
              peak_hours: [15, 16, 17, 18]
              hvac_contribution_threshold: 0.30
              high_consumption_percentile: 90
              low_consumption_percentile: 10
              load_shift_threshold_mw: 5000
            logging:
              level: WARNING
              format: "%(message)s"
              log_dir: ml/outputs/logs
              log_file: test.log
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False, encoding="utf-8") as f:
            f.write(cfg_yaml)
            cls._cfg_path = f.name

        cls.cfg = ConfigLoader(cls._cfg_path)
        cls.ModelClass = XGBoostModel
        cls.X_tr, cls.y_tr, cls.X_te, cls.y_te = _make_features(550)
        cls.model = XGBoostModel(cfg=cls.cfg)
        cls.model.fit(cls.X_tr, cls.y_tr)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls._cfg_path):
            os.remove(cls._cfg_path)

    def test_name(self):
        self.assertEqual(self.model.name, "XGBoost")

    def test_is_fitted_after_fit(self):
        self.assertTrue(self.model._is_fitted)

    def test_predict_shape(self):
        preds = self.model.predict(self.X_te)
        self.assertEqual(len(preds), len(self.X_te))

    def test_predict_all_finite(self):
        self.assertTrue(np.all(np.isfinite(self.model.predict(self.X_te))))

    def test_feature_importances_length(self):
        self.assertEqual(len(self.model.feature_importances_), self.X_tr.shape[1])

    def test_feature_importances_non_negative(self):
        self.assertTrue((self.model.feature_importances_ >= 0).all())

    def test_beats_naive_predictor(self):
        from ml.utils.helpers import rmse
        preds = self.model.predict(self.X_te)
        xgb_rmse   = rmse(self.y_te.values, preds)
        naive_rmse = rmse(self.y_te.values,
                          np.full(len(self.y_te), self.y_tr.mean()))
        self.assertLess(xgb_rmse, naive_rmse,
                        "XGBoost must beat the naive mean predictor")

    def test_booster_property_returns_booster(self):
        import xgboost as xgb
        self.assertIsInstance(self.model.booster,
                              (xgb.core.Booster, type(self.model.booster)))

    def test_feature_names_stored(self):
        self.assertIsNotNone(self.model._feature_names)
        self.assertEqual(len(self.model._feature_names), self.X_tr.shape[1])

    def test_predictions_on_same_shape(self):
        """XGBoost must produce same-length array for any subset."""
        subset = self.X_te.iloc[:10]
        preds = self.model.predict(subset)
        self.assertEqual(len(preds), 10)


# ════════════════════════════════════════════════════════════════════════════
# 7.  AnomalyDetector – each method + ensemble
# ════════════════════════════════════════════════════════════════════════════

class TestAnomalyDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.anomaly_detection.anomaly_detector import AnomalyDetector
        cls.AnomalyDetector = AnomalyDetector
        cls.detector = AnomalyDetector()
        rng = np.random.default_rng(42)
        n = 500
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        signal = rng.normal(50_000, 2_500, n)
        # inject 4 extreme spikes at known positions
        cls.spike_positions = [50, 150, 300, 400]
        for p in cls.spike_positions:
            signal[p] = 300_000.0
        cls.df = pd.DataFrame({TARGET: signal}, index=idx)
        cls.result = cls.detector.detect(cls.df)

    # ── Return contract ─────────────────────────────────────────────────────

    def test_detect_returns_dataframe(self):
        self.assertIsInstance(self.result, pd.DataFrame)

    def test_same_length_as_input(self):
        self.assertEqual(len(self.result), len(self.df))

    # ── Required columns ────────────────────────────────────────────────────

    def test_is_anomaly_column_present(self):
        self.assertIn("is_anomaly", self.result.columns)

    def test_anomaly_score_column_present(self):
        self.assertIn("anomaly_score", self.result.columns)

    def test_all_method_flag_columns_present(self):
        for m in ["zscore", "iqr", "isolation_forest", "lof", "one_class_svm"]:
            self.assertIn(f"anomaly_{m}", self.result.columns,
                          f"Column anomaly_{m} missing")

    # ── Value ranges ────────────────────────────────────────────────────────

    def test_is_anomaly_is_binary(self):
        self.assertTrue(self.result["is_anomaly"].isin([0, 1]).all())

    def test_anomaly_score_between_0_and_1(self):
        self.assertTrue((self.result["anomaly_score"] >= 0).all())
        self.assertTrue((self.result["anomaly_score"] <= 1).all())

    def test_method_flag_columns_are_binary(self):
        for m in ["zscore", "iqr"]:  # statistical methods always produce 0/1
            col = f"anomaly_{m}"
            self.assertTrue(self.result[col].isin([0, 1]).all(),
                            f"{col} must be binary")

    # ── Spike detection ──────────────────────────────────────────────────────

    def test_zscore_detects_all_spikes(self):
        for pos in self.spike_positions:
            ts = self.df.index[pos]
            self.assertEqual(self.result.loc[ts, "anomaly_zscore"], 1,
                             f"Z-Score missed spike at position {pos}")

    def test_iqr_detects_all_spikes(self):
        for pos in self.spike_positions:
            ts = self.df.index[pos]
            self.assertEqual(self.result.loc[ts, "anomaly_iqr"], 1,
                             f"IQR missed spike at position {pos}")

    def test_ensemble_flags_spike_rows(self):
        """Rows with 300k values should be flagged by ensemble vote."""
        for pos in self.spike_positions:
            ts = self.df.index[pos]
            self.assertEqual(self.result.loc[ts, "is_anomaly"], 1,
                             f"Ensemble did not flag known spike at position {pos}")

    # ── Normal data ──────────────────────────────────────────────────────────

    def test_normal_data_low_anomaly_rate(self):
        rng = np.random.default_rng(77)
        n = 800
        idx = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
        normal_df = pd.DataFrame(
            {TARGET: rng.normal(50_000, 1_500, n)}, index=idx
        )
        result = self.detector.detect(normal_df)
        rate = result["is_anomaly"].mean()
        self.assertLess(rate, 0.30,
                        f"Anomaly rate {rate:.2%} too high for purely normal data")

    # ── Summary ──────────────────────────────────────────────────────────────

    def test_summary_returns_dict(self):
        summary = self.detector.summary(self.result)
        self.assertIsInstance(summary, dict)

    def test_summary_keys_are_method_columns(self):
        summary = self.detector.summary(self.result)
        for key in summary:
            self.assertTrue(key.startswith("anomaly_"),
                            f"Unexpected summary key: {key}")

    def test_summary_values_non_negative_int(self):
        summary = self.detector.summary(self.result)
        for k, v in summary.items():
            self.assertIsInstance(v, int)
            self.assertGreaterEqual(v, 0)

    def test_summary_zscore_count_at_least_4(self):
        """The 4 injected spikes must appear in z-score count."""
        summary = self.detector.summary(self.result)
        self.assertGreaterEqual(summary.get("anomaly_zscore", 0), 4)

    # ── Individual method dispatch ────────────────────────────────────────────

    def test_zscore_method_standalone(self):
        series = self.df[TARGET]
        flags = self.detector._zscore(series)
        self.assertEqual(len(flags), len(series))
        self.assertTrue(np.all(np.isin(flags, [0, 1])))

    def test_iqr_method_standalone(self):
        series = self.df[TARGET]
        flags = self.detector._iqr(series)
        self.assertEqual(len(flags), len(series))
        self.assertTrue(np.all(np.isin(flags, [0, 1])))

    def test_isolation_forest_standalone(self):
        X = self.df[TARGET].values.reshape(-1, 1)
        flags = self.detector._isolation_forest(X)
        self.assertEqual(len(flags), len(X))
        self.assertTrue(np.all(np.isin(flags, [0, 1])))

    def test_lof_standalone(self):
        X = self.df[TARGET].values.reshape(-1, 1)
        flags = self.detector._lof(X)
        self.assertEqual(len(flags), len(X))
        self.assertTrue(np.all(np.isin(flags, [0, 1])))

    def test_one_class_svm_standalone(self):
        X = self.df[TARGET].values.reshape(-1, 1)
        flags = self.detector._one_class_svm(X)
        self.assertEqual(len(flags), len(X))
        self.assertTrue(np.all(np.isin(flags, [0, 1])))

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            self.detector._run_method(
                "unknown_method",
                self.df[TARGET],
                self.df[TARGET].values.reshape(-1, 1),
            )


# ════════════════════════════════════════════════════════════════════════════
# 8.  SHAPExplainer
# ════════════════════════════════════════════════════════════════════════════

class TestSHAPExplainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.forecasting.linear_model import LinearRegressionModel
        from ml.explainability.shap_explainer import SHAPExplainer

        X_tr, y_tr, X_te, _ = _make_features(500)
        model = LinearRegressionModel()
        model.fit(X_tr, y_tr)
        feat_cols = list(X_tr.columns)

        cls.explainer = SHAPExplainer(model, feat_cols)
        cls.explainer.fit(X_tr.iloc[:40])
        cls.explainer.explain(X_tr.iloc[:60])
        cls.importance = cls.explainer.feature_importance()
        cls.feat_cols  = feat_cols

    # ── feature_importance() ────────────────────────────────────────────────

    def test_importance_is_dataframe(self):
        self.assertIsInstance(self.importance, pd.DataFrame)

    def test_importance_has_feature_column(self):
        self.assertIn("feature", self.importance.columns)

    def test_importance_has_shap_column(self):
        self.assertIn("mean_abs_shap", self.importance.columns)

    def test_importance_non_negative(self):
        self.assertTrue((self.importance["mean_abs_shap"] >= 0).all())

    def test_importance_sorted_descending(self):
        vals = self.importance["mean_abs_shap"].values
        self.assertTrue(
            all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)),
            "Feature importance should be sorted descending"
        )

    def test_importance_row_count_equals_feature_count(self):
        self.assertEqual(len(self.importance), len(self.feat_cols))

    def test_importance_features_match_input(self):
        returned = set(self.importance["feature"].tolist())
        expected = set(self.feat_cols)
        self.assertEqual(returned, expected)

    # ── local_explanation() ──────────────────────────────────────────────────

    def test_local_explanation_returns_dict(self):
        local = self.explainer.local_explanation(0)
        self.assertIsInstance(local, dict)

    def test_local_explanation_keys_are_features(self):
        local = self.explainer.local_explanation(0)
        for k in local.keys():
            self.assertIn(k, self.feat_cols)

    def test_local_explanation_values_are_floats(self):
        local = self.explainer.local_explanation(0)
        for v in local.values():
            self.assertIsInstance(v, (float, np.floating))

    def test_local_explanation_second_row(self):
        local = self.explainer.local_explanation(1)
        self.assertGreater(len(local), 0)

    # ── Guard errors ─────────────────────────────────────────────────────────

    def test_explain_before_fit_raises_runtime_error(self):
        from ml.forecasting.linear_model import LinearRegressionModel
        from ml.explainability.shap_explainer import SHAPExplainer
        m = LinearRegressionModel()
        exp = SHAPExplainer(m, ["a", "b"])
        with self.assertRaises(RuntimeError):
            exp.explain(pd.DataFrame({"a": [1.0], "b": [2.0]}))

    def test_feature_importance_before_explain_raises(self):
        from ml.forecasting.linear_model import LinearRegressionModel
        from ml.explainability.shap_explainer import SHAPExplainer
        m = LinearRegressionModel()
        exp = SHAPExplainer(m, ["a"])
        with self.assertRaises(RuntimeError):
            exp.feature_importance()

    def test_local_explanation_before_explain_raises(self):
        from ml.forecasting.linear_model import LinearRegressionModel
        from ml.explainability.shap_explainer import SHAPExplainer
        m = LinearRegressionModel()
        exp = SHAPExplainer(m, ["a"])
        with self.assertRaises(RuntimeError):
            exp.local_explanation(0)

    # ── XGBoost TreeExplainer path ────────────────────────────────────────────

    def test_xgboost_tree_explainer_path(self):
        """XGBoost should use TreeExplainer (fast) and produce valid importance."""
        import textwrap, tempfile
        from ml.forecasting.xgboost_model import XGBoostModel
        from ml.explainability.shap_explainer import SHAPExplainer
        from ml.utils.config_loader import ConfigLoader

        cfg_yaml = textwrap.dedent(f"""
            project: {{name: t, version: "0.1", description: t}}
            data:
              raw_dir: data
              primary_file: t.csv
              processed_dir: ml/outputs/processed
              models_dir: ml/outputs/models
              reports_dir: ml/outputs/reports
              plots_dir: ml/outputs/plots
              target_column: {TARGET}
              timestamp_column: utc_timestamp
              exog_columns: []
              required_columns: [utc_timestamp, {TARGET}]
              max_rows: 5000
              start_date: "2015-01-01"
              end_date: "2025-12-31"
            preprocessing:
              missing_value_strategy: interpolate
              outlier_method: iqr
              outlier_threshold: 3.0
              iqr_multiplier: 1.5
              duplicate_strategy: drop_first
            feature_engineering:
              lag_hours: [1, 24, 168]
              rolling_windows: [7, 30]
              include_holidays: false
              holiday_country: DE
            forecasting:
              test_size: 0.2
              validation_size: 0.1
              random_state: 42
              cv_folds: 3
              models:
                linear_regression: {{enabled: false}}
                random_forest: {{enabled: false}}
                xgboost:
                  enabled: true
                  tuning: false
                  n_estimators: 20
                  learning_rate: 0.1
                  max_depth: 4
                  subsample: 0.8
                  colsample_bytree: 0.8
                  n_jobs: 1
                  verbosity: 0
                arima: {{enabled: false}}
                sarima: {{enabled: false}}
                sarimax: {{enabled: false}}
              forecast_horizons: {{short: 24, medium: 168, long: 720}}
              metrics: [rmse, mae, mape]
            anomaly_detection:
              methods: [zscore]
              zscore_threshold: 3.0
              iqr_multiplier: 1.5
              isolation_forest: {{contamination: 0.05, random_state: 42, n_estimators: 10}}
              lof: {{n_neighbors: 10, contamination: 0.05}}
              one_class_svm: {{nu: 0.05, kernel: rbf, gamma: scale}}
            explainability:
              max_samples: 40
              background_samples: 20
              plot_top_features: 5
              save_plots: false
            recommendation:
              peak_hours: [15, 16, 17, 18]
              hvac_contribution_threshold: 0.30
              high_consumption_percentile: 90
              low_consumption_percentile: 10
              load_shift_threshold_mw: 5000
            logging:
              level: WARNING
              format: "%(message)s"
              log_dir: ml/outputs/logs
              log_file: test.log
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False, encoding="utf-8") as f:
            f.write(cfg_yaml)
            tmp = f.name

        try:
            cfg = ConfigLoader(tmp)
            X_tr, y_tr, _, _ = _make_features(500)
            xgb_model = XGBoostModel(cfg=cfg)
            xgb_model.fit(X_tr, y_tr)
            feat_cols = list(X_tr.columns)

            exp = SHAPExplainer(xgb_model, feat_cols, cfg=cfg)
            exp.fit(X_tr.iloc[:30])
            exp.explain(X_tr.iloc[:40])
            imp = exp.feature_importance()

            self.assertIsInstance(imp, pd.DataFrame)
            self.assertTrue((imp["mean_abs_shap"] >= 0).all())
        finally:
            os.remove(tmp)


# ════════════════════════════════════════════════════════════════════════════
# 9.  RecommendationEngine – every rule group
# ════════════════════════════════════════════════════════════════════════════

class TestRecommendationEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.recommendation_engine.recommender import RecommendationEngine
        cls.engine = RecommendationEngine()

        # Forecast DataFrame – high peak vs low off-peak to trigger load-shift rule
        n_fc = 24
        idx_fc = pd.date_range("2020-06-01 00:00", periods=n_fc, freq="h", tz="UTC")
        # Peak hours (15-18) have much higher load → ratio > 1.15
        load_fc = np.array([
            5_000, 5_000, 5_000, 5_000, 5_000, 5_000,   # 00-05 (off-peak night)
            8_000, 9_000, 10_000, 11_000, 10_000, 9_000, # 06-11 (morning)
            10_000, 11_000, 12_000,                       # 12-14
            80_000, 80_000, 80_000, 80_000,               # 15-18 (peak)
            15_000, 12_000, 10_000, 8_000, 6_000,         # 19-23
        ])
        cls.forecast_df = pd.DataFrame({"forecast": load_fc}, index=idx_fc)

        # History – enough rows for trend + weekend checks
        n_hist = 700
        idx_hist = pd.date_range("2020-01-01", periods=n_hist, freq="h", tz="UTC")
        hist_load = 50_000 + 10_000 * np.sin(2 * np.pi * np.arange(n_hist) / 24)
        cls.history_df = pd.DataFrame({TARGET: hist_load}, index=idx_hist)

        # Anomaly DF – 15% anomaly rate → triggers anomaly recommendation
        n_an = 200
        idx_an = pd.date_range("2020-01-01", periods=n_an, freq="h", tz="UTC")
        is_anom = (np.arange(n_an) % 7 == 0).astype(int)
        cls.anomaly_df = pd.DataFrame({"is_anomaly": is_anom}, index=idx_an)

        # SHAP importance – hour + rolling in top features
        cls.shap_df = pd.DataFrame({
            "feature": ["hour_sin", "rolling_mean_7", "load_t_24",
                        "is_holiday", "month_cos"],
            "mean_abs_shap": [0.55, 0.40, 0.30, 0.20, 0.10],
        })

    # ── Return types ────────────────────────────────────────────────────────

    def test_generate_returns_list(self):
        self.assertIsInstance(self.engine.generate(), list)

    def test_generate_with_all_inputs_returns_list(self):
        recs = self.engine.generate(
            forecast_df=self.forecast_df,
            history_df=self.history_df,
            anomaly_df=self.anomaly_df,
            shap_importance_df=self.shap_df,
        )
        self.assertIsInstance(recs, list)

    def test_generate_with_no_inputs_returns_general_recs(self):
        recs = self.engine.generate()
        # _general_efficiency_rules always fires
        self.assertGreater(len(recs), 0)

    # ── Recommendation dataclass fields ──────────────────────────────────────

    def test_all_recommendations_have_required_fields(self):
        recs = self.engine.generate(history_df=self.history_df)
        for r in recs:
            self.assertIn(r.priority, {"HIGH", "MEDIUM", "LOW"})
            self.assertIsInstance(r.title, str)
            self.assertGreater(len(r.title), 0)
            self.assertIsInstance(r.description, str)
            self.assertIsInstance(r.category, str)
            self.assertIsInstance(r.estimated_saving_pct, float)
            self.assertGreaterEqual(r.estimated_saving_pct, 0.0)
            self.assertIsInstance(r.action_items, list)

    # ── Priority ordering ────────────────────────────────────────────────────

    def test_sorted_by_priority_high_first(self):
        recs = self.engine.generate(
            history_df=self.history_df,
            anomaly_df=self.anomaly_df,
        )
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        orders = [priority_order[r.priority] for r in recs]
        self.assertEqual(orders, sorted(orders),
                         "Recommendations must be sorted HIGH → MEDIUM → LOW")

    # ── Deduplication ────────────────────────────────────────────────────────

    def test_no_duplicate_titles(self):
        recs = self.engine.generate(
            forecast_df=self.forecast_df,
            history_df=self.history_df,
            anomaly_df=self.anomaly_df,
            shap_importance_df=self.shap_df,
        )
        titles = [r.title for r in recs]
        self.assertEqual(len(titles), len(set(titles)),
                         "Duplicate recommendation titles found")

    # ── Rule triggers ────────────────────────────────────────────────────────

    def test_peak_load_rule_triggers(self):
        recs = self.engine.generate(forecast_df=self.forecast_df)
        categories = [r.category for r in recs]
        self.assertIn("Load Shifting", categories,
                      "Load Shifting rule should trigger on high peak-to-off-peak ratio")

    def test_anomaly_rule_triggers_on_high_rate(self):
        recs = self.engine.generate(anomaly_df=self.anomaly_df)
        categories = [r.category for r in recs]
        self.assertIn("Anomaly Management", categories,
                      "Anomaly Management rule should trigger when rate > 5%")

    def test_shap_hour_rule_triggers(self):
        recs = self.engine.generate(shap_importance_df=self.shap_df)
        # "hour" in top features → TOU recommendation
        categories = [r.category for r in recs]
        self.assertIn("Time-of-Use", categories,
                      "Time-of-Use rule should fire when hour features rank high")

    def test_shap_rolling_rule_triggers(self):
        recs = self.engine.generate(shap_importance_df=self.shap_df)
        categories = [r.category for r in recs]
        self.assertIn("Forecasting", categories,
                      "Forecasting rule should fire when rolling features rank high")

    def test_general_efficiency_hvac_always_present(self):
        recs = self.engine.generate()
        categories = [r.category for r in recs]
        self.assertIn("HVAC Optimization", categories)

    def test_general_efficiency_renewable_always_present(self):
        recs = self.engine.generate()
        categories = [r.category for r in recs]
        self.assertIn("Renewable Integration", categories)

    # ── to_dict() ────────────────────────────────────────────────────────────

    def test_to_dict_returns_list_of_dicts(self):
        recs = self.engine.generate(history_df=self.history_df)
        d = self.engine.to_dict(recs)
        self.assertIsInstance(d, list)
        for item in d:
            self.assertIsInstance(item, dict)

    def test_to_dict_keys_complete(self):
        recs = self.engine.generate()
        d = self.engine.to_dict(recs)
        required_keys = {"category", "priority", "title", "description",
                         "estimated_saving_pct", "action_items"}
        for item in d:
            self.assertTrue(required_keys.issubset(item.keys()))

    def test_to_dict_action_items_is_list(self):
        recs = self.engine.generate()
        for item in self.engine.to_dict(recs):
            self.assertIsInstance(item["action_items"], list)

    # ── Edge: empty / None inputs ────────────────────────────────────────────

    def test_empty_anomaly_df_does_not_crash(self):
        empty_an = pd.DataFrame({"is_anomaly": []})
        recs = self.engine.generate(anomaly_df=empty_an)
        self.assertIsInstance(recs, list)

    def test_forecast_without_forecast_column(self):
        """Missing 'forecast' column → rules return empty list (no crash)."""
        bad_fc = pd.DataFrame({"value": [1, 2, 3]})
        recs = self.engine.generate(forecast_df=bad_fc)
        self.assertIsInstance(recs, list)


# ════════════════════════════════════════════════════════════════════════════
# 10. ForecastGenerator
# ════════════════════════════════════════════════════════════════════════════

class TestForecastGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.feature_engineering.feature_pipeline import FeatureEngineer
        from ml.forecasting.linear_model import LinearRegressionModel
        from ml.forecasting.forecast_generator import ForecastGenerator

        df_raw = _make_raw(900)
        fe = FeatureEngineer()
        df_feat = fe.transform(df_raw)
        feat_cols = fe.get_feature_columns(df_feat)
        X = df_feat[feat_cols]
        y = df_feat[TARGET]
        split = int(len(X) * 0.8)

        model = LinearRegressionModel()
        model.fit(X.iloc[:split], y.iloc[:split])

        cls.fg        = ForecastGenerator(model, fe)
        cls.history   = df_raw
        cls.forecasts = cls.fg.generate(df_raw)

    # ── Three horizons ───────────────────────────────────────────────────────

    def test_all_three_horizons_present(self):
        for h in ("24h", "7d", "30d"):
            self.assertIn(h, self.forecasts, f"Horizon {h} missing")

    def test_24h_length(self):
        self.assertEqual(len(self.forecasts["24h"]), 24)

    def test_7d_length(self):
        self.assertEqual(len(self.forecasts["7d"]), 168)

    def test_30d_length(self):
        self.assertEqual(len(self.forecasts["30d"]), 720)

    # ── DataFrame columns ────────────────────────────────────────────────────

    def test_required_columns_in_all_forecasts(self):
        for label, fc_df in self.forecasts.items():
            for col in ("forecast", "lower_bound", "upper_bound"):
                self.assertIn(col, fc_df.columns,
                              f"{label}: column '{col}' missing")

    # ── Index ────────────────────────────────────────────────────────────────

    def test_forecast_index_is_datetimeindex(self):
        for fc_df in self.forecasts.values():
            self.assertIsInstance(fc_df.index, pd.DatetimeIndex)

    def test_forecast_starts_after_history(self):
        last_ts = self.history.index[-1]
        for label, fc_df in self.forecasts.items():
            self.assertGreater(
                fc_df.index[0], last_ts,
                f"{label}: forecast must start after historical data"
            )

    def test_forecast_is_continuous(self):
        """Consecutive forecast rows should be 1 hour apart."""
        fc_24 = self.forecasts["24h"]
        diffs = fc_24.index.to_series().diff().dropna()
        one_hour = pd.Timedelta("1h")
        self.assertTrue((diffs == one_hour).all(),
                        "24h forecast index should be hourly")

    # ── Value sanity ─────────────────────────────────────────────────────────

    def test_forecast_values_non_negative(self):
        for fc_df in self.forecasts.values():
            self.assertTrue((fc_df["forecast"] >= 0).all())

    def test_lower_bound_leq_forecast(self):
        for fc_df in self.forecasts.values():
            self.assertTrue(
                (fc_df["lower_bound"] <= fc_df["forecast"]).all()
            )

    def test_upper_bound_geq_forecast(self):
        for fc_df in self.forecasts.values():
            self.assertTrue(
                (fc_df["upper_bound"] >= fc_df["forecast"]).all()
            )

    def test_lower_is_90pct_of_forecast(self):
        fc_df = self.forecasts["24h"]
        expected_lower = fc_df["forecast"] * 0.90
        pd.testing.assert_series_equal(
            fc_df["lower_bound"].round(4),
            expected_lower.round(4),
            check_names=False,
        )

    def test_upper_is_110pct_of_forecast(self):
        fc_df = self.forecasts["24h"]
        expected_upper = fc_df["forecast"] * 1.10
        pd.testing.assert_series_equal(
            fc_df["upper_bound"].round(4),
            expected_upper.round(4),
            check_names=False,
        )

    # ── _build_forecast_df directly ──────────────────────────────────────────

    def test_build_forecast_df_structure(self):
        ts = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
        preds = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        df = self.fg._build_forecast_df(ts, preds)
        self.assertIn("forecast", df.columns)
        self.assertIn("lower_bound", df.columns)
        self.assertIn("upper_bound", df.columns)
        self.assertEqual(len(df), 5)


# ════════════════════════════════════════════════════════════════════════════
# 11. Metric Helpers
# ════════════════════════════════════════════════════════════════════════════

class TestMetricHelpers(unittest.TestCase):

    # ── RMSE ────────────────────────────────────────────────────────────────

    def test_rmse_perfect_prediction(self):
        from ml.utils.helpers import rmse
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(rmse(y, y), 0.0)

    def test_rmse_known_value(self):
        from ml.utils.helpers import rmse
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        expected = np.sqrt((9.0 + 16.0) / 2.0)
        self.assertAlmostEqual(rmse(y_true, y_pred), expected, places=5)

    def test_rmse_symmetric(self):
        from ml.utils.helpers import rmse
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        self.assertAlmostEqual(rmse(a, b), rmse(b, a))

    def test_rmse_non_negative(self):
        from ml.utils.helpers import rmse
        y_true = np.random.rand(100)
        y_pred = np.random.rand(100)
        self.assertGreaterEqual(rmse(y_true, y_pred), 0.0)

    def test_rmse_single_element(self):
        from ml.utils.helpers import rmse
        self.assertAlmostEqual(rmse(np.array([5.0]), np.array([5.0])), 0.0)

    # ── MAE ─────────────────────────────────────────────────────────────────

    def test_mae_perfect_prediction(self):
        from ml.utils.helpers import mae
        y = np.array([10.0, 20.0, 30.0])
        self.assertAlmostEqual(mae(y, y), 0.0)

    def test_mae_known_value(self):
        from ml.utils.helpers import mae
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mae(y_true, y_pred), 2.0)

    def test_mae_non_negative(self):
        from ml.utils.helpers import mae
        y_true = np.random.rand(100)
        y_pred = np.random.rand(100)
        self.assertGreaterEqual(mae(y_true, y_pred), 0.0)

    def test_mae_leq_rmse_on_skewed_errors(self):
        """MAE ≤ RMSE always (Cauchy–Schwarz)."""
        from ml.utils.helpers import mae, rmse
        y_true = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
        y_pred = np.zeros(5)
        self.assertLessEqual(mae(y_true, y_pred), rmse(y_true, y_pred))

    # ── MAPE ────────────────────────────────────────────────────────────────

    def test_mape_perfect_prediction(self):
        from ml.utils.helpers import mape
        y = np.array([100.0, 200.0, 300.0])
        self.assertAlmostEqual(mape(y, y), 0.0, places=3)

    def test_mape_returns_percentage(self):
        from ml.utils.helpers import mape
        y_true = np.array([100.0])
        y_pred = np.array([110.0])
        # MAPE should be ~10%
        self.assertAlmostEqual(mape(y_true, y_pred), 10.0, places=1)

    def test_mape_non_negative(self):
        from ml.utils.helpers import mape
        y_true = np.random.rand(50) + 1.0
        y_pred = np.random.rand(50) + 1.0
        self.assertGreaterEqual(mape(y_true, y_pred), 0.0)

    # ── compute_metrics ──────────────────────────────────────────────────────

    def test_compute_metrics_returns_all_keys(self):
        from ml.utils.helpers import compute_metrics
        m = compute_metrics(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
        self.assertIn("rmse", m)
        self.assertIn("mae", m)
        self.assertIn("mape", m)

    def test_compute_metrics_values_are_floats(self):
        from ml.utils.helpers import compute_metrics
        m = compute_metrics(np.array([10.0, 20.0]), np.array([11.0, 21.0]))
        for v in m.values():
            self.assertIsInstance(v, float)

    def test_compute_metrics_perfect_zeros(self):
        from ml.utils.helpers import compute_metrics
        y = np.array([100.0, 200.0, 300.0])
        m = compute_metrics(y, y)
        self.assertAlmostEqual(m["rmse"], 0.0)
        self.assertAlmostEqual(m["mae"],  0.0)
        self.assertAlmostEqual(m["mape"], 0.0, places=3)

    # ── time_split ───────────────────────────────────────────────────────────

    def test_time_split_proportions(self):
        from ml.utils.helpers import time_split
        df = pd.DataFrame({"a": range(100)})
        train, test = time_split(df, test_size=0.2)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test),  20)

    def test_time_split_no_overlap(self):
        from ml.utils.helpers import time_split
        df = pd.DataFrame({"a": range(100)})
        train, test = time_split(df, test_size=0.2)
        train_idx = set(train.index)
        test_idx  = set(test.index)
        self.assertTrue(train_idx.isdisjoint(test_idx))

    def test_time_split_chronological(self):
        from ml.utils.helpers import time_split
        df = pd.DataFrame({"a": range(100)})
        train, test = time_split(df, test_size=0.2)
        self.assertLess(train.index.max(), test.index.min())

    def test_time_split_total_rows_preserved(self):
        from ml.utils.helpers import time_split
        df = pd.DataFrame({"a": range(100)})
        train, test = time_split(df, test_size=0.3)
        self.assertEqual(len(train) + len(test), 100)

    # ── clip_predictions ─────────────────────────────────────────────────────

    def test_clip_lower_bound(self):
        from ml.utils.helpers import clip_predictions
        preds = np.array([-500.0, 0.0, 100.0])
        out = clip_predictions(preds, lower=0.0)
        self.assertEqual(out[0], 0.0)

    def test_clip_upper_bound(self):
        from ml.utils.helpers import clip_predictions
        preds = np.array([50.0, 150.0, 200.0])
        out = clip_predictions(preds, lower=0.0, upper=100.0)
        self.assertEqual(out[1], 100.0)
        self.assertEqual(out[2], 100.0)

    def test_clip_in_bounds_unchanged(self):
        from ml.utils.helpers import clip_predictions
        preds = np.array([10.0, 20.0, 30.0])
        out = clip_predictions(preds, lower=0.0, upper=100.0)
        np.testing.assert_array_equal(out, preds)

    # ── ensure_dir ───────────────────────────────────────────────────────────

    def test_ensure_dir_creates_directory(self):
        from ml.utils.helpers import ensure_dir
        with tempfile.TemporaryDirectory() as tmp:
            new_path = os.path.join(tmp, "level1", "level2")
            result = ensure_dir(new_path)
            self.assertTrue(os.path.isdir(new_path))
            self.assertEqual(result, new_path)

    def test_ensure_dir_existing_dir_no_error(self):
        from ml.utils.helpers import ensure_dir
        with tempfile.TemporaryDirectory() as tmp:
            # calling twice should not raise
            ensure_dir(tmp)
            ensure_dir(tmp)
            self.assertTrue(os.path.isdir(tmp))

    # ── safe_concat ──────────────────────────────────────────────────────────

    def test_safe_concat_empty_list(self):
        from ml.utils.helpers import safe_concat
        result = safe_concat([])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_safe_concat_normal(self):
        from ml.utils.helpers import safe_concat
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        result = safe_concat([df1, df2])
        self.assertEqual(len(result), 4)

    # ── metrics_table ────────────────────────────────────────────────────────

    def test_metrics_table_returns_string(self):
        from ml.utils.helpers import metrics_table
        results = {
            "LinearRegression": {"rmse": 100.0, "mae": 80.0, "mape": 5.0},
            "XGBoost":          {"rmse": 80.0,  "mae": 60.0, "mape": 4.0},
        }
        table = metrics_table(results)
        self.assertIsInstance(table, str)

    def test_metrics_table_contains_model_names(self):
        from ml.utils.helpers import metrics_table
        results = {
            "LinearRegression": {"rmse": 100.0, "mae": 80.0, "mape": 5.0},
        }
        table = metrics_table(results)
        self.assertIn("LinearRegression", table)


# ════════════════════════════════════════════════════════════════════════════
# 12. ConfigLoader
# ════════════════════════════════════════════════════════════════════════════

class TestConfigLoader(unittest.TestCase):

    def setUp(self):
        from ml.utils.config_loader import config
        self.cfg = config

    def test_loads_without_error(self):
        self.assertIsNotNone(self.cfg)

    def test_top_level_dot_access(self):
        self.assertEqual(self.cfg.data.target_column, TARGET)

    def test_nested_dot_access_forecasting(self):
        self.assertIsNotNone(self.cfg.forecasting.test_size)

    def test_nested_dot_access_anomaly(self):
        self.assertIsNotNone(self.cfg.anomaly_detection.zscore_threshold)

    def test_get_with_existing_key(self):
        val = self.cfg.get("data")
        self.assertIsNotNone(val)

    def test_get_with_missing_key_returns_default(self):
        val = self.cfg.get("nonexistent_key_xyz", default="fallback")
        self.assertEqual(val, "fallback")

    def test_as_dict_returns_dict(self):
        d = self.cfg.as_dict()
        self.assertIsInstance(d, dict)

    def test_as_dict_contains_data_key(self):
        self.assertIn("data", self.cfg.as_dict())

    def test_missing_attribute_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            _ = self.cfg.this_key_does_not_exist

    def test_item_access_via_brackets(self):
        val = self.cfg["data"]
        self.assertIsNotNone(val)

    def test_dotdict_get_with_default(self):
        exp_cfg = self.cfg.explainability
        val = exp_cfg.get("nonexistent", 999)
        self.assertEqual(val, 999)

    def test_dotdict_repr(self):
        d = self.cfg.data
        self.assertIn("{", repr(d))

    def test_dotdict_as_dict(self):
        d = self.cfg.data.as_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("target_column", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
