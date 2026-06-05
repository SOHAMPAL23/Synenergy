"""
EnerVision AI - Integration Tests: Full Pipeline
Tests every stage end-to-end with synthetic data, no disk I/O required.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ─── Synthetic data helpers ───────────────────────────────────────────────────

def _make_energy_df(n: int = 800, freq: str = "h") -> pd.DataFrame:
    """Return a realistic energy DataFrame with target + exog columns."""
    idx = pd.date_range("2020-01-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)
    load = (
        50_000
        + 10_000 * np.sin(2 * np.pi * np.arange(n) / 24)
        + 5_000 * np.sin(2 * np.pi * np.arange(n) / (24 * 7))
        + rng.normal(0, 1_500, n)
    )
    solar = np.clip(rng.normal(5_000, 3_000, n), 0, None)
    wind  = np.clip(rng.normal(20_000, 8_000, n), 0, None)
    return pd.DataFrame(
        {
            "DE_load_actual_entsoe_transparency": load,
            "DE_solar_generation_actual": solar,
            "DE_wind_generation_actual": wind,
        },
        index=idx,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1-2: Data Ingestion & Schema Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaValidatorFull(unittest.TestCase):

    def setUp(self):
        from ml.ingestion.schema_validator import SchemaValidator, SchemaValidationError
        self.SchemaValidationError = SchemaValidationError
        self.validator = SchemaValidator()
        self.df = _make_energy_df(300)

    def test_valid_dataframe_passes_silently(self):
        result = self.validator.validate(self.df)
        self.assertIsNone(result)

    def test_report_contains_expected_keys(self):
        report = self.validator.report(self.df)
        for key in ("total_rows", "missing_per_column"):
            self.assertIn(key, report)

    def test_empty_dataframe_raises(self):
        with self.assertRaises((self.SchemaValidationError, Exception)):
            self.validator.validate(pd.DataFrame())

    def test_integer_index_raises(self):
        df = self.df.copy()
        df.index = range(len(df))
        with self.assertRaises(self.SchemaValidationError):
            self.validator.validate(df)

    def test_missing_target_column_raises(self):
        df = self.df.drop(columns=["DE_load_actual_entsoe_transparency"])
        with self.assertRaises(self.SchemaValidationError):
            self.validator.validate(df)

    def test_all_nan_target_raises(self):
        df = self.df.copy()
        df["DE_load_actual_entsoe_transparency"] = np.nan
        with self.assertRaises(self.SchemaValidationError):
            self.validator.validate(df)

    def test_five_row_df_raises(self):
        with self.assertRaises(self.SchemaValidationError):
            self.validator.validate(_make_energy_df(5))

    def test_report_total_rows(self):
        df = _make_energy_df(250)
        report = self.validator.report(df)
        self.assertEqual(report["total_rows"], 250)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3: Preprocessing / Cleaning
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataCleanerFull(unittest.TestCase):

    def setUp(self):
        from ml.preprocessing.cleaner import DataCleaner
        self.cleaner = DataCleaner()

    def _make_dirty(self, n: int = 400) -> pd.DataFrame:
        df = _make_energy_df(n)
        df.iloc[20:25, 0] = np.nan          # introduce NaNs
        df.iloc[100, 0] = 999_999.0         # extreme high spike
        df.iloc[200, 0] = -50_000.0         # extreme low spike
        return pd.concat([df, df.iloc[[0]]])  # duplicate row

    def test_clean_removes_duplicates(self):
        df_clean = self.cleaner.clean(self._make_dirty())
        self.assertFalse(df_clean.index.duplicated().any())

    def test_clean_fills_missing_values(self):
        df_clean = self.cleaner.clean(self._make_dirty())
        target = "DE_load_actual_entsoe_transparency"
        self.assertEqual(df_clean[target].isnull().sum(), 0)

    def test_clean_caps_high_outlier(self):
        df_clean = self.cleaner.clean(self._make_dirty())
        target = "DE_load_actual_entsoe_transparency"
        self.assertLess(df_clean[target].max(), 999_999.0)

    def test_clean_caps_low_outlier(self):
        df_clean = self.cleaner.clean(self._make_dirty())
        target = "DE_load_actual_entsoe_transparency"
        self.assertGreater(df_clean[target].min(), -50_000.0)

    def test_clean_preserves_datetimeindex(self):
        df_clean = self.cleaner.clean(self._make_dirty())
        self.assertIsInstance(df_clean.index, pd.DatetimeIndex)

    def test_clean_returns_dataframe(self):
        result = self.cleaner.clean(self._make_dirty())
        self.assertIsInstance(result, pd.DataFrame)

    def test_clean_already_clean_data(self):
        """Cleaning already-clean data should not crash and return similar length."""
        df = _make_energy_df(200)
        df_clean = self.cleaner.clean(df)
        # Should still have roughly the same rows
        self.assertGreaterEqual(len(df_clean), len(df) - 5)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4: Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineerFull(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.feature_engineering.feature_pipeline import FeatureEngineer
        cls.fe = FeatureEngineer()
        cls.df_raw = _make_energy_df(700)
        cls.df_feat = cls.fe.transform(cls.df_raw)
        cls.feat_cols = cls.fe.get_feature_columns(cls.df_feat)

    def test_transform_returns_dataframe(self):
        self.assertIsInstance(self.df_feat, pd.DataFrame)

    def test_no_nulls_in_output(self):
        self.assertEqual(self.df_feat.isnull().sum().sum(), 0)

    def test_target_column_preserved(self):
        self.assertIn("DE_load_actual_entsoe_transparency", self.df_feat.columns)

    def test_time_features_present(self):
        for col in ["hour", "day", "month", "week", "quarter", "season",
                    "is_weekend", "is_holiday"]:
            self.assertIn(col, self.df_feat.columns, f"Missing: {col}")

    def test_cyclical_features_present(self):
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            self.assertIn(col, self.df_feat.columns)

    def test_lag_features_present(self):
        for lag in [1, 24, 168]:
            self.assertIn(f"load_t_{lag}", self.df_feat.columns)

    def test_rolling_features_present(self):
        for w in [7, 30]:
            self.assertIn(f"rolling_mean_{w}", self.df_feat.columns)
            self.assertIn(f"rolling_std_{w}", self.df_feat.columns)

    def test_hour_values_in_range(self):
        self.assertTrue((self.df_feat["hour"] >= 0).all())
        self.assertTrue((self.df_feat["hour"] <= 23).all())

    def test_season_values_in_range(self):
        self.assertTrue(self.df_feat["season"].isin([0, 1, 2, 3]).all())

    def test_is_weekend_binary(self):
        self.assertTrue(self.df_feat["is_weekend"].isin([0, 1]).all())

    def test_feature_columns_exclude_target(self):
        self.assertNotIn("DE_load_actual_entsoe_transparency", self.feat_cols)

    def test_feature_columns_non_empty(self):
        self.assertGreater(len(self.feat_cols), 10)

    def test_row_count_reduced_due_to_lags(self):
        # 168-hour max lag → at least 168 rows dropped
        self.assertLess(len(self.df_feat), len(self.df_raw))

    def test_cyclical_hour_sin_range(self):
        self.assertTrue((self.df_feat["hour_sin"] >= -1.0).all())
        self.assertTrue((self.df_feat["hour_sin"] <= 1.0).all())


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5: Model Training (ModelSelector)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_train_test_split(n: int = 700):
    """Return (X_train, y_train, X_test, y_test) from engineered features."""
    from ml.feature_engineering.feature_pipeline import FeatureEngineer
    df_raw = _make_energy_df(n)
    fe = FeatureEngineer()
    df = fe.transform(df_raw)
    feat_cols = fe.get_feature_columns(df)
    X = df[feat_cols]
    y = df["DE_load_actual_entsoe_transparency"]
    split = int(len(X) * 0.8)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


class TestModelSelectorML(unittest.TestCase):
    """Tests ModelSelector with only ML models enabled (fast)."""

    @classmethod
    def setUpClass(cls):
        from ml.forecasting.model_selector import ModelSelector
        from ml.utils.config_loader import ConfigLoader
        import yaml, textwrap

        # Build a minimal config that enables only fast ML models
        minimal_yaml = textwrap.dedent("""
            project:
              name: test
              version: "0.1"
              description: test
            data:
              raw_dir: data
              primary_file: test.csv
              processed_dir: ml/outputs/processed
              models_dir: ml/outputs/models
              reports_dir: ml/outputs/reports
              plots_dir: ml/outputs/plots
              target_column: DE_load_actual_entsoe_transparency
              timestamp_column: utc_timestamp
              exog_columns: []
              required_columns:
                - utc_timestamp
                - DE_load_actual_entsoe_transparency
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
                linear_regression:
                  enabled: true
                random_forest:
                  enabled: true
                  n_estimators: 20
                  max_depth: 5
                  min_samples_split: 5
                  n_jobs: 1
                xgboost:
                  enabled: true
                  n_estimators: 30
                  learning_rate: 0.1
                  max_depth: 4
                  subsample: 0.8
                  colsample_bytree: 0.8
                  n_jobs: 1
                  verbosity: 0
                  tuning: false
                arima:
                  enabled: false
                sarima:
                  enabled: false
                sarimax:
                  enabled: false
              forecast_horizons:
                short: 24
                medium: 168
                long: 720
              metrics: ["rmse", "mae", "mape"]
            anomaly_detection:
              methods: ["zscore", "iqr"]
              zscore_threshold: 3.0
              iqr_multiplier: 1.5
              isolation_forest:
                contamination: 0.05
                random_state: 42
                n_estimators: 50
              lof:
                n_neighbors: 20
                contamination: 0.05
              one_class_svm:
                nu: 0.05
                kernel: rbf
                gamma: scale
            explainability:
              max_samples: 50
              background_samples: 30
              plot_top_features: 10
              save_plots: false
            recommendation:
              peak_hours: [15, 16, 17, 18]
              hvac_contribution_threshold: 0.30
              high_consumption_percentile: 90
              low_consumption_percentile: 10
              load_shift_threshold_mw: 5000
            logging:
              level: INFO
              format: "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
              log_dir: ml/outputs/logs
              log_file: pipeline.log
        """)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(minimal_yaml)
            cls._cfg_path = f.name

        cls.cfg = ConfigLoader(cls._cfg_path)
        cls.X_tr, cls.y_tr, cls.X_te, cls.y_te = _make_train_test_split(600)
        selector = ModelSelector(cfg=cls.cfg)
        cls.best_model, cls.results = selector.run(cls.X_tr, cls.y_tr, cls.X_te, cls.y_te)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls._cfg_path):
            os.remove(cls._cfg_path)

    def test_returns_non_empty_results(self):
        self.assertGreater(len(self.results), 0)

    def test_best_model_is_fitted(self):
        self.assertTrue(self.best_model._is_fitted)

    def test_best_model_has_name(self):
        self.assertIn(self.best_model.name,
                      ["LinearRegression", "RandomForest", "XGBoost"])

    def test_results_contain_rmse(self):
        for model_name, metrics in self.results.items():
            self.assertIn("rmse", metrics)
            self.assertIn("mae", metrics)
            self.assertIn("mape", metrics)

    def test_best_model_rmse_lowest(self):
        best_rmse = self.results[self.best_model.name]["rmse"]
        for m in self.results.values():
            self.assertLessEqual(best_rmse, m["rmse"] + 1e-6)

    def test_best_model_can_predict(self):
        preds = self.best_model.predict(self.X_te)
        self.assertEqual(len(preds), len(self.X_te))

    def test_predictions_all_finite(self):
        preds = self.best_model.predict(self.X_te)
        self.assertTrue(np.all(np.isfinite(preds)))


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 6: Forecast Generator
# ═══════════════════════════════════════════════════════════════════════════════

class TestForecastGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.feature_engineering.feature_pipeline import FeatureEngineer
        from ml.forecasting.linear_model import LinearRegressionModel
        from ml.forecasting.forecast_generator import ForecastGenerator

        df_raw = _make_energy_df(800)
        cls.fe = FeatureEngineer()
        df = cls.fe.transform(df_raw)
        feat_cols = cls.fe.get_feature_columns(df)
        X = df[feat_cols]
        y = df["DE_load_actual_entsoe_transparency"]
        split = int(len(X) * 0.8)

        model = LinearRegressionModel()
        model.fit(X.iloc[:split], y.iloc[:split])

        cls.fg = ForecastGenerator(model, cls.fe)
        cls.history_df = df_raw
        cls.forecasts = cls.fg.generate(df_raw)

    def test_returns_dict_with_three_horizons(self):
        self.assertIn("24h", self.forecasts)
        self.assertIn("7d", self.forecasts)
        self.assertIn("30d", self.forecasts)

    def test_forecast_df_has_required_columns(self):
        for label, fc_df in self.forecasts.items():
            for col in ["forecast", "lower_bound", "upper_bound"]:
                self.assertIn(col, fc_df.columns, f"{label} missing '{col}'")

    def test_forecast_lengths(self):
        self.assertEqual(len(self.forecasts["24h"]), 24)
        self.assertEqual(len(self.forecasts["7d"]), 168)

    def test_forecast_index_is_datetimeindex(self):
        for fc_df in self.forecasts.values():
            self.assertIsInstance(fc_df.index, pd.DatetimeIndex)

    def test_lower_bound_below_forecast(self):
        for fc_df in self.forecasts.values():
            self.assertTrue((fc_df["lower_bound"] <= fc_df["forecast"]).all())

    def test_upper_bound_above_forecast(self):
        for fc_df in self.forecasts.values():
            self.assertTrue((fc_df["upper_bound"] >= fc_df["forecast"]).all())

    def test_forecast_values_positive(self):
        for fc_df in self.forecasts.values():
            self.assertTrue((fc_df["forecast"] >= 0).all())

    def test_forecast_after_history_end(self):
        last_ts = self.history_df.index[-1]
        for label, fc_df in self.forecasts.items():
            self.assertGreater(
                fc_df.index[0], last_ts,
                f"{label} forecast starts before or at last historical point"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 7: Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnomalyDetectorFull(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.anomaly_detection.anomaly_detector import AnomalyDetector
        cls.detector = AnomalyDetector()
        rng = np.random.default_rng(0)
        n = 600
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        signal = rng.normal(50_000, 3_000, n)
        cls.known_spike_positions = [50, 200, 400]
        for p in cls.known_spike_positions:
            signal[p] = 250_000.0
        cls.df = pd.DataFrame({"DE_load_actual_entsoe_transparency": signal}, index=idx)
        cls.result = cls.detector.detect(cls.df)

    def test_returns_dataframe(self):
        self.assertIsInstance(self.result, pd.DataFrame)

    def test_same_length_as_input(self):
        self.assertEqual(len(self.result), len(self.df))

    def test_is_anomaly_column_present(self):
        self.assertIn("is_anomaly", self.result.columns)

    def test_anomaly_score_column_present(self):
        self.assertIn("anomaly_score", self.result.columns)

    def test_all_method_flag_columns_present(self):
        for method in ["zscore", "iqr", "isolation_forest", "lof", "one_class_svm"]:
            self.assertIn(f"anomaly_{method}", self.result.columns)

    def test_is_anomaly_is_binary(self):
        self.assertTrue(self.result["is_anomaly"].isin([0, 1]).all())

    def test_anomaly_score_in_range(self):
        self.assertTrue((self.result["anomaly_score"] >= 0).all())
        self.assertTrue((self.result["anomaly_score"] <= 1).all())

    def test_zscore_detects_spikes(self):
        """Inserted 250k spikes must be detected by z-score."""
        for pos in self.known_spike_positions:
            ts = self.df.index[pos]
            self.assertEqual(
                self.result.loc[ts, "anomaly_zscore"], 1,
                f"Z-Score missed spike at position {pos}"
            )

    def test_summary_returns_dict(self):
        summary = self.detector.summary(self.result)
        self.assertIsInstance(summary, dict)
        self.assertGreater(len(summary), 0)

    def test_summary_counts_non_negative(self):
        summary = self.detector.summary(self.result)
        for k, v in summary.items():
            self.assertGreaterEqual(v, 0)

    def test_normal_data_low_anomaly_rate(self):
        rng = np.random.default_rng(99)
        n = 1_000
        idx = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
        normal_df = pd.DataFrame(
            {"DE_load_actual_entsoe_transparency": rng.normal(50_000, 2_000, n)},
            index=idx,
        )
        result = self.detector.detect(normal_df)
        rate = result["is_anomaly"].mean()
        self.assertLess(rate, 0.3, f"Anomaly rate {rate:.2%} too high for normal data")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 8: SHAP Explainability
# ═══════════════════════════════════════════════════════════════════════════════

class TestSHAPExplainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.feature_engineering.feature_pipeline import FeatureEngineer
        from ml.forecasting.linear_model import LinearRegressionModel
        from ml.explainability.shap_explainer import SHAPExplainer

        df_raw = _make_energy_df(600)
        fe = FeatureEngineer()
        df = fe.transform(df_raw)
        feat_cols = fe.get_feature_columns(df)
        X = df[feat_cols]
        y = df["DE_load_actual_entsoe_transparency"]

        model = LinearRegressionModel()
        model.fit(X, y)

        cls.explainer = SHAPExplainer(model, feat_cols)
        cls.explainer.fit(X.iloc[:50])
        cls.explainer.explain(X.iloc[:80])
        cls.importance_df = cls.explainer.feature_importance()

    def test_feature_importance_is_dataframe(self):
        self.assertIsInstance(self.importance_df, pd.DataFrame)

    def test_feature_importance_has_columns(self):
        self.assertIn("feature", self.importance_df.columns)
        self.assertIn("mean_abs_shap", self.importance_df.columns)

    def test_feature_importance_non_negative(self):
        self.assertTrue((self.importance_df["mean_abs_shap"] >= 0).all())

    def test_feature_importance_sorted_descending(self):
        vals = self.importance_df["mean_abs_shap"].values
        self.assertTrue(all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)))

    def test_local_explanation_returns_dict(self):
        local = self.explainer.local_explanation(0)
        self.assertIsInstance(local, dict)
        self.assertGreater(len(local), 0)

    def test_local_explanation_keys_are_feature_names(self):
        local = self.explainer.local_explanation(0)
        from ml.feature_engineering.feature_pipeline import FeatureEngineer
        # All keys should be strings
        for k in local.keys():
            self.assertIsInstance(k, str)

    def test_explain_before_fit_raises(self):
        from ml.explainability.shap_explainer import SHAPExplainer
        from ml.forecasting.linear_model import LinearRegressionModel
        model = LinearRegressionModel()
        exp = SHAPExplainer(model, ["feat_a", "feat_b"])
        with self.assertRaises(RuntimeError):
            exp.explain(pd.DataFrame({"feat_a": [1.0], "feat_b": [2.0]}))

    def test_feature_importance_before_explain_raises(self):
        from ml.explainability.shap_explainer import SHAPExplainer
        from ml.forecasting.linear_model import LinearRegressionModel
        model = LinearRegressionModel()
        exp = SHAPExplainer(model, ["feat_a"])
        with self.assertRaises(RuntimeError):
            exp.feature_importance()


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 9: Recommendation Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecommendationEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.recommendation_engine.recommender import RecommendationEngine
        cls.engine = RecommendationEngine()

        # Synthetic forecast DataFrame
        idx = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
        rng = np.random.default_rng(7)
        cls.forecast_df = pd.DataFrame(
            {"forecast": rng.uniform(5_000, 80_000, 24)},
            index=idx,
        )

        # History with weekend variation
        hist_idx = pd.date_range("2020-01-01", periods=700, freq="h", tz="UTC")
        load = 50_000 + 10_000 * np.sin(2 * np.pi * np.arange(700) / 24)
        cls.history_df = pd.DataFrame(
            {"DE_load_actual_entsoe_transparency": load},
            index=hist_idx,
        )

        # Anomaly DF with ~10% anomalies
        n = 200
        an_idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        is_anomaly = (np.arange(n) % 10 == 0).astype(int)
        cls.anomaly_df = pd.DataFrame({"is_anomaly": is_anomaly}, index=an_idx)

        # SHAP importance
        cls.shap_df = pd.DataFrame({
            "feature": ["hour", "rolling_mean_7", "load_t_24", "is_holiday", "month"],
            "mean_abs_shap": [0.5, 0.4, 0.3, 0.2, 0.1],
        })

    def test_generate_returns_list(self):
        recs = self.engine.generate()
        self.assertIsInstance(recs, list)

    def test_generate_with_all_inputs_returns_recs(self):
        recs = self.engine.generate(
            forecast_df=self.forecast_df,
            history_df=self.history_df,
            anomaly_df=self.anomaly_df,
            shap_importance_df=self.shap_df,
        )
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)

    def test_recommendations_have_required_fields(self):
        recs = self.engine.generate(history_df=self.history_df)
        for r in recs:
            self.assertIn(r.priority, {"HIGH", "MEDIUM", "LOW"})
            self.assertIsInstance(r.title, str)
            self.assertIsInstance(r.description, str)
            self.assertIsInstance(r.category, str)
            self.assertIsInstance(r.estimated_saving_pct, float)
            self.assertIsInstance(r.action_items, list)

    def test_saving_pct_non_negative(self):
        recs = self.engine.generate(history_df=self.history_df)
        for r in recs:
            self.assertGreaterEqual(r.estimated_saving_pct, 0.0)

    def test_sorted_by_priority(self):
        recs = self.engine.generate(history_df=self.history_df)
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        orders = [priority_order[r.priority] for r in recs]
        self.assertEqual(orders, sorted(orders))

    def test_no_duplicate_titles(self):
        recs = self.engine.generate(
            history_df=self.history_df,
            anomaly_df=self.anomaly_df,
            shap_importance_df=self.shap_df,
        )
        titles = [r.title for r in recs]
        self.assertEqual(len(titles), len(set(titles)))

    def test_to_dict_returns_list_of_dicts(self):
        recs = self.engine.generate(history_df=self.history_df)
        d = self.engine.to_dict(recs)
        self.assertIsInstance(d, list)
        for item in d:
            self.assertIsInstance(item, dict)
            for key in ("category", "priority", "title", "description",
                        "estimated_saving_pct", "action_items"):
                self.assertIn(key, item)

    def test_anomaly_rules_triggered_when_high_anomaly_rate(self):
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        # 15% anomaly rate → should trigger anomaly recommendation
        is_anomaly = (np.arange(n) % 7 == 0).astype(int)
        an_df = pd.DataFrame({"is_anomaly": is_anomaly}, index=idx)
        recs = self.engine.generate(anomaly_df=an_df)
        titles = [r.title for r in recs]
        anomaly_recs = [t for t in titles if "Anomal" in t]
        self.assertGreater(len(anomaly_recs), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelperUtilities(unittest.TestCase):

    def test_rmse_perfect(self):
        from ml.utils.helpers import rmse
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(rmse(y, y), 0.0)

    def test_mae_perfect(self):
        from ml.utils.helpers import mae
        y = np.array([10.0, 20.0, 30.0])
        self.assertAlmostEqual(mae(y, y), 0.0)

    def test_mape_perfect(self):
        from ml.utils.helpers import mape
        y = np.array([100.0, 200.0, 300.0])
        self.assertAlmostEqual(mape(y, y), 0.0, places=3)

    def test_rmse_known_value(self):
        from ml.utils.helpers import rmse
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        expected = np.sqrt((9 + 16) / 2)
        self.assertAlmostEqual(rmse(y_true, y_pred), expected, places=5)

    def test_compute_metrics_all_keys(self):
        from ml.utils.helpers import compute_metrics
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        m = compute_metrics(y_true, y_pred)
        self.assertIn("rmse", m)
        self.assertIn("mae", m)
        self.assertIn("mape", m)
        self.assertGreater(m["rmse"], 0)

    def test_time_split_sizes(self):
        from ml.utils.helpers import time_split
        df = pd.DataFrame({"a": range(100)})
        train, test = time_split(df, test_size=0.2)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)

    def test_time_split_chronological_order(self):
        from ml.utils.helpers import time_split
        df = pd.DataFrame({"a": range(100)})
        train, test = time_split(df, test_size=0.2)
        self.assertTrue((train.index < test.index[0]).all())

    def test_clip_predictions(self):
        from ml.utils.helpers import clip_predictions
        preds = np.array([-100.0, 50.0, 200.0])
        clipped = clip_predictions(preds, lower=0.0, upper=100.0)
        self.assertEqual(clipped[0], 0.0)
        self.assertEqual(clipped[2], 100.0)

    def test_ensure_dir_creates_directory(self):
        from ml.utils.helpers import ensure_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = os.path.join(tmpdir, "subdir", "nested")
            result = ensure_dir(new_path)
            self.assertTrue(os.path.isdir(new_path))
            self.assertEqual(result, new_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Config Loader
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigLoader(unittest.TestCase):

    def test_default_config_loads(self):
        from ml.utils.config_loader import config
        self.assertIsNotNone(config)

    def test_dot_notation_access(self):
        from ml.utils.config_loader import config
        self.assertEqual(
            config.data.target_column,
            "DE_load_actual_entsoe_transparency"
        )

    def test_nested_dot_access(self):
        from ml.utils.config_loader import config
        self.assertIsNotNone(config.forecasting.test_size)

    def test_get_with_default(self):
        from ml.utils.config_loader import config
        val = config.get("nonexistent_key", default="fallback")
        self.assertEqual(val, "fallback")

    def test_as_dict_returns_dict(self):
        from ml.utils.config_loader import config
        d = config.as_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("data", d)

    def test_missing_key_raises_attribute_error(self):
        from ml.utils.config_loader import config
        with self.assertRaises(AttributeError):
            _ = config.nonexistent_key_xyz

    def test_custom_config_path(self):
        from ml.utils.config_loader import ConfigLoader
        cfg = ConfigLoader()
        self.assertIsNotNone(cfg.data.target_column)


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Forecasting Models
# ═══════════════════════════════════════════════════════════════════════════════

def _make_model_data(n: int = 500):
    from ml.feature_engineering.feature_pipeline import FeatureEngineer
    df_raw = _make_energy_df(n)
    fe = FeatureEngineer()
    df = fe.transform(df_raw)
    feat_cols = fe.get_feature_columns(df)
    X = df[feat_cols]
    y = df["DE_load_actual_entsoe_transparency"]
    split = int(len(X) * 0.8)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


class TestLinearRegressionModelFull(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.forecasting.linear_model import LinearRegressionModel
        cls.X_tr, cls.y_tr, cls.X_te, cls.y_te = _make_model_data()
        cls.model = LinearRegressionModel()
        cls.model.fit(cls.X_tr, cls.y_tr)

    def test_name(self):
        self.assertEqual(self.model.name, "LinearRegression")

    def test_fit_returns_self(self):
        from ml.forecasting.linear_model import LinearRegressionModel
        m = LinearRegressionModel()
        ret = m.fit(self.X_tr, self.y_tr)
        self.assertIs(ret, m)

    def test_is_fitted_after_fit(self):
        self.assertTrue(self.model._is_fitted)

    def test_predict_shape(self):
        preds = self.model.predict(self.X_te)
        self.assertEqual(len(preds), len(self.X_te))

    def test_predict_all_finite(self):
        preds = self.model.predict(self.X_te)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_evaluate_returns_metrics_dict(self):
        m = self.model.evaluate(self.X_te, self.y_te)
        for k in ["rmse", "mae", "mape"]:
            self.assertIn(k, m)
            self.assertGreater(m[k], 0)

    def test_predict_before_fit_raises(self):
        from ml.forecasting.linear_model import LinearRegressionModel
        m = LinearRegressionModel()
        with self.assertRaises(Exception):
            m.predict(self.X_te)


class TestRandomForestModelFull(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.forecasting.random_forest_model import RandomForestModel
        cls.X_tr, cls.y_tr, cls.X_te, cls.y_te = _make_model_data()
        cls.model = RandomForestModel()
        cls.model.fit(cls.X_tr, cls.y_tr)

    def test_name(self):
        self.assertEqual(self.model.name, "RandomForest")

    def test_predict_shape(self):
        preds = self.model.predict(self.X_te)
        self.assertEqual(len(preds), len(self.X_te))

    def test_feature_importances_length(self):
        imp = self.model.feature_importances_
        self.assertEqual(len(imp), self.X_tr.shape[1])

    def test_feature_importances_sum_to_one(self):
        imp = self.model.feature_importances_
        self.assertAlmostEqual(imp.sum(), 1.0, places=5)

    def test_feature_importances_non_negative(self):
        imp = self.model.feature_importances_
        self.assertTrue((imp >= 0).all())


class TestXGBoostModelFull(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.forecasting.xgboost_model import XGBoostModel
        from ml.utils.helpers import compute_metrics
        cls.X_tr, cls.y_tr, cls.X_te, cls.y_te = _make_model_data()
        cls.model = XGBoostModel()
        cls.model.fit(cls.X_tr, cls.y_tr)
        cls.compute_metrics = staticmethod(compute_metrics)

    def test_name(self):
        self.assertEqual(self.model.name, "XGBoost")

    def test_predict_shape(self):
        preds = self.model.predict(self.X_te)
        self.assertEqual(len(preds), len(self.X_te))

    def test_beats_naive_predictor(self):
        preds = self.model.predict(self.X_te)
        xgb_rmse = self.compute_metrics(self.y_te.values, preds)["rmse"]
        naive_rmse = self.compute_metrics(
            self.y_te.values, np.full(len(self.y_te), self.y_tr.mean())
        )["rmse"]
        self.assertLess(xgb_rmse, naive_rmse)

    def test_predictions_all_finite(self):
        preds = self.model.predict(self.X_te)
        self.assertTrue(np.all(np.isfinite(preds)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
