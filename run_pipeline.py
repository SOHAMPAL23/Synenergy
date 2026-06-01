"""
EnerVision AI - Pipeline Entry Point
Run the complete ML pipeline from the command line.

Usage
-----
    # Full pipeline run (train + forecast + explain + recommend)
    python run_pipeline.py

    # Skip retraining — use saved model for inference only
    python run_pipeline.py --skip-training

    # Run unit tests only
    python run_pipeline.py --test

    # Quick smoke test on a small data slice
    python run_pipeline.py --dry-run
"""

import argparse
import json
import os
import sys
import unittest

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def run_pipeline(skip_training: bool = False) -> dict:
    from ml.pipeline import EnerVisionPipeline
    pipeline = EnerVisionPipeline()
    results = pipeline.run(skip_training=skip_training)
    return results


def run_tests() -> bool:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(ROOT, "ml", "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def dry_run() -> None:
    """Quick smoke test: run ingestion + features on first 2000 rows only."""
    print("\n[DRY RUN] Smoke testing ingestion + feature engineering…\n")
    import yaml
    from ml.utils.config_loader import ConfigLoader

    # Temporarily patch max_rows
    cfg_path = os.path.join(ROOT, "ml", "config", "config.yaml")
    with open(cfg_path) as f:
        cfg_data = yaml.safe_load(f)
    cfg_data["data"]["max_rows"] = 2000

    tmp_cfg_path = os.path.join(ROOT, "ml", "config", "_tmp_dryrun_config.yaml")
    with open(tmp_cfg_path, "w") as f:
        yaml.dump(cfg_data, f)

    try:
        cfg = ConfigLoader(tmp_cfg_path)
        from ml.ingestion.data_loader import DataLoader
        from ml.ingestion.schema_validator import SchemaValidator
        from ml.preprocessing.cleaner import DataCleaner
        from ml.feature_engineering.feature_pipeline import FeatureEngineer

        loader = DataLoader(cfg=cfg)
        df_raw = loader.load()
        print(f"  [OK] Loaded {len(df_raw)} rows")

        validator = SchemaValidator(cfg=cfg)
        validator.validate(df_raw)
        print(f"  [OK] Schema validation passed")

        cleaner = DataCleaner(cfg=cfg)
        df_clean = cleaner.clean(df_raw)
        print(f"  [OK] Cleaning done: {len(df_clean)} rows")

        fe = FeatureEngineer(cfg=cfg)
        df_feat = fe.transform(df_clean)
        print(f"  [OK] Features: {df_feat.shape[1]} columns, {df_feat.shape[0]} rows")
        print(f"  [OK] Feature columns: {fe.get_feature_columns(df_feat)[:5]} ...")
        print("\n[DRY RUN] Smoke test PASSED\n")
    finally:
        if os.path.exists(tmp_cfg_path):
            os.remove(tmp_cfg_path)


def print_results_summary(results: dict) -> None:
    print("\n" + "=" * 65)
    print("  EnerVision AI — Pipeline Results Summary")
    print("=" * 65)

    print(f"\nBest Model: {results['best_model']}")

    print("\nModel Metrics:")
    for model_name, metrics in results.get("metrics", {}).items():
        print(f"   {model_name:<25} RMSE={metrics['rmse']:>10.2f} | "
              f"MAE={metrics['mae']:>10.2f} | MAPE={metrics['mape']:>6.2f}%")

    print("\nForecasts Generated:")
    for horizon in results.get("forecasts", {}).keys():
        print(f"   • {horizon}")

    print("\nAnomaly Detection:")
    for method, count in results.get("anomaly_summary", {}).items():
        print(f"   {method:<30} {count:>6} anomalies")

    print("\nTop Recommendations:")
    for i, rec in enumerate(results.get("recommendations", [])[:5], 1):
        print(f"   {i}. [{rec['priority']}] {rec['title']}")
        print(f"      Est. saving: {rec['estimated_saving_pct']:.1f}%")

    if results.get("shap_importance"):
        print("\nTop 5 SHAP Features:")
        for feat in results["shap_importance"][:5]:
            print(f"   {feat['feature']:<30} {feat['mean_abs_shap']:>10.4f}")

    print("\nOutputs saved to: ml/outputs/")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="EnerVision AI — Energy Forecasting Pipeline"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load pre-trained model from disk instead of retraining.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the full unit test suite.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a quick smoke test without full training.",
    )
    args = parser.parse_args()

    if args.test:
        print("\n[TEST] Running EnerVision AI unit test suite…\n")
        success = run_tests()
        sys.exit(0 if success else 1)

    if args.dry_run:
        dry_run()
        sys.exit(0)

    print("\nStarting EnerVision AI Pipeline...\n")
    results = run_pipeline(skip_training=args.skip_training)
    print_results_summary(results)

    # Save summary JSON
    out_path = os.path.join(ROOT, "ml", "outputs", "pipeline_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Full summary saved -> {out_path}")


if __name__ == "__main__":
    main()
