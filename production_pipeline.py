#!/usr/bin/env python
"""
Production Pipeline: Daily Retrain, MLflow Tracking, Drift Detection
"""
import warnings
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
FEATURES_FILE = DATA_DIR / "matches_pre_match.parquet"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("football-betting")

TARGETS = {
    "result": {"type": "multiclass", "classes": [-1, 0, 1]},
    "over_25": {"type": "binary", "threshold": 2.5},
    "btts": {"type": "binary", "threshold": None},
}

EXCLUDE_FEATURES = {
    "Season", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Result",
    "imp_prob_H", "imp_prob_D", "imp_prob_A", "overround",
    "imp_prob_O25", "imp_prob_U25", "ou25_favorite",
    "favorite", "favorite_prob", "odds_ratio_H_A", "draw_prob",
    "home_win_prob", "away_win_prob", "market_entropy", "market_max_prob",
    "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5",
    "B365AHH", "B365AHA",
    # "Div" REMOVED - now used as categorical feature to distinguish league dynamics
}


def prepare_data(df: pd.DataFrame) -> tuple:
    df = df.sort_values("Date").reset_index(drop=True)
    df["target_result"] = df["Result"].map({-1: 0, 0: 1, 1: 2})
    df["target_over_25"] = (df["FTHG"] + df["FTAG"] > 2.5).astype(int)
    df["target_btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)

    feat_cols = [c for c in df.columns if c not in EXCLUDE_FEATURES and not c.startswith("target_")]
    cat_cols = df[feat_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype("category")

    num_cols = [c for c in feat_cols if c not in cat_cols]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    return df, feat_cols, cat_cols


def get_walkforward_splits(df: pd.DataFrame, n_splits: int = 8, embargo_months: int = 1):
    df = df.sort_values("Date").copy()
    df["YearMonth"] = df["Date"].dt.to_period("M")
    months = sorted(df["YearMonth"].unique())

    min_train_months = 12
    test_months_per_split = max(1, (len(months) - min_train_months) // n_splits)

    splits = []
    for i in range(n_splits):
        test_start_idx = min_train_months + i * test_months_per_split
        test_end_idx = test_start_idx + test_months_per_split
        if test_end_idx >= len(months):
            break
        train_end_idx = test_start_idx - embargo_months
        if train_end_idx <= 0:
            continue
        train_months = months[:train_end_idx]
        test_months = months[test_start_idx:test_end_idx]
        train_mask = df["YearMonth"].isin(train_months)
        test_mask = df["YearMonth"].isin(test_months)
        if train_mask.sum() < 1000 or test_mask.sum() < 50:
            continue
        splits.append((train_mask, test_mask))
    return splits


def train_lgb(X_train, y_train, X_val, y_val, cat_cols, task_type, params):
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    lgb_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=lgb_train)

    default_params = {
        "objective": "multiclass" if task_type == "multiclass" else "binary",
        "metric": "multi_logloss" if task_type == "multiclass" else "binary_logloss",
        "num_class": 3 if task_type == "multiclass" else 1,
        "learning_rate": 0.03, "num_leaves": 63, "max_depth": -1,
        "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 5, "lambda_l1": 0.1, "lambda_l2": 0.1,
        "verbosity": -1, "n_jobs": -1, "seed": 42, "deterministic": True,
    }
    default_params.update(params)

    model = lgb.train(
        default_params, lgb_train, valid_sets=[lgb_val],
        num_boost_round=2000, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    return model


def calibrate_probabilities(model, X_val, y_val, task_type):
    val_probs = model.predict(X_val, num_iteration=model.best_iteration)
    if task_type == "multiclass":
        calibrators = []
        for i in range(val_probs.shape[1]):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(val_probs[:, i], (y_val == i).astype(int))
            calibrators.append(iso)
        return calibrators
    else:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_probs, y_val)
        return iso


def predict_calibrated(model, X, calibrators, task_type):
    probs = model.predict(X, num_iteration=model.best_iteration)
    if task_type == "multiclass":
        cal_probs = np.column_stack([calibrators[i].predict(probs[:, i]) for i in range(probs.shape[1])])
        cal_probs = cal_probs / cal_probs.sum(axis=1, keepdims=True)
        return cal_probs
    else:
        return calibrators.predict(probs)


def optuna_objective(trial, df, feat_cols, cat_cols, target_name, task_type, n_splits=5):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }
    splits = get_walkforward_splits(df, n_splits=n_splits)
    scores = []
    target_col = f"target_{target_name}"
    for train_mask, test_mask in splits:
        X_train = df.loc[train_mask, feat_cols]
        y_train = df.loc[train_mask, target_col]
        X_val = df.loc[test_mask, feat_cols]
        y_val = df.loc[test_mask, target_col]
        try:
            model = train_lgb(X_train, y_train, X_val, y_val, cat_cols, task_type, params)
            val_probs = model.predict(X_val, num_iteration=model.best_iteration)
            scores.append(log_loss(y_val, val_probs))
        except Exception:
            return 1.0
    return np.mean(scores)


def calculate_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, feat_cols: list) -> dict:
    """Calculate PSI (Population Stability Index) for feature drift detection."""
    drift_report = {}
    for col in feat_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_vals = reference_df[col].dropna()
        cur_vals = current_df[col].dropna()
        if len(ref_vals) < 100 or len(cur_vals) < 100:
            continue

        # Bin into 10 quantiles based on reference
        try:
            bins = np.percentile(ref_vals, np.linspace(0, 100, 11))
            bins = np.unique(bins)
            if len(bins) < 3:
                continue

            ref_hist, _ = np.histogram(ref_vals, bins=bins)
            cur_hist, _ = np.histogram(cur_vals, bins=bins)

            ref_pct = ref_hist / ref_hist.sum()
            cur_pct = cur_hist / cur_hist.sum()

            # Avoid division by zero
            ref_pct = np.clip(ref_pct, 0.0001, 1)
            cur_pct = np.clip(cur_pct, 0.0001, 1)

            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            drift_report[col] = {"psi": float(psi), "drift": bool(psi > 0.2)}
        except Exception:
            pass

    high_drift = {k: v for k, v in drift_report.items() if v["drift"]}
    return {"drift_features": high_drift, "max_psi": max([v["psi"] for v in drift_report.values()]) if drift_report else 0}


def train_production_models(retrain: bool = True):
    """Train production models with MLflow tracking."""
    print("=" * 60)
    print("PRODUCTION MODEL TRAINING")
    print("=" * 60)

    df = pd.read_parquet(FEATURES_FILE)
    print(f"[LOAD] {len(df):,} matches")
    df, feat_cols, cat_cols = prepare_data(df)
    print(f"[FEATURES] {len(feat_cols)} features")

    # Reference distribution for drift detection (last 6 months of training data)
    ref_date = df["Date"].max() - pd.DateOffset(months=6)
    reference_df = df[df["Date"] >= ref_date].copy()

    all_models = {}
    all_calibrators = {}
    best_params_all = {}

    for target_name, target_info in TARGETS.items():
        print(f"\n[TRAIN] {target_name} ({target_info['type']})")
        target_col = f"target_{target_name}"

        with mlflow.start_run(run_name=f"train_{target_name}_{datetime.now().strftime('%Y%m%d')}"):
            mlflow.log_params({"target": target_name, "type": target_info["type"], "n_features": len(feat_cols)})

            if retrain:
                early_mask = df["Date"] < df["Date"].quantile(0.3)
                study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(
                    lambda t: optuna_objective(t, df[early_mask], feat_cols, cat_cols, target_name, target_info["type"]),
                    n_trials=30, timeout=120, show_progress_bar=False
                )
                best_params = study.best_params
                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_logloss", study.best_value)
            else:
                # Load previous best params
                param_file = MODEL_DIR / f"{target_name}_best_params.json"
                if param_file.exists():
                    best_params = json.loads(param_file.read_text())
                else:
                    best_params = {}

            # Full walk-forward training on all data (latest fold for production)
            splits = get_walkforward_splits(df, n_splits=8)
            fold_models = []
            fold_calibrators = []

            for fold, (train_mask, test_mask) in enumerate(splits):
                X_train = df.loc[train_mask, feat_cols]
                y_train = df.loc[train_mask, target_col]
                X_val = df.loc[test_mask, feat_cols]
                y_val = df.loc[test_mask, target_col]

                model = train_lgb(X_train, y_train, X_val, y_val, cat_cols, target_info["type"], best_params)
                cal = calibrate_probabilities(model, X_val, y_val, target_info["type"])

                fold_models.append(model)
                fold_calibrators.append(cal)

                # Log fold metrics
                val_probs = predict_calibrated(model, X_val, cal, target_info["type"])
                val_loss = log_loss(y_val, val_probs)
                mlflow.log_metric(f"fold_{fold}_logloss", val_loss)

            # Use latest fold model for production
            production_model = fold_models[-1]
            production_cal = fold_calibrators[-1]

            # Save model artifacts
            model_path = MODEL_DIR / f"{target_name}_model.txt"
            production_model.save_model(str(model_path))

            cal_path = MODEL_DIR / f"{target_name}_calibrator.pkl"
            with open(cal_path, "wb") as f:
                pickle.dump(production_cal, f)

            param_path = MODEL_DIR / f"{target_name}_best_params.json"
            param_path.write_text(json.dumps(best_params))

            # Log to MLflow
            mlflow.lightgbm.log_model(production_model, f"model_{target_name}")
            mlflow.log_artifact(str(cal_path))
            mlflow.log_artifact(str(param_path))

            # Feature importance
            importance = pd.DataFrame({
                "feature": feat_cols,
                "importance": production_model.feature_importance(importance_type="gain")
            }).sort_values("importance", ascending=False)
            imp_path = MODEL_DIR / f"{target_name}_importance.csv"
            importance.to_csv(imp_path, index=False)
            mlflow.log_artifact(str(imp_path))

            all_models[target_name] = production_model
            all_calibrators[target_name] = production_cal
            best_params_all[target_name] = best_params

            print(f"  Saved: {model_path}, CV logloss: {study.best_value if retrain else 'loaded'}")

    # Drift detection
    print("\n[DRIFT] Checking feature drift...")
    current_df = df[df["Date"] >= df["Date"].max() - pd.DateOffset(months=1)].copy()
    drift_report = calculate_feature_drift(reference_df, current_df, feat_cols)

    drift_path = MODEL_DIR / "drift_report.json"
    drift_path.write_text(json.dumps(drift_report, indent=2))
    mlflow.log_artifact(str(drift_path))

    if drift_report["drift_features"]:
        print(f"  WARNING: {len(drift_report['drift_features'])} features with PSI > 0.2")
        for feat, info in list(drift_report["drift_features"].items())[:10]:
            print(f"    {feat}: PSI={info['psi']:.3f}")
    else:
        print("  No significant drift detected")

    print("\n[DONE] Production models ready")
    return all_models, all_calibrators, best_params_all, feat_cols, cat_cols


def generate_daily_predictions(models: dict, calibrators: dict, feat_cols: list, cat_cols: list,
                               match_date: str = None) -> pd.DataFrame:
    """Generate predictions for upcoming matches."""
    df = pd.read_parquet(FEATURES_FILE)
    df, _, _ = prepare_data(df)

    if match_date:
        target_date = pd.Timestamp(match_date)
    else:
        target_date = df["Date"].max() + pd.DateOffset(days=1)

    # Get matches for target date (or next available)
    upcoming = df[df["Date"] >= target_date].copy()
    if len(upcoming) == 0:
        print("No upcoming matches found")
        return pd.DataFrame()

    print(f"[PREDICT] {len(upcoming)} matches for {target_date.date()}")

    predictions = []
    for target_name, target_info in TARGETS.items():
        model = models[target_name]
        cal = calibrators[target_name]
        probs = predict_calibrated(model, upcoming[feat_cols], cal, target_info["type"])

        if target_info["type"] == "multiclass":
            upcoming[f"pred_{target_name}_A"] = probs[:, 0]
            upcoming[f"pred_{target_name}_D"] = probs[:, 1]
            upcoming[f"pred_{target_name}_H"] = probs[:, 2]
        else:
            upcoming[f"pred_{target_name}_1"] = probs
            upcoming[f"pred_{target_name}_0"] = 1 - probs

    # Save predictions
    pred_cols = ["Date", "HomeTeam", "AwayTeam", "Div", "Season"] + \
                [c for c in upcoming.columns if c.startswith("pred_")]
    upcoming[pred_cols].to_parquet(DATA_DIR / f"predictions_{target_date.strftime('%Y%m%d')}.parquet", index=False)

    return upcoming[pred_cols]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Retrain models with Optuna")
    parser.add_argument("--predict", action="store_true", help="Generate daily predictions")
    parser.add_argument("--date", type=str, help="Target date for predictions (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.retrain or not (MODEL_DIR / "result_model.txt").exists():
        models, calibrators, params, feat_cols, cat_cols = train_production_models(retrain=True)
    else:
        # Load existing models
        print("[LOAD] Loading existing production models...")
        models = {}
        calibrators = {}
        feat_cols = None
        for target_name in TARGETS:
            model_path = MODEL_DIR / f"{target_name}_model.txt"
            cal_path = MODEL_DIR / f"{target_name}_calibrator.pkl"
            models[target_name] = lgb.Booster(model_file=str(model_path))
            with open(cal_path, "rb") as f:
                calibrators[target_name] = pickle.load(f)
        df = pd.read_parquet(FEATURES_FILE)
        df, feat_cols, cat_cols = prepare_data(df)

    if args.predict:
        generate_daily_predictions(models, calibrators, feat_cols, cat_cols, args.date)


if __name__ == "__main__":
    main()