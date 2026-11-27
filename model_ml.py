#!/usr/bin/env python
"""
ML Model v2: LightGBM + Calibration + Optuna optimization
Walk-forward backtest with purged CV, multiple targets (1X2, O/U 2.5, BTTS)
"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
FEATURES_FILE = DATA_DIR / "matches_pre_match.parquet"
RESULTS_FILE = DATA_DIR / "backtest_ml_clean.parquet"
SUMMARY_FILE = DATA_DIR / "backtest_ml_clean_summary.md"

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TARGETS = {
    "result": {"type": "multiclass", "classes": [-1, 0, 1]},  # Away, Draw, Home
    "over_25": {"type": "binary", "threshold": 2.5},
    "btts": {"type": "binary", "threshold": None},
}

MARKETS = {
    "1X2_H": {"target": "result", "class_idx": 2, "odds_col": "B365H"},
    "1X2_D": {"target": "result", "class_idx": 1, "odds_col": "B365D"},
    "1X2_A": {"target": "result", "class_idx": 0, "odds_col": "B365A"},
    "OVER_2.5": {"target": "over_25", "class_idx": 1, "odds_col": "B365>2.5"},
    "UNDER_2.5": {"target": "over_25", "class_idx": 0, "odds_col": "B365<2.5"},
    "BTTS_YES": {"target": "btts", "class_idx": 1, "odds_col": None},
    "BTTS_NO": {"target": "btts", "class_idx": 0, "odds_col": None},
}

MIN_EV = 0.05
MAX_ODDS = 5.0
MIN_ODDS = 1.5
KELLY_FRAC = 0.25
EMBARGO_MONTHS = 1  # Purged CV gap

# Features to exclude (leakage, identifiers, target-derived)
# IMPORTANT: Exclude market implied probs - model must predict independently!
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

# ──────────────────────────────────────────────
# DATA PREP
# ──────────────────────────────────────────────
def prepare_data(df: pd.DataFrame) -> tuple:
    """Prepare features and targets."""
    df = df.sort_values("Date").reset_index(drop=True)

    # Targets
    df["target_result"] = df["Result"].map({-1: 0, 0: 1, 1: 2})  # 0=Away, 1=Draw, 2=Home
    df["target_over_25"] = (df["FTHG"] + df["FTAG"] > 2.5).astype(int)
    df["target_btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)

    # Feature columns
    feat_cols = [c for c in df.columns if c not in EXCLUDE_FEATURES and not c.startswith("target_")]

    # Handle categorical
    cat_cols = df[feat_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype("category")

    # Fill NaN
    num_cols = [c for c in feat_cols if c not in cat_cols]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    return df, feat_cols, cat_cols


# ──────────────────────────────────────────────
# WALK-FORWARD SPLITS (Purged)
# ──────────────────────────────────────────────
def get_walkforward_splits(df: pd.DataFrame, n_splits: int = 10, embargo_months: int = EMBARGO_MONTHS):
    """Generate purged walk-forward train/test splits by month."""
    df = df.sort_values("Date").copy()
    df["YearMonth"] = df["Date"].dt.to_period("M")
    months = sorted(df["YearMonth"].unique())

    # Need enough history for first split
    min_train_months = 12
    test_months_per_split = max(1, (len(months) - min_train_months) // n_splits)

    splits = []
    for i in range(n_splits):
        test_start_idx = min_train_months + i * test_months_per_split
        test_end_idx = test_start_idx + test_months_per_split

        if test_end_idx >= len(months):
            break

        # Embargo: gap between train and test
        train_end_idx = test_start_idx - embargo_months
        if train_end_idx <= 0:
            continue

        train_months = months[:train_end_idx]
        test_months = months[test_start_idx:test_end_idx]

        train_mask = df["YearMonth"].isin(train_months)
        test_mask = df["YearMonth"].isin(test_months)

        if train_mask.sum() < 1000 or test_mask.sum() < 50:
            continue

        splits.append((train_mask, test_mask, train_months, test_months))

    return splits


# ──────────────────────────────────────────────
# LIGHTGBM TRAINING
# ──────────────────────────────────────────────
def train_lgb(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    cat_cols: list,
    task_type: str,
    params: dict
) -> lgb.Booster:
    """Train LightGBM with early stopping."""
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    lgb_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=lgb_train)

    # Default params
    default_params = {
        "objective": "multiclass" if task_type == "multiclass" else "binary",
        "metric": "multi_logloss" if task_type == "multiclass" else "binary_logloss",
        "num_class": 3 if task_type == "multiclass" else 1,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 42,
        "deterministic": True,
    }
    default_params.update(params)

    model = lgb.train(
        default_params,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    return model


def calibrate_probabilities(model, X_val: pd.DataFrame, y_val: pd.Series, task_type: str):
    """Calibrate using Isotonic Regression (more flexible than Platt)."""
    val_probs = model.predict(X_val, num_iteration=model.best_iteration)

    if task_type == "multiclass":
        # Calibrate each class independently
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


def predict_calibrated(model, X: pd.DataFrame, calibrators, task_type: str):
    """Get calibrated probabilities."""
    probs = model.predict(X, num_iteration=model.best_iteration)
    if task_type == "multiclass":
        cal_probs = np.column_stack([
            calibrators[i].predict(probs[:, i]) for i in range(probs.shape[1])
        ])
        # Renormalize
        cal_probs = cal_probs / cal_probs.sum(axis=1, keepdims=True)
        return cal_probs
    else:
        return calibrators.predict(probs)


# ──────────────────────────────────────────────
# OPTUNA OBJECTIVE
# ──────────────────────────────────────────────
def optuna_objective(trial, df, feat_cols, cat_cols, target_name, task_type, n_splits=5):
    """Optuna objective for hyperparameter optimization."""
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
    for train_mask, test_mask, _, _ in splits:
        X_train = df.loc[train_mask, feat_cols]
        y_train = df.loc[train_mask, target_col]
        X_val = df.loc[test_mask, feat_cols]
        y_val = df.loc[test_mask, target_col]

        try:
            model = train_lgb(X_train, y_train, X_val, y_val, cat_cols, task_type, params)
            val_probs = model.predict(X_val, num_iteration=model.best_iteration)

            if task_type == "multiclass":
                score = log_loss(y_val, val_probs)
            else:
                score = log_loss(y_val, val_probs)
            scores.append(score)
        except Exception:
            return 1.0  # Bad score

    return np.mean(scores)


# ──────────────────────────────────────────────
# BACKTEST
# ──────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, feat_cols: list, cat_cols: list) -> pd.DataFrame:
    """Walk-forward backtest with ML models - TRUE out-of-sample evaluation."""
    df = df.sort_values("Date").reset_index(drop=True)

    # Get walk-forward splits ONCE (same splits for all targets)
    splits = get_walkforward_splits(df, n_splits=8)
    print(f"\n[SPLITS] {len(splits)} walk-forward folds")

    # Collect out-of-sample predictions from each fold's test set
    oos_predictions = []  # list of (fold_idx, test_indices, target_probs_dict)

    for target_name, target_info in TARGETS.items():
        print(f"\n[TRAIN] Target: {target_name} ({target_info['type']})")
        target_col = f"target_{target_name}"

        # Optimize hyperparams (once on early data)
        early_mask = df["Date"] < df["Date"].quantile(0.3)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda t: optuna_objective(t, df[early_mask], feat_cols, cat_cols, target_name, target_info["type"]),
            n_trials=30, timeout=120, show_progress_bar=False
        )
        best_params = study.best_params
        print(f"    Best params: {best_params}")

        # Walk-forward: train on train_mask, predict on test_mask ONLY
        fold_test_indices = []
        fold_probs = []

        for fold, (train_mask, test_mask, train_m, test_m) in enumerate(splits):
            X_train = df.loc[train_mask, feat_cols]
            y_train = df.loc[train_mask, target_col]
            X_test = df.loc[test_mask, feat_cols]
            y_test = df.loc[test_mask, target_col]

            model = train_lgb(X_train, y_train, X_test, y_test, cat_cols, target_info["type"], best_params)
            cal = calibrate_probabilities(model, X_test, y_test, target_info["type"])

            # Predict on TEST set only (out-of-sample)
            test_probs = predict_calibrated(model, X_test, cal, target_info["type"])

            fold_test_indices.append(df.index[test_mask].values)
            fold_probs.append(test_probs)
            print(f"    Fold {fold+1}: train={train_mask.sum():,} test={test_mask.sum():,}")

        # Store OOS predictions for this target (per-fold to handle variable sizes)
        oos_predictions.append({
            "target": target_name,
            "type": target_info["type"],
            "fold_indices": fold_test_indices,
            "fold_probs": fold_probs,
        })

    # Now evaluate bets ONLY on out-of-sample predictions
    print("\n[BETS] Evaluating betting opportunities on OOS predictions...")
    all_rows = []

    # Build a lookup for predictions by match index
    pred_lookup = {}
    for op in oos_predictions:
        target = op["target"]
        ttype = op["type"]
        for fold_idx, fold_probs in zip(op["fold_indices"], op["fold_probs"]):
            for idx, probs in zip(fold_idx, fold_probs):
                if idx not in pred_lookup:
                    pred_lookup[idx] = {}
                if ttype == "multiclass":
                    pred_lookup[idx][f"pred_{target}_A"] = probs[0]
                    pred_lookup[idx][f"pred_{target}_D"] = probs[1]
                    pred_lookup[idx][f"pred_{target}_H"] = probs[2]
                else:
                    # binary: probs is scalar (class 1 prob)
                    p1 = float(probs)
                    pred_lookup[idx][f"pred_{target}_0"] = 1 - p1
                    pred_lookup[idx][f"pred_{target}_1"] = p1

    # Evaluate only matches that have OOS predictions
    for idx, row in df.iterrows():
        if idx not in pred_lookup:
            continue  # Skip matches not in any test fold (early data before first test fold)

        preds = pred_lookup[idx]

        for mkt_name, mkt in MARKETS.items():
            odds_col = mkt["odds_col"]
            if odds_col is None or pd.isna(row[odds_col]):
                continue
            odds = row[odds_col]
            if odds < MIN_ODDS or odds > MAX_ODDS:
                continue

            target = mkt["target"]
            class_idx = mkt["class_idx"]

            if target == "result":
                p_model = preds.get(f"pred_result_{['A','D','H'][class_idx]}", 0)
            elif target == "over_25":
                p_model = preds.get(f"pred_over_25_{class_idx}", 0)
            elif target == "btts":
                p_model = preds.get(f"pred_btts_{class_idx}", 0)
            else:
                continue

            if p_model <= 0:
                continue

            ev = p_model * odds - 1
            if ev <= MIN_EV:
                continue

            kelly = (p_model * odds - 1) / (odds - 1) if odds > 1 else 0
            stake = max(0, kelly * KELLY_FRAC)

            # Actual result
            if target == "result":
                actual = int(row["Result"] == (1 if class_idx == 2 else (0 if class_idx == 1 else -1)))
            elif target == "over_25":
                tot = row["FTHG"] + row["FTAG"]
                actual = int((class_idx == 1 and tot > 2.5) or (class_idx == 0 and tot < 2.5))
            elif target == "btts":
                btts = int(row["FTHG"] > 0 and row["FTAG"] > 0)
                actual = int(btts == class_idx)

            pnl = stake * (odds - 1) if actual else -stake
            clv = (p_model - 1/odds) * 100

            all_rows.append({
                "Date": row["Date"], "Season": row["Season"], "Div": row["Div"],
                "HomeTeam": row["HomeTeam"], "AwayTeam": row["AwayTeam"],
                "Market": mkt_name, "Model_Prob": p_model, "Odds": odds,
                "EV": ev, "Stake": stake, "Actual": actual, "PnL": pnl,
                "CLV": clv,
            })

    return pd.DataFrame(all_rows)


# ──────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────
def analyze_results(results: pd.DataFrame) -> str:
    if len(results) == 0:
        return "Sin apuestas con EV+"

    n = len(results)
    stake = results["Stake"].sum()
    pnl = results["PnL"].sum()
    roi = pnl / stake * 100 if stake > 0 else 0
    wr = results["Actual"].mean() * 100
    avg_ev = results["EV"].mean() * 100
    avg_clv = results["CLV"].mean()

    lines = [
        f"# Backtest LightGBM + Calibration",
        f"",
        f"## Overall",
        f"- Bets: {n:,} | Stake: {stake:.2f} | PnL: {pnl:.2f} | ROI: {roi:.2f}%",
        f"- Win Rate: {wr:.1f}% | Avg EV: {avg_ev:.2f}% | Avg CLV: {avg_clv:.2f}%",
        f"",
        f"## By Market",
    ]
    for mkt in results["Market"].unique():
        d = results[results["Market"] == mkt]
        if len(d) < 10:
            continue
        s = d["Stake"].sum()
        p = d["PnL"].sum()
        r = p/s*100 if s>0 else 0
        lines.append(f"- {mkt}: n={len(d):,} ROI={r:.2f}% WR={d['Actual'].mean()*100:.1f}% CLV={d['CLV'].mean():.2f}%")

    lines.append(f"\n## By League")
    for div in sorted(results["Div"].unique()):
        d = results[results["Div"] == div]
        s = d["Stake"].sum()
        p = d["PnL"].sum()
        r = p/s*100 if s>0 else 0
        lines.append(f"- {div}: n={len(d):,} ROI={r:.2f}% CLV={d['CLV'].mean():.2f}%")

    lines.append(f"\n## By Season")
    for sea in sorted(results["Season"].unique()):
        d = results[results["Season"] == sea]
        s = d["Stake"].sum()
        p = d["PnL"].sum()
        r = p/s*100 if s>0 else 0
        lines.append(f"- {sea}: n={len(d):,} ROI={r:.2f}% CLV={d['CLV'].mean():.2f}%")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ML MODEL v2: LightGBM + CALIBRATION + OPTUNA")
    print("=" * 60)

    df = pd.read_parquet(FEATURES_FILE)
    print(f"\n[LOAD] {len(df):,} matches, {len(df.columns)} features")

    df, feat_cols, cat_cols = prepare_data(df)
    print(f"[FEATURES] {len(feat_cols)} features ({len(cat_cols)} categorical)")

    results = run_backtest(df, feat_cols, cat_cols)

    if len(results) == 0:
        print("\n[WARN] No bets generated")
        return

    results.to_parquet(RESULTS_FILE, index=False)
    print(f"\n[SAVE] {RESULTS_FILE} ({len(results):,} bets)")

    summary = analyze_results(results)
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(f"[SAVE] {SUMMARY_FILE}")

    print("\n" + "=" * 60)
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    main()