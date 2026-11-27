#!/usr/bin/env python
"""
Validate model on a specific historical day using our dataset's odds
"""
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
FEATURES_FILE = DATA_DIR / "matches_pre_match.parquet"

EXCLUDE_FEATURES = {
    "Div", "Season", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Result",
    "imp_prob_H", "imp_prob_D", "imp_prob_A", "overround",
    "imp_prob_O25", "imp_prob_U25", "ou25_favorite",
    "favorite", "favorite_prob", "odds_ratio_H_A", "draw_prob",
    "home_win_prob", "away_win_prob", "market_entropy", "market_max_prob",
    "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5",
    "B365AHH", "B365AHA",
}

TARGETS = {
    "result": {"type": "multiclass", "classes": [-1, 0, 1]},
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


def predict_calibrated(model, X, calibrators, task_type):
    probs = model.predict(X, num_iteration=model.best_iteration)
    if task_type == "multiclass":
        cal_probs = np.column_stack([
            calibrators[i].predict(probs[:, i]) for i in range(probs.shape[1])
        ])
        cal_probs = cal_probs / cal_probs.sum(axis=1, keepdims=True)
        return cal_probs
    else:
        return calibrators.predict(probs)


def load_models():
    models = {}
    calibrators = {}
    for target_name in TARGETS:
        model_path = MODEL_DIR / f"{target_name}_model.txt"
        cal_path = MODEL_DIR / f"{target_name}_calibrator.pkl"
        if model_path.exists():
            models[target_name] = lgb.Booster(model_file=str(model_path))
            with open(cal_path, "rb") as f:
                calibrators[target_name] = joblib.load(f)
    return models, calibrators


def main(target_date: str = "2025-05-25"):
    print("=" * 60)
    print(f"VALIDATION - {target_date} (Historical odds)")
    print("=" * 60)

    # Load full historical data
    df = pd.read_parquet(FEATURES_FILE)
    df = df.sort_values("Date").reset_index(drop=True)

    # Get target day matches
    target_dt = pd.Timestamp(target_date)
    day_matches = df[df["Date"] == target_dt].copy()
    print(f"\n[MATCHES] {len(day_matches)} matches on {target_date}")

    # Historical data BEFORE target date (for features)
    hist_df = df[df["Date"] < target_dt].copy()
    print(f"[HISTORY] {len(hist_df)} matches before {target_date}")

    # Prepare features on full historical data
    hist_df, feat_cols, cat_cols = prepare_data(hist_df)

    # Build features for target day using historical data
    print("\n[FEATURES] Building features for target day...")
    from fetch_live import build_live_features
    # Convert day_matches to fixture-like format (matching API structure)
    fixtures = []
    for _, row in day_matches.iterrows():
        fixtures.append({
            "fixture": {"id": 0, "date": row["Date"].isoformat()},
            "teams": {"home": {"name": row["HomeTeam"]}, "away": {"name": row["AwayTeam"]}},
            "league": {"code": row["Div"]},
            "odds": {
                "home": row.get("B365H"),
                "draw": row.get("B365D"),
                "away": row.get("B365A"),
                "over_2_5": row.get("B365>2.5"),
                "under_2_5": row.get("B365<2.5"),
            }
        })

    live_df = build_live_features(fixtures, hist_df, target_dt)

    # Load models and predict
    print("\n[MODELS] Loading and predicting...")
    models, calibrators = load_models()

    # Align features: ensure all training features exist in live_df
    for c in feat_cols:
        if c not in live_df.columns:
            # Fill with median from training data
            live_df[c] = hist_df[c].median() if c in hist_df.columns else 0

    X_live = live_df[feat_cols].copy()
    for c in cat_cols:
        if c in X_live.columns:
            X_live[c] = X_live[c].astype("category")

    for target_name, target_info in TARGETS.items():
        if target_name not in models:
            continue
        model = models[target_name]
        cal = calibrators[target_name]
        probs = predict_calibrated(model, X_live, cal, target_info["type"])

        if target_info["type"] == "multiclass":
            live_df[f"pred_{target_name}_A"] = probs[:, 0]
            live_df[f"pred_{target_name}_D"] = probs[:, 1]
            live_df[f"pred_{target_name}_H"] = probs[:, 2]
        else:
            live_df[f"pred_{target_name}_1"] = probs
            live_df[f"pred_{target_name}_0"] = 1 - probs

    # Evaluate bets using historical odds from day_matches
    print("\n[BETS] Evaluating with historical odds...")
    all_rows = []

    for i, (_, live_row) in enumerate(live_df.iterrows()):
        hist_row = day_matches.iloc[i]

        for mkt_name, mkt in MARKETS.items():
            odds_col = mkt["odds_col"]
            if odds_col is None or odds_col not in day_matches.columns:
                continue
            odds = hist_row[odds_col]
            if pd.isna(odds):
                continue
            if odds < MIN_ODDS or odds > MAX_ODDS:
                continue

            target = mkt["target"]
            class_idx = mkt["class_idx"]

            if target == "result":
                p_model = live_row.get(f"pred_result_{['A','D','H'][class_idx]}", 0)
            elif target == "over_25":
                p_model = live_row.get(f"pred_over_25_{class_idx}", 0)
            elif target == "btts":
                p_model = live_row.get(f"pred_btts_{class_idx}", 0)
            else:
                continue

            if p_model <= 0:
                continue

            ev = p_model * odds - 1
            if ev <= MIN_EV:
                continue

            kelly = (p_model * odds - 1) / (odds - 1) if odds > 1 else 0
            stake = max(0, kelly * KELLY_FRAC)
            clv = (p_model - 1/odds) * 100

            # Actual result
            if target == "result":
                actual = int(hist_row["Result"] == (1 if class_idx == 2 else (0 if class_idx == 1 else -1)))
            elif target == "over_25":
                tot = hist_row["FTHG"] + hist_row["FTAG"]
                actual = int((class_idx == 1 and tot > 2.5) or (class_idx == 0 and tot < 2.5))
            elif target == "btts":
                btts = int(hist_row["FTHG"] > 0 and hist_row["FTAG"] > 0)
                actual = int(btts == class_idx)

            pnl = stake * (odds - 1) if actual else -stake

            all_rows.append({
                "Date": hist_row["Date"].strftime("%Y-%m-%d"),
                "League": hist_row["Div"],
                "Home": hist_row["HomeTeam"],
                "Away": hist_row["AwayTeam"],
                "Market": mkt_name,
                "Model_Prob": round(p_model, 4),
                "Odds": odds,
                "EV%": round(ev * 100, 2),
                "Stake%": round(stake * 100, 2),
                "CLV%": round(clv, 2),
                "Actual": actual,
                "Result": "WIN" if actual else "LOSS",
                "PnL": round(pnl, 4),
            })

    bets_df = pd.DataFrame(all_rows)

    # Print results
    if len(bets_df) == 0:
        print("\n[NO BETS] NO EV+ BETS FOUND")
        return

    print(f"\n{'='*110}")
    print(f"BETTING RECOMMENDATIONS - {len(bets_df)} EV+ opportunities (validated with historical odds)")
    print(f"{'='*110}")
    print(f"{'Date':<12} {'League':<5} {'Match':<35} {'Market':<12} {'Prob':>6} {'Odds':>6} {'EV%':>7} {'Stake%':>7} {'CLV%':>6} {'Result':>6}")
    print(f"{'-'*110}")

    for _, row in bets_df.sort_values("EV%", ascending=False).iterrows():
        match = f"{row['Home'][:17]:<17} vs {row['Away'][:17]:<17}"
        result_mark = "[OK]" if row["Actual"] == 1 else "[FAIL]"
        print(f"{row['Date']:<12} {row['League']:<5} {match:<35} {row['Market']:<12} "
              f"{row['Model_Prob']:>6.2%} {row['Odds']:>6.2f} {row['EV%']:>7.2f}% "
              f"{row['Stake%']:>7.2f}% {row['CLV%']:>6.2f}% {result_mark:>6}")

    # Summary
    total_stake = bets_df["Stake%"].sum()
    total_pnl = bets_df["PnL"].sum()
    roi = total_pnl / total_stake * 100 if total_stake > 0 else 0
    wr = bets_df["Actual"].mean() * 100
    avg_ev = bets_df["EV%"].mean()
    avg_clv = bets_df["CLV%"].mean()

    print(f"{'-'*110}")
    print(f"TOTAL: {len(bets_df)} bets | Stake: {total_stake:.2f}% | PnL: {total_pnl:.4f} | ROI: {roi:.2f}%")
    print(f"Win Rate: {wr:.1f}% | Avg EV: {avg_ev:.2f}% | Avg CLV: {avg_clv:.2f}%")
    print(f"{'='*110}")

    # By market
    print(f"\nBY MARKET:")
    for mkt in bets_df["Market"].unique():
        d = bets_df[bets_df["Market"] == mkt]
        s = d["Stake%"].sum()
        p = d["PnL"].sum()
        r = p/s*100 if s>0 else 0
        w = d["Actual"].mean() * 100
        print(f"  {mkt}: n={len(d)} ROI={r:.2f}% WR={w:.1f}% CLV={d['CLV%'].mean():.2f}%")


if __name__ == "__main__":
    import sys
    target_date = sys.argv[1] if len(sys.argv) > 1 else "2025-05-25"
    main(target_date)