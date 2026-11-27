#!/usr/bin/env python
"""
Ensemble Model: LightGBM + Poisson + Elo
Combines three models for robust out-of-sample predictions
"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
FEATURES_FILE = DATA_DIR / "matches_pre_match.parquet"
XG_FILE = DATA_DIR / "matches_with_xg.parquet"
RESULTS_FILE = DATA_DIR / "backtest_ensemble.parquet"
SUMMARY_FILE = DATA_DIR / "backtest_ensemble_summary.md"

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
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
EMBARGO_MONTHS = 1

# Ensemble weights (will be optimized on validation)
ENSEMBLE_WEIGHTS = {
    "lgb": 0.5,
    "poisson": 0.3,
    "elo": 0.2,
}

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

# ──────────────────────────────────────────────
# ELO RATINGS (fast, pre-computed)
# ──────────────────────────────────────────────
INITIAL_ELO = 1500
HOME_ADV_ELO = 65
K_FACTOR = 20

def calculate_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Elo ratings for each team over time (walk-forward)."""
    df = df.sort_values("Date").copy()
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
    elo = {t: INITIAL_ELO for t in teams}

    home_elo_pre = []
    away_elo_pre = []

    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        h_elo, a_elo = elo[h], elo[a]

        home_elo_pre.append(h_elo)
        away_elo_pre.append(a_elo)

        # Expected scores
        h_exp = 1 / (1 + 10 ** ((a_elo - h_elo - HOME_ADV_ELO) / 400))
        a_exp = 1 / (1 + 10 ** ((h_elo + HOME_ADV_ELO - a_elo) / 400))

        # Actual result
        if row["Result"] == 1:
            h_score, a_score = 1.0, 0.0
        elif row["Result"] == 0:
            h_score, a_score = 0.5, 0.5
        else:
            h_score, a_score = 0.0, 1.0

        # Update
        elo[h] += K_FACTOR * (h_score - h_exp)
        elo[a] += K_FACTOR * (a_score - a_exp)

    df["home_elo_pre"] = home_elo_pre
    df["away_elo_pre"] = away_elo_pre
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]

    return df


def elo_probs(elo_diff: float) -> dict:
    """Convert Elo difference to 1X2 probabilities."""
    # Home win prob with home advantage already in elo_diff
    pH = 1 / (1 + 10 ** (-elo_diff / 400))
    pA = 1 / (1 + 10 ** (elo_diff / 400))
    pD = 1 - pH - pA
    # Renormalize
    s = pH + pD + pA
    return {"pH": pH/s, "pD": pD/s, "pA": pA/s}


def elo_over_under(elo_diff: float, total_goals_avg: float = 2.5) -> dict:
    """Rough O/U 2.5 and BTTS from Elo."""
    # Stronger teams -> more goals, weaker -> fewer
    # This is a simple heuristic
    base_over = 0.5
    pO25 = base_over + elo_diff * 0.0001  # Small adjustment
    pO25 = np.clip(pO25, 0.2, 0.8)
    return {"pO25": pO25, "pU25": 1-pO25, "pBTTS": 0.5, "pNBTTS": 0.5}


# ──────────────────────────────────────────────
# POISSON + xG (from model_poisson_xg.py)
# ──────────────────────────────────────────────
HALF_LIFE_DAYS = 150
DIXON_COLES_RHO = 0.13
MAX_GOALS = 8

def time_weights(dates: pd.Series, ref_date: pd.Timestamp, half_life: float = HALF_LIFE_DAYS) -> np.ndarray:
    days_diff = (ref_date - dates).dt.days.clip(lower=0).values
    return np.power(0.5, days_diff / half_life)


def dixon_coles_matrix(lambda_h: float, lambda_a: float, rho: float = DIXON_COLES_RHO, max_g: int = MAX_GOALS) -> np.ndarray:
    x = np.arange(max_g + 1)
    y = np.arange(max_g + 1)
    pmf_x = poisson.pmf(x, lambda_h)
    pmf_y = poisson.pmf(y, lambda_a)
    mat = np.outer(pmf_x, pmf_y)

    if max_g >= 0:
        mat[0, 0] *= (1 - lambda_h * lambda_a * rho)
    if max_g >= 1:
        mat[1, 0] *= (1 + lambda_h * rho)
        mat[0, 1] *= (1 + lambda_a * rho)
        mat[1, 1] *= (1 - rho)

    mat /= mat.sum()
    return mat


def probs_from_matrix(mat: np.ndarray) -> dict:
    max_g = mat.shape[0] - 1
    pH = mat[1:, :].sum() - np.diag(mat[1:, 1:]).sum()
    pD = np.diag(mat).sum()
    pA = mat[:, 1:].sum() - np.diag(mat[1:, 1:]).sum()
    s = pH + pD + pA
    pH, pD, pA = pH/s, pD/s, pA/s

    pO25 = sum(mat[i, j] for i in range(max_g+1) for j in range(max_g+1) if i + j > 2)
    pU25 = 1 - pO25

    pBTTS = 1 - mat[0, :].sum() - mat[:, 0].sum() + mat[0, 0]
    pNBTTS = 1 - pBTTS

    return {"pH": pH, "pD": pD, "pA": pA, "pO25": pO25, "pU25": pU25, "pBTTS": pBTTS, "pNBTTS": pNBTTS}


def estimate_strengths_xg(df_hist: pd.DataFrame, ref_date: pd.Timestamp) -> tuple:
    df = df_hist[df_hist["Date"] < ref_date].copy()
    if len(df) < 10:
        return {}, {}, 1.0, 2.5

    w = time_weights(df["Date"], ref_date)
    df["w"] = w

    total_xg = (df["home_xg_proxy"] * w).sum() + (df["away_xg_proxy"] * w).sum()
    total_w = w.sum() * 2
    league_avg_xg = total_xg / total_w if total_w > 0 else 2.5

    h_xg = (df["home_xg_proxy"] * w).sum()
    a_xg = (df["away_xg_proxy"] * w).sum()
    home_adv = h_xg / a_xg if a_xg > 0 else 1.0

    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    h_idx = np.array([team_to_idx[t] for t in df["HomeTeam"]])
    a_idx = np.array([team_to_idx[t] for t in df["AwayTeam"]])

    df["home_xg_att"] = df["home_xg_roll5"].fillna(df["home_xg_proxy"])
    df["away_xg_att"] = df["away_xg_roll5"].fillna(df["away_xg_proxy"])
    df["home_xga_def"] = df["home_xga_roll5"].fillna(df["home_xga_proxy"])
    df["away_xga_def"] = df["away_xga_roll5"].fillna(df["away_xga_proxy"])

    attack = np.ones(n)
    defense = np.ones(n)

    for _ in range(4):
        num_h = np.bincount(h_idx, weights=df["home_xg_att"] * df["w"], minlength=n)
        den_h = np.bincount(h_idx, weights=defense[a_idx] * home_adv * league_avg_xg * df["w"], minlength=n)
        num_a = np.bincount(a_idx, weights=df["away_xg_att"] * df["w"], minlength=n)
        den_a = np.bincount(a_idx, weights=defense[h_idx] * league_avg_xg * df["w"], minlength=n)

        num = num_h + num_a
        den = den_h + den_a
        mask = den > 0
        attack[mask] = num[mask] / den[mask]

        num_h = np.bincount(h_idx, weights=df["home_xga_def"] * df["w"], minlength=n)
        den_h = np.bincount(h_idx, weights=attack[a_idx] * league_avg_xg * df["w"], minlength=n)
        num_a = np.bincount(a_idx, weights=df["away_xga_def"] * df["w"], minlength=n)
        den_a = np.bincount(a_idx, weights=attack[h_idx] * home_adv * league_avg_xg * df["w"], minlength=n)

        num = num_h + num_a
        den = den_h + den_a
        mask = den > 0
        defense[mask] = num[mask] / den[mask]

    attack = attack / attack.mean()
    defense = defense / defense.mean()

    attack_dict = {t: attack[i] for t, i in team_to_idx.items()}
    defense_dict = {t: defense[i] for t, i in team_to_idx.items()}

    return attack_dict, defense_dict, home_adv, league_avg_xg


def predict_poisson(home: str, away: str, attack: dict, defense: dict, home_adv: float, league_avg: float) -> dict:
    lh = league_avg * attack.get(home, 1.0) * defense.get(away, 1.0) * home_adv
    la = league_avg * attack.get(away, 1.0) * defense.get(home, 1.0)
    lh = np.clip(lh, 0.2, 4.5)
    la = np.clip(la, 0.2, 4.5)
    mat = dixon_coles_matrix(lh, la)
    probs = probs_from_matrix(mat)
    probs["lambda_h"] = lh
    probs["lambda_a"] = la
    return probs


# ──────────────────────────────────────────────
# LIGHTGBM TRAINING (from model_ml.py)
# ──────────────────────────────────────────────
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


def get_walkforward_splits(df: pd.DataFrame, n_splits: int = 8, embargo_months: int = EMBARGO_MONTHS):
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

        splits.append((train_mask, test_mask, train_months, test_months))

    return splits


def train_lgb(X_train, y_train, X_val, y_val, cat_cols, task_type, params):
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    lgb_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=lgb_train)

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
        cal_probs = np.column_stack([
            calibrators[i].predict(probs[:, i]) for i in range(probs.shape[1])
        ])
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
    for train_mask, test_mask, _, _ in splits:
        X_train = df.loc[train_mask, feat_cols]
        y_train = df.loc[train_mask, target_col]
        X_val = df.loc[test_mask, feat_cols]
        y_val = df.loc[test_mask, target_col]

        try:
            model = train_lgb(X_train, y_train, X_val, y_val, cat_cols, task_type, params)
            val_probs = model.predict(X_val, num_iteration=model.best_iteration)
            score = log_loss(y_val, val_probs)
            scores.append(score)
        except Exception:
            return 1.0

    return np.mean(scores)


# ──────────────────────────────────────────────
# ENSEMBLE BACKTEST
# ──────────────────────────────────────────────
def run_ensemble_backtest(df: pd.DataFrame, feat_cols: list, cat_cols: list) -> pd.DataFrame:
    """Walk-forward ensemble backtest with LGB + Poisson + Elo."""
    df = df.sort_values("Date").reset_index(drop=True)

    # Load xG data for Poisson model
    xg_df = pd.read_parquet(XG_FILE)
    xg_cols = [c for c in xg_df.columns if c not in df.columns]
    xg_cols = ["Date", "HomeTeam", "AwayTeam"] + xg_cols
    df = df.merge(xg_df[xg_cols], on=["Date", "HomeTeam", "AwayTeam"], how="left")

    # Pre-compute Elo
    df = calculate_elo(df)

    splits = get_walkforward_splits(df, n_splits=8)
    print(f"\n[SPLITS] {len(splits)} walk-forward folds")

    # Collect OOS predictions from each model
    all_oos = {
        "lgb": [],
        "poisson": [],
        "elo": [],
    }

    # --- LGB predictions ---
    print("\n[1/3] Training LightGBM models...")
    for target_name, target_info in TARGETS.items():
        print(f"  Target: {target_name} ({target_info['type']})")
        target_col = f"target_{target_name}"

        early_mask = df["Date"] < df["Date"].quantile(0.3)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda t: optuna_objective(t, df[early_mask], feat_cols, cat_cols, target_name, target_info["type"]),
            n_trials=20, timeout=60, show_progress_bar=False
        )
        best_params = study.best_params
        print(f"    Best params: {best_params}")

        fold_test_indices = []
        fold_probs = []

        for fold, (train_mask, test_mask, train_m, test_m) in enumerate(splits):
            X_train = df.loc[train_mask, feat_cols]
            y_train = df.loc[train_mask, target_col]
            X_test = df.loc[test_mask, feat_cols]
            y_test = df.loc[test_mask, target_col]

            model = train_lgb(X_train, y_train, X_test, y_test, cat_cols, target_info["type"], best_params)
            cal = calibrate_probabilities(model, X_test, y_test, target_info["type"])

            test_probs = predict_calibrated(model, X_test, cal, target_info["type"])

            fold_test_indices.append(df.index[test_mask].values)
            fold_probs.append(test_probs)
            print(f"    Fold {fold+1}: train={train_mask.sum():,} test={test_mask.sum():,}")

        all_oos["lgb"].append({
            "target": target_name,
            "type": target_info["type"],
            "fold_indices": fold_test_indices,
            "fold_probs": fold_probs,
        })

    # --- Poisson predictions ---
    print("\n[2/3] Generating Poisson predictions...")
    for fold, (train_mask, test_mask, train_m, test_m) in enumerate(splits):
        train_df = df[train_mask]
        test_df = df[test_mask]

        ref_date = test_df["Date"].min()
        attack, defense, home_adv, league_avg = estimate_strengths_xg(train_df, ref_date)

        if not attack:
            continue

        fold_test_indices = []
        fold_probs = []

        for _, row in test_df.iterrows():
            probs = predict_poisson(row["HomeTeam"], row["AwayTeam"], attack, defense, home_adv, league_avg)
            fold_test_indices.append(row.name)
            # Store as [pH, pD, pA, pO25, pU25, pBTTS, pNBTTS] for easy indexing
            probs_arr = np.array([probs["pH"], probs["pD"], probs["pA"],
                                  probs["pO25"], probs["pU25"], probs["pBTTS"], probs["pNBTTS"]])
            fold_probs.append(probs_arr)

        all_oos["poisson"].append({
            "fold_indices": [fold_test_indices],
            "fold_probs": [np.array(fold_probs)],
        })

    # --- Elo predictions ---
    print("\n[3/3] Generating Elo predictions...")
    for fold, (train_mask, test_mask, train_m, test_m) in enumerate(splits):
        test_df = df[test_mask]

        fold_test_indices = []
        fold_probs = []

        for _, row in test_df.iterrows():
            elo_diff = row["elo_diff"]
            probs_1x2 = elo_probs(elo_diff)
            probs_ou = elo_over_under(elo_diff)

            probs_arr = np.array([probs_1x2["pH"], probs_1x2["pD"], probs_1x2["pA"],
                                  probs_ou["pO25"], probs_ou["pU25"],
                                  probs_ou["pBTTS"], probs_ou["pNBTTS"]])
            fold_test_indices.append(row.name)
            fold_probs.append(probs_arr)

        all_oos["elo"].append({
            "fold_indices": [fold_test_indices],
            "fold_probs": [np.array(fold_probs)],
        })

    # --- Combine ensemble predictions ---
    print("\n[ENSEMBLE] Combining predictions...")
    pred_lookup = {}

    # Process each target
    for target_name, target_info in TARGETS.items():
        ttype = target_info["type"]

        # LGB predictions for this target
        lgb_op = next(op for op in all_oos["lgb"] if op["target"] == target_name)

        # Poisson and Elo use same indexing across all targets
        for fold_idx in range(len(splits)):
            test_indices = all_oos["poisson"][fold_idx]["fold_indices"][0]
            poisson_probs = all_oos["poisson"][fold_idx]["fold_probs"][0]
            elo_probs_fold = all_oos["elo"][fold_idx]["fold_probs"][0]

            # LGB fold data
            lgb_indices = lgb_op["fold_indices"][fold_idx]
            lgb_probs = lgb_op["fold_probs"][fold_idx]

            # Match indices (should be same order)
            for i, idx in enumerate(test_indices):
                if idx not in pred_lookup:
                    pred_lookup[idx] = {}

                # Get probabilities from each model
                if ttype == "multiclass":
                    # Result: 1X2
                    lgb_p = lgb_probs[i]  # [pA, pD, pH]
                    pois_p = poisson_probs[i, :3]  # [pH, pD, pA]
                    elo_p = elo_probs_fold[i, :3]  # [pH, pD, pA]

                    # Reorder LGB to [pH, pD, pA] for averaging
                    lgb_p_reordered = np.array([lgb_p[2], lgb_p[1], lgb_p[0]])

                    # Weighted ensemble
                    ens_p = (ENSEMBLE_WEIGHTS["lgb"] * lgb_p_reordered +
                            ENSEMBLE_WEIGHTS["poisson"] * pois_p +
                            ENSEMBLE_WEIGHTS["elo"] * elo_p)
                    ens_p = ens_p / ens_p.sum()

                    pred_lookup[idx][f"pred_{target_name}_H"] = ens_p[0]
                    pred_lookup[idx][f"pred_{target_name}_D"] = ens_p[1]
                    pred_lookup[idx][f"pred_{target_name}_A"] = ens_p[2]
                else:
                    # Binary: over_25 or btts
                    if target_name == "over_25":
                        lgb_p = lgb_probs[i]  # scalar p(over)
                        pois_p = poisson_probs[i, 3]  # pO25
                        elo_p = elo_probs_fold[i, 3]  # pO25
                    else:  # btts
                        lgb_p = lgb_probs[i]
                        pois_p = poisson_probs[i, 5]  # pBTTS
                        elo_p = elo_probs_fold[i, 5]  # pBTTS

                    ens_p = (ENSEMBLE_WEIGHTS["lgb"] * lgb_p +
                            ENSEMBLE_WEIGHTS["poisson"] * pois_p +
                            ENSEMBLE_WEIGHTS["elo"] * elo_p)
                    ens_p = np.clip(ens_p, 0.01, 0.99)

                    pred_lookup[idx][f"pred_{target_name}_1"] = ens_p
                    pred_lookup[idx][f"pred_{target_name}_0"] = 1 - ens_p

    # --- Evaluate bets on OOS predictions ---
    print("\n[BETS] Evaluating betting opportunities on OOS ensemble predictions...")
    all_rows = []

    for idx, row in df.iterrows():
        if idx not in pred_lookup:
            continue

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
        f"# Backtest Ensemble (LGB + Poisson + Elo)",
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


def main():
    print("=" * 60)
    print("ENSEMBLE MODEL: LGB + POISSON + ELO")
    print("=" * 60)

    df = pd.read_parquet(FEATURES_FILE)
    print(f"\n[LOAD] {len(df):,} matches, {len(df.columns)} features")

    df, feat_cols, cat_cols = prepare_data(df)
    print(f"[FEATURES] {len(feat_cols)} features ({len(cat_cols)} categorical)")

    results = run_ensemble_backtest(df, feat_cols, cat_cols)

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