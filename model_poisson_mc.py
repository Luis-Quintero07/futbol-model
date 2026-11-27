#!/usr/bin/env python
"""
Poisson + Monte Carlo (10k+) as PRIMARY prediction engine.

Pipeline:
1. LightGBM (160 features) → calibrated probabilities (pH, pD, pA, pO25, pBTTS)
2. Invert probabilities → Poisson lambdas (lambda_h, lambda_a) via optimization
3. Run 10,000+ Monte Carlo sims with Dixon-Coles adjustment
4. Empirical MC probabilities → EV calculation + Kelly staking
5. Triple-source consensus (Model MC + Bet365 + Polymarket) with KL divergence

This replaces direct LightGBM prediction with Monte Carlo empirical probs.
"""

import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
FEATURES_FILE = DATA_DIR / "matches_pre_match.parquet"

RHO = 0.13  # Dixon-Coles
MAX_GOALS = 10
MC_SIMS = 10000  # Primary iteration count
MC_SIMS_HIGH = 50000  # For final validation

MIN_EV = 0.05
MAX_ODDS = 5.0
MIN_ODDS = 1.5
KELLY_FRAC = 0.25

TARGETS = {
    "result": {"type": "multiclass", "classes": [-1, 0, 1]},
    "over_25": {"type": "binary", "threshold": 2.5},
    "btts": {"type": "binary", "threshold": None},
}

MARKETS = {
    "1X2_H": {"target": "result", "class_idx": 2, "odds_key": "home"},
    "1X2_D": {"target": "result", "class_idx": 1, "odds_key": "draw"},
    "1X2_A": {"target": "result", "class_idx": 0, "odds_key": "away"},
    "OVER_2.5": {"target": "over_25", "class_idx": 1, "odds_key": "over_2_5"},
    "UNDER_2.5": {"target": "over_25", "class_idx": 0, "odds_key": "under_2_5"},
    "BTTS_YES": {"target": "btts", "class_idx": 1, "odds_key": "btts_yes"},
    "BTTS_NO": {"target": "btts", "class_idx": 0, "odds_key": "btts_no"},
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
}

COL_TZ = timezone(timedelta(hours=-5))


# ──────────────────────────────────────────────
# DIXON-COLES + MONTE CARLO CORE
# ──────────────────────────────────────────────
def dixon_coles_prob(h_goals: int, a_goals: int, lambda_h: float, lambda_a: float, rho: float = RHO) -> float:
    """Dixon-Coles adjusted probability for (h_goals, a_goals)"""
    p = poisson.pmf(h_goals, lambda_h) * poisson.pmf(a_goals, lambda_a)

    if h_goals == 0 and a_goals == 0:
        tau = 1 - lambda_h * lambda_a * rho
    elif h_goals == 1 and a_goals == 0:
        tau = 1 + lambda_a * rho
    elif h_goals == 0 and a_goals == 1:
        tau = 1 + lambda_h * rho
    elif h_goals == 1 and a_goals == 1:
        tau = 1 - rho
    else:
        tau = 1.0

    return p * tau


def match_probs_from_lambdas(lambda_h: float, lambda_a: float, max_goals: int = MAX_GOALS) -> dict:
    """Compute exact DC probabilities from lambdas (no simulation)"""
    h_win = d_win = a_win = over25 = btts = 0.0

    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = dixon_coles_prob(hg, ag, lambda_h, lambda_a)
            if hg > ag:
                h_win += p
            elif hg == ag:
                d_win += p
            else:
                a_win += p

            if hg + ag > 2.5:
                over25 += p
            if hg > 0 and ag > 0:
                btts += p

    total = h_win + d_win + a_win
    if total > 0:
        h_win, d_win, a_win = h_win/total, d_win/total, a_win/total

    return {
        'H': h_win, 'D': d_win, 'A': a_win,
        'over_25': over25, 'under_25': 1 - over25,
        'btts_yes': btts, 'btts_no': 1 - btts
    }


def objective_lambdas(params, target_probs):
    """Objective: minimize squared error between DC probs and target ML probs"""
    lambda_h, lambda_a = params
    if lambda_h <= 0 or lambda_a <= 0:
        return 1e6
    dc_probs = match_probs_from_lambdas(lambda_h, lambda_a)
    error = 0.0
    for key in ['H', 'D', 'A', 'over_25', 'btts_yes']:
        target = target_probs.get(key, 0)
        if target > 0:
            error += (dc_probs[key] - target) ** 2
    return error


def estimate_lambdas(ml_probs: dict, init_guess=(1.5, 1.0)) -> tuple:
    """
    Estimate Poisson lambdas that match ML model probabilities.
    Returns (lambda_h, lambda_a, fit_error)
    """
    target = {
        'H': ml_probs['pH'],
        'D': ml_probs['pD'],
        'A': ml_probs['pA'],
        'over_25': ml_probs['pO25'],
        'btts_yes': ml_probs['pBTTS']
    }

    # Coarse grid search (5x5 = 25 evaluations vs 1254)
    best_error = float('inf')
    best_params = init_guess

    for lh in np.linspace(0.3, 4.0, 5):
        for la in np.linspace(0.3, 3.5, 5):
            error = objective_lambdas((lh, la), target)
            if error < best_error:
                best_error = error
                best_params = (lh, la)

    # Refine with 2D bounded optimization
    from scipy.optimize import minimize
    result = minimize(
        lambda params: objective_lambdas(params, target),
        x0=best_params,
        bounds=[(0.2, 4.5), (0.2, 4.5)],
        method='L-BFGS-B'
    )
    best_lh, best_la = result.x

    final_error = objective_lambdas((best_lh, best_la), target)
    return best_lh, best_la, final_error


def simulate_match(lambda_h: float, lambda_a: float, n_sims: int = MC_SIMS, max_goals: int = MAX_GOALS) -> dict:
    """Run Monte Carlo simulation using DC-adjusted Poisson"""
    # Pre-compute DC probabilities for all score combinations
    score_probs = {}
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            score_probs[(hg, ag)] = dixon_coles_prob(hg, ag, lambda_h, lambda_a)

    # Normalize
    total = sum(score_probs.values())
    score_probs = {k: v/total for k, v in score_probs.items()}

    scores = list(score_probs.keys())
    probs = list(score_probs.values())

    # Sample
    samples = np.random.choice(len(scores), size=n_sims, p=probs)
    home_goals = np.array([scores[s][0] for s in samples])
    away_goals = np.array([scores[s][1] for s in samples])

    total_goals = home_goals + away_goals
    results = np.where(home_goals > away_goals, 'H', np.where(home_goals == away_goals, 'D', 'A'))
    over25 = (total_goals > 2.5).astype(int)
    btts = ((home_goals > 0) & (away_goals > 0)).astype(int)

    return {
        'home_goals': home_goals,
        'away_goals': away_goals,
        'total_goals': total_goals,
        'results': results,
        'over25': over25,
        'btts': btts,
        'score_probs': score_probs
    }


def empirical_probs_from_sim(sim: dict) -> dict:
    """Extract empirical probabilities from Monte Carlo results"""
    return {
        'H': (sim['results'] == 'H').mean(),
        'D': (sim['results'] == 'D').mean(),
        'A': (sim['results'] == 'A').mean(),
        'over_25': sim['over25'].mean(),
        'under_25': 1 - sim['over25'].mean(),
        'btts_yes': sim['btts'].mean(),
        'btts_no': 1 - sim['btts'].mean()
    }


def top_scorelines(sim: dict, top_n: int = 10) -> list:
    """Get top N scorelines with percentages"""
    score_counts = pd.Series(list(zip(sim['home_goals'], sim['away_goals']))).value_counts()
    top = score_counts.head(top_n)
    return [((hg, ag), count / MC_SIMS * 100) for (hg, ag), count in top.items()]


# ──────────────────────────────────────────────
# KL DIVERGENCE + CONSENSUS (Triple-Source)
# ──────────────────────────────────────────────
def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0, ln2])"""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))


def three_way_js(p_model: np.ndarray, p_bet365: np.ndarray, p_poly: np.ndarray) -> float:
    """Average pairwise JS divergence between three sources"""
    return (js_divergence(p_model, p_bet365) +
            js_divergence(p_model, p_poly) +
            js_divergence(p_bet365, p_poly)) / 3


def consensus_2_vs_1(p_model: np.ndarray, p_bet365: np.ndarray, p_poly: np.ndarray) -> tuple:
    """
    Returns (majority_class, confidence) or (None, 0) if 3-way split.
    For 1X2: 0=A, 1=D, 2=H. For binary: 0=No/Under, 1=Yes/Over.
    """
    preds = [int(np.argmax(p_model)), int(np.argmax(p_bet365)), int(np.argmax(p_poly))]
    from collections import Counter
    votes = Counter(preds)
    majority_class, count = votes.most_common(1)[0]

    if count >= 2:
        return majority_class, count / 3.0
    return None, 0.0


# ──────────────────────────────────────────────
# LIGHTGBM LOADING + PREDICTION
# ──────────────────────────────────────────────
def load_models_and_calibrators():
    models = {}
    calibrators = {}
    for target_name in TARGETS:
        model_path = MODEL_DIR / f"{target_name}_model.txt"
        cal_path = MODEL_DIR / f"{target_name}_calibrator.pkl"
        if model_path.exists():
            models[target_name] = lgb.Booster(model_file=str(model_path))
            with open(cal_path, "rb") as f:
                calibrators[target_name] = joblib.load(f)
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    return models, calibrators


def prepare_data(df: pd.DataFrame) -> tuple:
    df = df.sort_values("Date").reset_index(drop=True)
    df["target_result"] = df["Result"].map({-1: 0, 0: 1, 1: 2}) if "Result" in df.columns else 0
    df["target_over_25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int) if "FTHG" in df.columns else 0
    df["target_btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int) if "FTHG" in df.columns else 0

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


def get_lgbm_probs(live_df: pd.DataFrame, models: dict, calibrators: dict, feat_cols: list, cat_cols: list) -> pd.DataFrame:
    """Get calibrated LightGBM probabilities for all targets"""
    X_live = live_df[feat_cols].copy()
    for c in cat_cols:
        if c in X_live.columns:
            X_live[c] = X_live[c].astype("category")

    for target_name, target_info in TARGETS.items():
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

    return live_df


# ──────────────────────────────────────────────
# MAIN POISSON + MC PIPELINE
# ──────────────────────────────────────────────
def run_poisson_mc_for_fixture(row: pd.Series, n_sims: int = MC_SIMS, use_dc_exact: bool = False) -> dict:
    """
    Full Poisson + Monte Carlo pipeline for a single fixture.
    Input: row with LightGBM probabilities + odds
    Output: dict with lambdas, DC probs, empirical MC probs, EV bets

    If use_dc_exact=True: uses DC exact probabilities (instant, for backtesting)
    If use_dc_exact=False: runs Monte Carlo simulation (for live predictions with scorelines)
    """
    # 1. Extract LightGBM probabilities
    ml_probs = {
        'pH': row.get('pred_result_H', 0),
        'pD': row.get('pred_result_D', 0),
        'pA': row.get('pred_result_A', 0),
        'pO25': row.get('pred_over_25_1', 0),
        'pBTTS': row.get('pred_btts_1', 0)
    }

    if min(ml_probs.values()) <= 0:
        return {"error": "Missing ML probabilities"}

    # 2. Estimate lambdas from ML probs
    lambda_h, lambda_a, fit_error = estimate_lambdas(ml_probs)

    # 3. DC exact probabilities at estimated lambdas
    dc_probs = match_probs_from_lambdas(lambda_h, lambda_a)

    # 4. Monte Carlo simulation (or use DC exact for backtest)
    if use_dc_exact:
        emp_probs = dc_probs
        top_scores = []
        # Convert DC probs to top scorelines format
        score_probs = {}
        for hg in range(MAX_GOALS + 1):
            for ag in range(MAX_GOALS + 1):
                score_probs[(hg, ag)] = dixon_coles_prob(hg, ag, lambda_h, lambda_a)
        total = sum(score_probs.values())
        score_probs = {k: v/total for k, v in score_probs.items()}
        top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        top_scores = [((hg, ag), p*100) for (hg, ag), p in top_scores]
    else:
        sim = simulate_match(lambda_h, lambda_a, n_sims)
        emp_probs = empirical_probs_from_sim(sim)
        top_scores = top_scorelines(sim)

    # 5. Odds extraction (map B365 columns to expected keys)
    odds = {
        'home': row.get('B365H'),
        'draw': row.get('B365D'),
        'away': row.get('B365A'),
        'over_2_5': row.get('B365>2.5'),
        'under_2_5': row.get('B365<2.5'),
        'btts_yes': row.get('btts_yes'),
        'btts_no': row.get('btts_no')
    }

    # Estimate missing odds from empirical probs
    if pd.isna(odds.get('over_2_5')) or odds.get('over_2_5') is None:
        margin = 0.05
        odds['over_2_5'] = (1 / emp_probs['over_25']) * (1 + margin)
        odds['under_2_5'] = (1 / emp_probs['under_25']) * (1 + margin)

    if pd.isna(odds.get('btts_yes')) or odds.get('btts_yes') is None:
        margin = 0.05
        odds['btts_yes'] = (1 / emp_probs['btts_yes']) * (1 + margin)
        odds['btts_no'] = (1 / emp_probs['btts_no']) * (1 + margin)

    # Fill BTTS if missing (not in historical data)
    if pd.isna(odds.get('btts_yes')) or odds.get('btts_yes') is None:
        margin = 0.05
        odds['btts_yes'] = (1 / emp_probs['btts_yes']) * (1 + margin)
        odds['btts_no'] = (1 / emp_probs['btts_no']) * (1 + margin)

    # 6. Polymarket odds (placeholder)
    poly_odds = {}  # fetch_polymarket_odds(row['Fixture_ID'], row['HomeTeam'], row['AwayTeam'], row['Date'])

    # 7. EV calculation with empirical MC probs
    ev_bets = []
    markets = {
        '1X2_H': ('H', emp_probs['H'], odds.get('home'), ml_probs['pH'], dc_probs['H']),
        '1X2_D': ('D', emp_probs['D'], odds.get('draw'), ml_probs['pD'], dc_probs['D']),
        '1X2_A': ('A', emp_probs['A'], odds.get('away'), ml_probs['pA'], dc_probs['A']),
        'OVER_2.5': ('Over 2.5', emp_probs['over_25'], odds.get('over_2_5'), ml_probs['pO25'], dc_probs['over_25']),
        'UNDER_2.5': ('Under 2.5', emp_probs['under_25'], odds.get('under_2_5'), ml_probs['pU25'] if 'pU25' in ml_probs else 1-ml_probs['pO25'], dc_probs['under_25']),
        'BTTS_YES': ('BTTS Yes', emp_probs['btts_yes'], odds.get('btts_yes'), ml_probs['pBTTS'], dc_probs['btts_yes']),
        'BTTS_NO': ('BTTS No', emp_probs['btts_no'], odds.get('btts_no'), 1-ml_probs['pBTTS'], dc_probs['btts_no'])
    }

    for mkt_name, (label, p_mc, odd, p_lgbm, p_dc) in markets.items():
        if odd is None or pd.isna(odd) or odd < MIN_ODDS or odd > MAX_ODDS:
            continue

        # Triple-source probabilities
        p_model_vec = np.array([p_mc]) if 'Over' in label or 'BTTS' in label else np.array([emp_probs['A'], emp_probs['D'], emp_probs['H']])
        p_bet365_vec = odds_to_probs_vec(odd, label, odds)
        p_poly_vec = odds_to_probs_vec(0, label, poly_odds) if poly_odds else np.array([])

        # KL divergence & consensus (if Polymarket available)
        kl_score = 0.0
        consensus = None
        if len(p_poly_vec) > 0:
            kl_score = three_way_js(p_model_vec, p_bet365_vec, p_poly_vec)
            consensus, _ = consensus_2_vs_1(p_model_vec, p_bet365_vec, p_poly_vec)
            # Skip if all 3 agree (no edge)
            model_pred = int(np.argmax(p_model_vec))
            poly_pred = int(np.argmax(p_poly_vec))
            bet365_pred = int(np.argmax(p_bet365_vec))
            if model_pred == poly_pred == bet365_pred:
                continue
            if kl_score < 0.02:
                continue
            if consensus is None or consensus != model_pred - 1:
                continue

        # EV using empirical Monte Carlo probability
        ev = p_mc * odd - 1
        if ev <= MIN_EV:
            continue

        kelly = (p_mc * odd - 1) / (odd - 1) if odd > 1 else 0
        stake = max(0, kelly * KELLY_FRAC)
        clv = (p_mc - 1/odd) * 100

        ev_bets.append({
            'Market': mkt_name,
            'Label': label,
            'MC_Prob': round(p_mc, 4),
            'LGBM_Prob': round(p_lgbm, 4),
            'DC_Prob': round(p_dc, 4),
            'Odds': round(odd, 2),
            'EV%': round(ev * 100, 2),
            'Kelly%': round(kelly * 100, 2),
            'Stake%': round(stake * 100, 2),
            'CLV%': round(clv, 2),
            'KL_Score': round(kl_score, 4),
            'Consensus': "2-vs-1" if consensus is not None else "N/A"
        })

    return {
        'lambda_home': round(lambda_h, 3),
        'lambda_away': round(lambda_a, 3),
        'fit_error': round(fit_error, 6),
        'dc_probs': {k: round(v, 4) for k, v in dc_probs.items()},
        'mc_probs': {k: round(v, 4) for k, v in emp_probs.items()},
        'top_scorelines': top_scores,
        'ev_bets': pd.DataFrame(ev_bets) if ev_bets else pd.DataFrame(),
        'odds_used': odds
    }


def odds_to_probs_vec(odds_dict: dict, market_label: str, all_odds: dict) -> np.ndarray:
    """Convert odds to probability vector for a market"""
    if '1X2' in market_label:
        h, d, a = all_odds.get('home'), all_odds.get('draw'), all_odds.get('away')
        if h and d and a:
            inv = np.array([1/a, 1/d, 1/h])  # [A, D, H]
            return inv / inv.sum()
    elif 'Over' in market_label or 'Under' in market_label:
        o, u = all_odds.get('over_2_5'), all_odds.get('under_2_5')
        if o and u:
            inv = np.array([1/u, 1/o])  # [Under, Over]
            return inv / inv.sum()
    elif 'BTTS' in market_label:
        y, n = all_odds.get('btts_yes'), all_odds.get('btts_no')
        if y and n:
            inv = np.array([1/n, 1/y])  # [No, Yes]
            return inv / inv.sum()
    return np.array([])


def run_poisson_mc_batch(live_df: pd.DataFrame, n_sims: int = MC_SIMS) -> tuple:
    """Run Poisson + MC for all fixtures in live_df"""
    print(f"\n[POISSON+MC] Running {n_sims:,} simulations per fixture for {len(live_df)} matches...")

    all_results = []
    all_bets = []

    for idx, row in live_df.iterrows():
        fixture_name = f"{row['HomeTeam']} vs {row['AwayTeam']} ({row['Div']})"
        print(f"  [{idx+1}/{len(live_df)}] {fixture_name}")

        result = run_poisson_mc_for_fixture(row, n_sims)
        result['fixture'] = fixture_name
        result['date'] = row['Date']
        result['league'] = row['Div']
        result['home'] = row['HomeTeam']
        result['away'] = row['AwayTeam']
        result['fixture_id'] = row.get('Fixture_ID', idx)

        all_results.append(result)

        if len(result['ev_bets']) > 0:
            result['ev_bets']['Fixture'] = fixture_name
            result['ev_bets']['Date'] = row['Date']
            result['ev_bets']['League'] = row['Div']
            result['ev_bets']['Home'] = row['HomeTeam']
            result['ev_bets']['Away'] = row['AwayTeam']
            all_bets.append(result['ev_bets'])

    if all_bets:
        combined_bets = pd.concat(all_bets, ignore_index=True)
    else:
        combined_bets = pd.DataFrame()

    return all_results, combined_bets


def apply_production_filters(bets_df: pd.DataFrame, kelly_frac: float = 0.10, max_stake_pct: float = 25.0, max_bets_per_fixture: int = 1, max_ev: float = 50.0) -> pd.DataFrame:
    """
    Apply production betting filters:
    - Min EV thresholds by market (1X2: 10%, O/U: 7%, BTTS: 7%)
    - Max EV cap (default 50%) to flag odds anomalies
    - Odds validation (odds <= 4.0, implied prob >= 10%)
    - Max bets per fixture (highest EV only)
    - Recalculate stakes with specified Kelly fraction
    - Total stake cap
    """
    if len(bets_df) == 0:
        return bets_df

    bets_df = bets_df.copy()

    # 1. Min EV thresholds by market
    min_ev = {'1X2_H': 10, '1X2_D': 10, '1X2_A': 10, 'OVER_2.5': 7, 'UNDER_2.5': 7, 'BTTS_YES': 7, 'BTTS_NO': 7}
    bets_df['min_ev_req'] = bets_df['Market'].map(min_ev)
    bets_df = bets_df[bets_df['EV%'] >= bets_df['min_ev_req']]
    print(f"  [FILTER] Min EV: {len(bets_df)} bets remain")

    if len(bets_df) == 0:
        return bets_df

    # 2. Max EV cap - flag odds anomalies (e.g., Como vs Juventus 109% EV)
    bets_df = bets_df[bets_df['EV%'] <= max_ev]
    print(f"  [FILTER] Max EV {max_ev}%: {len(bets_df)} bets remain")

    if len(bets_df) == 0:
        return bets_df

    # 3. Odds validation - reject extreme odds (likely data errors)
    bets_df = bets_df[(bets_df['Odds'] <= 4.0) & (bets_df['MC_Prob'] >= 0.10)]
    print(f"  [FILTER] Odds validation: {len(bets_df)} bets remain")

    if len(bets_df) == 0:
        return bets_df

    # 4. Remove contradictory bets (both Over and Under on same match, both BTTS Yes/No)
    # Keep only highest EV per market pair per fixture
    bets_df = bets_df.sort_values('EV%', ascending=False)
    bets_df['market_pair'] = bets_df['Market'].replace({
        'OVER_2.5': 'O/U', 'UNDER_2.5': 'O/U',
        'BTTS_YES': 'BTTS', 'BTTS_NO': 'BTTS',
        '1X2_H': '1X2', '1X2_D': '1X2', '1X2_A': '1X2'
    })
    bets_df['pair_rank'] = bets_df.groupby(['Fixture', 'market_pair']).cumcount() + 1
    bets_df = bets_df[bets_df['pair_rank'] == 1]  # Keep highest EV per market pair
    print(f"  [FILTER] Contradiction removal: {len(bets_df)} bets remain")

    # 5. Max bets per fixture (highest EV overall)
    bets_df = bets_df.sort_values('EV%', ascending=False)
    bets_df['fixture_rank'] = bets_df.groupby('Fixture').cumcount() + 1
    bets_df = bets_df[bets_df['fixture_rank'] <= max_bets_per_fixture]
    print(f"  [FILTER] Max {max_bets_per_fixture} per fixture: {len(bets_df)} bets remain")

    if len(bets_df) == 0:
        return bets_df

    # 6. Recalculate stakes with specified Kelly fraction
    bets_df['Stake%'] = bets_df.apply(
        lambda r: max(0, (r['MC_Prob'] * r['Odds'] - 1) / (r['Odds'] - 1) * kelly_frac * 100) if r['Odds'] > 1 else 0, axis=1
    )

    # 7. Total stake cap
    total_stake = bets_df['Stake%'].sum()
    if total_stake > max_stake_pct:
        bets_df = bets_df.sort_values('EV%', ascending=False)
        bets_df['cum_stake'] = bets_df['Stake%'].cumsum()
        bets_df = bets_df[bets_df['cum_stake'] <= max_stake_pct]
        print(f"  [FILTER] Stake cap {max_stake_pct}%: {len(bets_df)} bets remain, stake={bets_df['Stake%'].sum():.2f}%")

    return bets_df


# ──────────────────────────────────────────────
# BACKTEST VALIDATION
# ──────────────────────────────────────────────
def backtest_poisson_mc(n_splits: int = 6, n_sims: int = 5000, use_pretrained: bool = True) -> pd.DataFrame:
    """
    Walk-forward backtest of Poisson + MC vs pure LightGBM

    If use_pretrained=True: uses pre-trained models from models/ (fast, no retraining)
    If use_pretrained=False: retrains models per fold (slow, proper walk-forward)
    """
    mode = "PRE-TRAINED" if use_pretrained else "RETRAIN"
    print(f"\n[BACKTEST] Poisson+MC vs LightGBM - {n_splits} folds, {n_sims} sims [{mode}]")

    df = pd.read_parquet(FEATURES_FILE)
    df, feat_cols, cat_cols = prepare_data(df)

    if use_pretrained:
        models, calibrators = load_models_and_calibrators()
        # Get probs for all data once
        df = get_lgbm_probs(df, models, calibrators, feat_cols, cat_cols)

    # Walk-forward splits
    df["YearMonth"] = df["Date"].dt.to_period("M")
    months = sorted(df["YearMonth"].unique())

    min_train_months = 12
    test_months_per_split = max(1, (len(months) - min_train_months) // n_splits)

    all_bets = []

    for fold in range(n_splits):
        test_start_idx = min_train_months + fold * test_months_per_split
        test_end_idx = test_start_idx + test_months_per_split
        if test_end_idx > len(months):
            break

        train_end_idx = test_start_idx - 1
        train_months = months[:train_end_idx]
        test_months = months[test_start_idx:test_end_idx]

        train_mask = df["YearMonth"].isin(train_months)
        test_mask = df["YearMonth"].isin(test_months)

        if train_mask.sum() < 1000 or test_mask.sum() < 50:
            continue

        print(f"  Fold {fold+1}: train={train_mask.sum():,} test={test_mask.sum():,}")

        if not use_pretrained:
            # Retrain models on train fold
            fold_models = {}
            fold_calibrators = {}
            for target_name, target_info in TARGETS.items():
                target_col = f"target_{target_name}"
                X_train = df.loc[train_mask, feat_cols]
                y_train = df.loc[train_mask, target_col]
                X_val = df.loc[test_mask, feat_cols]
                y_val = df.loc[test_mask, target_col]

                lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
                lgb_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=lgb_train)

                model = lgb.train({
                    "objective": "multiclass" if target_info["type"] == "multiclass" else "binary",
                    "metric": "multi_logloss" if target_info["type"] == "multiclass" else "binary_logloss",
                    "num_class": 3 if target_info["type"] == "multiclass" else 1,
                    "learning_rate": 0.03, "num_leaves": 63, "verbosity": -1, "seed": 42,
                }, lgb_train, valid_sets=[lgb_val], num_boost_round=2000,
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

                # Calibrate
                val_probs = model.predict(X_val, num_iteration=model.best_iteration)
                if target_info["type"] == "multiclass":
                    cals = []
                    for i in range(val_probs.shape[1]):
                        iso = __import__('sklearn.isotonic', fromlist=['IsotonicRegression']).IsotonicRegression(out_of_bounds="clip")
                        iso.fit(val_probs[:, i], (y_val == i).astype(int))
                        cals.append(iso)
                else:
                    iso = __import__('sklearn.isotonic', fromlist=['IsotonicRegression']).IsotonicRegression(out_of_bounds="clip")
                    iso.fit(val_probs, y_val)
                    cals = iso

                fold_models[target_name] = model
                fold_calibrators[target_name] = cals

            # Predict on test set
            test_df = df.loc[test_mask].copy()
            test_df = get_lgbm_probs(test_df, fold_models, fold_calibrators, feat_cols, cat_cols)
        else:
            test_df = df.loc[test_mask].copy()

        # Run Poisson+MC for each test match (use DC exact for speed)
        for _, row in test_df.iterrows():
            result = run_poisson_mc_for_fixture(row, n_sims, use_dc_exact=True)

            if 'ev_bets' not in result or len(result['ev_bets']) == 0:
                continue

            for _, bet in result['ev_bets'].iterrows():
                # Get actual result
                market = bet['Market']
                if market.startswith('1X2'):
                    actual_result = row['Result']
                    pred = {'1X2_H': 1, '1X2_D': 0, '1X2_A': -1}[market]
                    won = int(actual_result == pred)
                elif market in ['OVER_2.5', 'UNDER_2.5']:
                    total = row['FTHG'] + row['FTAG']
                    won = int((market == 'OVER_2.5' and total > 2.5) or (market == 'UNDER_2.5' and total < 2.5))
                elif market in ['BTTS_YES', 'BTTS_NO']:
                    btts = int(row['FTHG'] > 0 and row['FTAG'] > 0)
                    won = int((market == 'BTTS_YES' and btts == 1) or (market == 'BTTS_NO' and btts == 0))
                else:
                    won = 0

                stake = bet['Stake%'] / 100
                odds = bet['Odds']
                pnl = stake * (odds - 1) if won else -stake

                all_bets.append({
                    'Date': row['Date'], 'Season': row['Season'], 'Div': row['Div'],
                    'HomeTeam': row['HomeTeam'], 'AwayTeam': row['AwayTeam'],
                    'Market': market, 'MC_Prob': bet['MC_Prob'], 'LGBM_Prob': bet['LGBM_Prob'],
                    'Odds': odds, 'EV%': bet['EV%'], 'Stake%': bet['Stake%'],
                    'Actual': won, 'PnL': pnl, 'CLV%': bet['CLV%']
                })

    return pd.DataFrame(all_bets)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main(target_date: str = None, source: str = "oddsportal", pass_mode: str = "pre", backtest: bool = False):
    if target_date is None:
        target_date = (datetime.now(COL_TZ) + timedelta(days=1)).strftime("%Y-%m-%d")

    print("=" * 60)
    print(f"POISSON + MONTE CARLO ({MC_SIMS:,} sims) - {target_date}")
    print("=" * 60)

    if backtest:
        results = backtest_poisson_mc(n_splits=6, n_sims=5000)
        if len(results) == 0:
            print("No bets generated in backtest")
            return

        results.to_parquet(DATA_DIR / f"backtest_poisson_mc_{target_date}.parquet", index=False)

        n = len(results)
        stake = results['Stake%'].sum()
        pnl = results['PnL'].sum()
        roi = pnl / stake * 100 if stake > 0 else 0
        wr = results['Actual'].mean() * 100
        avg_ev = results['EV%'].mean()
        avg_clv = results['CLV%'].mean()

        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS: {n:,} bets | Stake: {stake:.2f} | PnL: {pnl:.2f} | ROI: {roi:.2f}%")
        print(f"Win Rate: {wr:.1f}% | Avg EV: {avg_ev:.2f}% | Avg CLV: {avg_clv:.2f}%")
        print(f"{'='*60}")

        return results

    # LIVE PREDICTION
    # 1. Load historical data for feature building
    historical_df = pd.read_parquet(FEATURES_FILE)

    # 2. Fetch fixtures (reuse fetch_live logic)
    from fetch_live import get_fixtures_for_date, get_fixtures_from_oddsportal, build_live_features, load_historical_data
    import asyncio

    print(f"\n[1/5] Fetching fixtures for {target_date} from {source}...")
    if source == "oddsportal":
        fixtures = asyncio.run(get_fixtures_from_oddsportal(target_date))
    else:
        fixtures = get_fixtures_for_date(target_date)
    print(f"       Found {len(fixtures)} fixtures")

    if len(fixtures) == 0:
        print("[NO BETS] No fixtures found")
        return

    # 3. Build features
    print(f"\n[2/5] Building features...")
    ref_date = pd.Timestamp(target_date)
    live_df = build_live_features(fixtures, historical_df, ref_date)

    # 4. Load models + get LightGBM probs
    print(f"\n[3/5] Loading LightGBM models...")
    models, calibrators = load_models_and_calibrators()
    df_hist, feat_cols, cat_cols = prepare_data(historical_df)
    live_df = get_lgbm_probs(live_df, models, calibrators, feat_cols, cat_cols)

    # 5. Run Poisson + Monte Carlo
    print(f"\n[4/5] Running Poisson + Monte Carlo ({MC_SIMS:,} sims)...")
    mc_results, all_bets = run_poisson_mc_batch(live_df, n_sims=MC_SIMS)

    # 6. Apply production filters
    print(f"\n[5/5] Applying production filters...")
    filtered_bets = apply_production_filters(all_bets, kelly_frac=0.10, max_stake_pct=25.0, max_bets_per_fixture=1)

    # 7. Output
    print(f"\n[6/6] Results:")
    if len(filtered_bets) > 0:
        print_bets_mc(filtered_bets)

        # Save
        suffix = "_post" if pass_mode == "post" else "_pre"
        filtered_bets.to_parquet(DATA_DIR / f"poisson_mc_bets_{target_date}{suffix}.parquet", index=False)

        # Save full MC results
        import json
        mc_summary = []
        for r in mc_results:
            mc_summary.append({
                'fixture': r['fixture'], 'date': str(r['date']), 'league': r['league'],
                'lambda_h': r['lambda_home'], 'lambda_a': r['lambda_away'],
                'fit_error': r['fit_error'],
                'dc_probs': r['dc_probs'],
                'mc_probs': r['mc_probs'],
                'top_scorelines': [[f"{hg}-{ag}", pct] for (hg, ag), pct in r['top_scorelines']],
                'n_bets': len(r['ev_bets'])
            })
        with open(DATA_DIR / f"poisson_mc_results_{target_date}{suffix}.json", 'w') as f:
            json.dump(mc_summary, f, indent=2)

        print(f"\n[SAVE] Bets + MC results saved")
    else:
        print("  No EV+ bets found after filters")


def print_bets_mc(bets_df: pd.DataFrame):
    """Pretty print Poisson+MC betting recommendations"""
    if len(bets_df) == 0:
        print("\n[NO BETS] NO EV+ BETS FOUND")
        return

    print(f"\n{'='*120}")
    print(f"[POISSON+MC] BETTING RECOMMENDATIONS - {len(bets_df)} EV+ opportunities")
    print(f"{'='*120}")
    print(f"{'Date':<18} {'League':<5} {'Match':<32} {'Market':<10} {'MC%':>5} {'LGBM%':>5} {'DC%':>5} {'Odds':>6} {'EV%':>7} {'Stake%':>7} {'CLV%':>6} {'KL':>5}")
    print(f"{'-'*120}")

    for _, row in bets_df.sort_values("EV%", ascending=False).iterrows():
        match = f"{row['Home'][:15]:<15} vs {row['Away'][:15]:<15}"
        mc_p = row.get('MC_Prob', 0)
        lgbm_p = row.get('LGBM_Prob', 0)
        dc_p = row.get('DC_Prob', 0)
        kl = row.get('KL_Score', 0)
        print(f"{row['Date']:<18} {row['League']:<5} {match:<32} {row['Market']:<10} "
              f"{mc_p:>5.1%} {lgbm_p:>5.1%} {dc_p:>5.1%} {row['Odds']:>6.2f} {row['EV%']:>7.2f}% "
              f"{row['Stake%']:>7.2f}% {row['CLV%']:>6.2f}% {kl:>5.3f}")

    total_stake = bets_df["Stake%"].sum()
    avg_ev = bets_df["EV%"].mean()
    avg_clv = bets_df["CLV%"].mean()
    print(f"{'-'*120}")
    print(f"TOTAL: {len(bets_df)} bets | Avg EV: {avg_ev:.2f}% | Avg CLV: {avg_clv:.2f}% | Total Stake: {total_stake:.2f}% bankroll")
    print(f"{'='*120}")


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("date", nargs="?", help="Target date (YYYY-MM-DD)")
    parser.add_argument("--source", choices=["api-football", "oddsportal"], default="oddsportal")
    parser.add_argument("--pass", dest="pass_mode", choices=["pre", "post"], default="pre")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward backtest")
    parser.add_argument("--sims", type=int, default=MC_SIMS, help="Monte Carlo iterations")
    args = parser.parse_args()

    if args.backtest:
        main(args.date, args.source, args.pass_mode, backtest=True)
    else:
        main(args.date, args.source, args.pass_mode)