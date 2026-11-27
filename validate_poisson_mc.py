#!/usr/bin/env python
"""
Quick validation: Poisson+MC vs pure LightGBM using PRE-TRAINED models.
No retraining - uses saved models from models/ directory.
"""
import warnings
warnings.filterwarnings('ignore')

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from pathlib import Path

# Config
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
FEATURES_FILE = DATA_DIR / "matches_pre_match.parquet"

RHO = 0.13
MAX_GOALS = 10
MC_SIMS = 1000  # Quick validation

MIN_EV = 0.05
MAX_ODDS = 5.0
MIN_ODDS = 1.5
KELLY_FRAC = 0.25

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
}


def dixon_coles_prob(h_goals, a_goals, lambda_h, lambda_a, rho=RHO):
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


def match_probs_from_lambdas(lambda_h, lambda_a, max_goals=MAX_GOALS):
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


def estimate_lambdas(ml_probs, init_guess=(1.5, 1.0)):
    target = {
        'H': ml_probs['pH'], 'D': ml_probs['pD'], 'A': ml_probs['pA'],
        'over_25': ml_probs['pO25'], 'btts_yes': ml_probs['pBTTS']
    }
    best_error = float('inf')
    best_params = init_guess
    for lh in np.linspace(0.3, 4.0, 38):
        for la in np.linspace(0.3, 3.5, 33):
            error = objective_lambdas((lh, la), target)
            if error < best_error:
                best_error = error
                best_params = (lh, la)
    result = minimize_scalar(
        lambda lh: objective_lambdas((lh, best_params[1]), target),
        bounds=(0.2, 4.5), method='bounded'
    )
    best_lh = result.x
    result = minimize_scalar(
        lambda la: objective_lambdas((best_lh, la), target),
        bounds=(0.2, 4.5), method='bounded'
    )
    best_la = result.x
    final_error = objective_lambdas((best_lh, best_la), target)
    return best_lh, best_la, final_error


def simulate_match(lambda_h, lambda_a, n_sims=MC_SIMS, max_goals=MAX_GOALS):
    score_probs = {}
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            score_probs[(hg, ag)] = dixon_coles_prob(hg, ag, lambda_h, lambda_a)
    total = sum(score_probs.values())
    score_probs = {k: v/total for k, v in score_probs.items()}
    scores = list(score_probs.keys())
    probs = list(score_probs.values())
    samples = np.random.choice(len(scores), size=n_sims, p=probs)
    home_goals = np.array([scores[s][0] for s in samples])
    away_goals = np.array([scores[s][1] for s in samples])
    total_goals = home_goals + away_goals
    results = np.where(home_goals > away_goals, 'H', np.where(home_goals == away_goals, 'D', 'A'))
    over25 = (total_goals > 2.5).astype(int)
    btts = ((home_goals > 0) & (away_goals > 0)).astype(int)
    return {
        'home_goals': home_goals, 'away_goals': away_goals,
        'total_goals': total_goals, 'results': results,
        'over25': over25, 'btts': btts,
        'score_probs': score_probs
    }


def empirical_probs_from_sim(sim):
    return {
        'H': (sim['results'] == 'H').mean(),
        'D': (sim['results'] == 'D').mean(),
        'A': (sim['results'] == 'A').mean(),
        'over_25': sim['over25'].mean(),
        'under_25': 1 - sim['over25'].mean(),
        'btts_yes': sim['btts'].mean(),
        'btts_no': 1 - sim['btts'].mean()
    }


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


def prepare_data(df):
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


def get_lgbm_probs(live_df, models, calibrators, feat_cols, cat_cols):
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


def run_poisson_mc_for_fixture(row, n_sims=MC_SIMS):
    ml_probs = {
        'pH': row.get('pred_result_H', 0),
        'pD': row.get('pred_result_D', 0),
        'pA': row.get('pred_result_A', 0),
        'pO25': row.get('pred_over_25_1', 0),
        'pBTTS': row.get('pred_btts_1', 0)
    }
    if min(ml_probs.values()) <= 0:
        return {"error": "Missing ML probabilities"}

    lambda_h, lambda_a, fit_error = estimate_lambdas(ml_probs)
    dc_probs = match_probs_from_lambdas(lambda_h, lambda_a)
    sim = simulate_match(lambda_h, lambda_a, n_sims)
    emp_probs = empirical_probs_from_sim(sim)

    # Get Bet365 odds
    odds = {
        'home': row.get('B365H'), 'draw': row.get('B365D'), 'away': row.get('B365A'),
        'over_2_5': row.get('B365>2.5'), 'under_2_5': row.get('B365<2.5'),
    }
    # Fill missing odds from empirical probs
    margin = 0.05
    if pd.isna(odds.get('over_2_5')) or odds.get('over_2_5') is None:
        odds['over_2_5'] = (1 / emp_probs['over_25']) * (1 + margin)
        odds['under_2_5'] = (1 / emp_probs['under_25']) * (1 + margin)

    ev_bets = []
    markets = {
        '1X2_H': ('H', emp_probs['H'], odds.get('home'), ml_probs['pH'], dc_probs['H']),
        '1X2_D': ('D', emp_probs['D'], odds.get('draw'), ml_probs['pD'], dc_probs['D']),
        '1X2_A': ('A', emp_probs['A'], odds.get('away'), ml_probs['pA'], dc_probs['A']),
        'OVER_2.5': ('Over 2.5', emp_probs['over_25'], odds.get('over_2_5'), ml_probs['pO25'], dc_probs['over_25']),
        'UNDER_2.5': ('Under 2.5', emp_probs['under_25'], odds.get('under_2_5'), 1-ml_probs['pO25'], dc_probs['under_25']),
    }

    for mkt_name, (label, p_mc, odd, p_lgbm, p_dc) in markets.items():
        if odd is None or pd.isna(odd) or odd < MIN_ODDS or odd > MAX_ODDS:
            continue
        ev = p_mc * odd - 1
        if ev <= MIN_EV:
            continue
        kelly = (p_mc * odd - 1) / (odd - 1) if odd > 1 else 0
        stake = max(0, kelly * KELLY_FRAC)
        clv = (p_mc - 1/odd) * 100
        ev_bets.append({
            'Market': mkt_name, 'Label': label,
            'MC_Prob': round(p_mc, 4), 'LGBM_Prob': round(p_lgbm, 4),
            'DC_Prob': round(p_dc, 4), 'Odds': round(odd, 2),
            'EV%': round(ev * 100, 2), 'Kelly%': round(kelly * 100, 2),
            'Stake%': round(stake * 100, 2), 'CLV%': round(clv, 2),
        })

    return {
        'lambda_home': round(lambda_h, 3), 'lambda_away': round(lambda_a, 3),
        'fit_error': round(fit_error, 6),
        'dc_probs': {k: round(v, 4) for k, v in dc_probs.items()},
        'mc_probs': {k: round(v, 4) for k, v in emp_probs.items()},
        'ev_bets': pd.DataFrame(ev_bets) if ev_bets else pd.DataFrame(),
    }


def quick_validation():
    print(f"[QUICK VALIDATION] Poisson+MC ({MC_SIMS} sims) vs pure LightGBM")
    print("=" * 60)

    # Load data
    df = pd.read_parquet(FEATURES_FILE)
    df, feat_cols, cat_cols = prepare_data(df)

    # Use last 100 matches for validation (out-of-sample)
    test_df = df.tail(100).copy()
    print(f"Test matches: {len(test_df)}")

    # Load PRE-TRAINED models
    models, calibrators = load_models_and_calibrators()

    # Get LightGBM probs
    test_df = get_lgbm_probs(test_df, models, calibrators, feat_cols, cat_cols)

    # Run Poisson+MC for each test match
    lgbm_bets = []
    mc_bets = []

    print("Running predictions...")
    for idx, row in test_df.iterrows():
        # Pure LightGBM bets
        ml_probs = {
            'pH': row.get('pred_result_H', 0), 'pD': row.get('pred_result_D', 0),
            'pA': row.get('pred_result_A', 0), 'pO25': row.get('pred_over_25_1', 0),
            'pBTTS': row.get('pred_btts_1', 0)
        }
        odds = {'home': row.get('B365H'), 'draw': row.get('B365D'), 'away': row.get('B365A'),
                'over_2_5': row.get('B365>2.5'), 'under_2_5': row.get('B365<2.5')}

        # LGBM EV bets
        for mkt_name, p_lgbm, odd_key in [
            ('1X2_H', ml_probs['pH'], 'home'), ('1X2_D', ml_probs['pD'], 'draw'),
            ('1X2_A', ml_probs['pA'], 'away'), ('OVER_2.5', ml_probs['pO25'], 'over_2_5'),
            ('UNDER_2.5', 1-ml_probs['pO25'], 'under_2_5')
        ]:
            odd = odds.get(odd_key)
            if odd and MIN_ODDS <= odd <= MAX_ODDS:
                ev = p_lgbm * odd - 1
                if ev > MIN_EV:
                    kelly = (p_lgbm * odd - 1) / (odd - 1) if odd > 1 else 0
                    stake = max(0, kelly * KELLY_FRAC)
                    clv = (p_lgbm - 1/odd) * 100
                    lgbm_bets.append({
                        'Date': row['Date'], 'Home': row['HomeTeam'], 'Away': row['AwayTeam'],
                        'Market': mkt_name, 'Prob': p_lgbm, 'Odds': odd,
                        'EV%': ev*100, 'Stake%': stake*100, 'CLV%': clv
                    })

        # Poisson+MC
        mc_result = run_poisson_mc_for_fixture(row, MC_SIMS)
        if len(mc_result['ev_bets']) > 0:
            for _, bet in mc_result['ev_bets'].iterrows():
                # Get actual result
                market = bet['Market']
                if market.startswith('1X2'):
                    actual_result = row['Result']
                    pred = {'1X2_H': 1, '1X2_D': 0, '1X2_A': -1}[market]
                    won = int(actual_result == pred)
                elif market in ['OVER_2.5', 'UNDER_2.5']:
                    total = row['FTHG'] + row['FTAG']
                    won = int((market == 'OVER_2.5' and total > 2.5) or (market == 'UNDER_2.5' and total < 2.5))
                else:
                    won = 0

                stake = bet['Stake%'] / 100
                odds_val = bet['Odds']
                pnl = stake * (odds_val - 1) if won else -stake

                mc_bets.append({
                    'Date': row['Date'], 'Home': row['HomeTeam'], 'Away': row['AwayTeam'],
                    'Market': market, 'MC_Prob': bet['MC_Prob'], 'LGBM_Prob': bet['LGBM_Prob'],
                    'Odds': odds_val, 'EV%': bet['EV%'], 'Stake%': bet['Stake%'],
                    'Actual': won, 'PnL': pnl, 'CLV%': bet['CLV%']
                })

    lgbm_df = pd.DataFrame(lgbm_bets)
    mc_df = pd.DataFrame(mc_bets)

    print(f"\n{'='*60}")
    print("COMPARISON: Pure LightGBM vs Poisson+MC")
    print(f"{'='*60}")

    for name, bets_df in [("LightGBM", lgbm_df), ("Poisson+MC", mc_df)]:
        if len(bets_df) == 0:
            print(f"\n{name}: NO BETS")
            continue
        n = len(bets_df)
        stake = bets_df['Stake%'].sum()
        pnl = bets_df['PnL'].sum() if 'PnL' in bets_df.columns else 0
        roi = pnl / stake * 100 if stake > 0 else 0
        wr = bets_df['Actual'].mean() * 100 if 'Actual' in bets_df.columns else 0
        avg_ev = bets_df['EV%'].mean()
        avg_clv = bets_df['CLV%'].mean()
        print(f"\n{name}: {n} bets")
        print(f"  Stake: {stake:.2f}% | PnL: {pnl:.2f} | ROI: {roi:.2f}%")
        print(f"  Win Rate: {wr:.1f}% | Avg EV: {avg_ev:.2f}% | Avg CLV: {avg_clv:.2f}%")

    # Save for inspection
    mc_df.to_parquet(DATA_DIR / "quick_val_mc.parquet", index=False)
    lgbm_df.to_parquet(DATA_DIR / "quick_val_lgbm.parquet", index=False)
    print(f"\nSaved validation results to data/quick_val_*.parquet")


if __name__ == "__main__":
    quick_validation()