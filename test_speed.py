import warnings
warnings.filterwarnings('ignore')
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from pathlib import Path
import time

DATA_DIR = Path('data')
MODEL_DIR = Path('models')
FEATURES_FILE = DATA_DIR / 'matches_pre_match.parquet'

EXCLUDE_FEATURES = {
    'Season', 'Date', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Result',
    'imp_prob_H', 'imp_prob_D', 'imp_prob_A', 'overround',
    'imp_prob_O25', 'imp_prob_U25', 'ou25_favorite',
    'favorite', 'favorite_prob', 'odds_ratio_H_A', 'draw_prob',
    'home_win_prob', 'away_win_prob', 'market_entropy', 'market_max_prob',
    'B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5',
    'B365AHH', 'B365AHA',
}

TARGETS = {
    'result': {'type': 'multiclass', 'classes': [-1, 0, 1]},
    'over_25': {'type': 'binary', 'threshold': 2.5},
    'btts': {'type': 'binary', 'threshold': None},
}

def prepare_data(df):
    df = df.sort_values('Date').reset_index(drop=True)
    df['target_result'] = df['Result'].map({-1: 0, 0: 1, 1: 2}) if 'Result' in df.columns else 0
    df['target_over_25'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int) if 'FTHG' in df.columns else 0
    df['target_btts'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int) if 'FTHG' in df.columns else 0
    feat_cols = [c for c in df.columns if c not in EXCLUDE_FEATURES and not c.startswith('target_')]
    cat_cols = df[feat_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype('category')
    num_cols = [c for c in feat_cols if c not in cat_cols]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    return df, feat_cols, cat_cols

def predict_calibrated(model, X, calibrators, task_type):
    probs = model.predict(X, num_iteration=model.best_iteration)
    if task_type == 'multiclass':
        cal_probs = np.column_stack([
            calibrators[i].predict(probs[:, i]) for i in range(probs.shape[1])
        ])
        cal_probs = cal_probs / cal_probs.sum(axis=1, keepdims=True)
        return cal_probs
    else:
        return calibrators.predict(probs)

df = pd.read_parquet(FEATURES_FILE)
df, feat_cols, cat_cols = prepare_data(df)
test_df = df.tail(10).copy()

models = {}
calibrators = {}
for target_name in TARGETS:
    model_path = MODEL_DIR / f'{target_name}_model.txt'
    cal_path = MODEL_DIR / f'{target_name}_calibrator.pkl'
    models[target_name] = lgb.Booster(model_file=str(model_path))
    with open(cal_path, 'rb') as f:
        calibrators[target_name] = joblib.load(f)

X_live = test_df[feat_cols].copy()
for c in cat_cols:
    if c in X_live.columns:
        X_live[c] = X_live[c].astype('category')

for target_name, target_info in TARGETS.items():
    model = models[target_name]
    cal = calibrators[target_name]
    probs = predict_calibrated(model, X_live, cal, target_info['type'])
    if target_info['type'] == 'multiclass':
        test_df[f'pred_{target_name}_A'] = probs[:, 0]
        test_df[f'pred_{target_name}_D'] = probs[:, 1]
        test_df[f'pred_{target_name}_H'] = probs[:, 2]
    else:
        test_df[f'pred_{target_name}_1'] = probs
        test_df[f'pred_{target_name}_0'] = 1 - probs

print('LGBM probs computed')

# Now test MC for 10 matches
RHO = 0.13
MAX_GOALS = 10
MC_SIMS = 1000

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
            if hg > ag: h_win += p
            elif hg == ag: d_win += p
            else: a_win += p
            if hg + ag > 2.5: over25 += p
            if hg > 0 and ag > 0: btts += p
    total = h_win + d_win + a_win
    if total > 0:
        h_win, d_win, a_win = h_win/total, d_win/total, a_win/total
    return {'H': h_win, 'D': d_win, 'A': a_win, 'over_25': over25, 'under_25': 1 - over25, 'btts_yes': btts, 'btts_no': 1 - btts}

def objective_lambdas(params, target_probs):
    lambda_h, lambda_a = params
    if lambda_h <= 0 or lambda_a <= 0: return 1e6
    dc_probs = match_probs_from_lambdas(lambda_h, lambda_a)
    error = 0.0
    for key in ['H', 'D', 'A', 'over_25', 'btts_yes']:
        target = target_probs.get(key, 0)
        if target > 0: error += (dc_probs[key] - target) ** 2
    return error

def estimate_lambdas(ml_probs, init_guess=(1.5, 1.0)):
    target = {'H': ml_probs['pH'], 'D': ml_probs['pD'], 'A': ml_probs['pA'], 'over_25': ml_probs['pO25'], 'btts_yes': ml_probs['pBTTS']}
    best_error = float('inf')
    best_params = init_guess
    for lh in np.linspace(0.3, 4.0, 5):
        for la in np.linspace(0.3, 3.5, 5):
            error = objective_lambdas((lh, la), target)
            if error < best_error:
                best_error = error
                best_params = (lh, la)
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
    return {'results': results, 'over25': over25, 'btts': btts}

start = time.time()
for idx, row in test_df.iterrows():
    ml_probs = {'pH': row.get('pred_result_H', 0), 'pD': row.get('pred_result_D', 0),
                'pA': row.get('pred_result_A', 0), 'pO25': row.get('pred_over_25_1', 0),
                'pBTTS': row.get('pred_btts_1', 0)}
    lambda_h, lambda_a, fit_error = estimate_lambdas(ml_probs)
    dc_probs = match_probs_from_lambdas(lambda_h, lambda_a)
    sim = simulate_match(lambda_h, lambda_a, 1000)
    emp_probs = {'H': (sim['results'] == 'H').mean(), 'D': (sim['results'] == 'D').mean(), 'A': (sim['results'] == 'A').mean(),
                 'over_25': sim['over25'].mean(), 'under_25': 1 - sim['over25'].mean(),
                 'btts_yes': sim['btts'].mean(), 'btts_no': 1 - sim['btts'].mean()}
elapsed = time.time() - start
print(f'10 matches MC (1000 sims each): {elapsed:.2f}s')
print(f'Per match: {elapsed/10:.3f}s')