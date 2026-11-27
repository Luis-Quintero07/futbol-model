import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from model_poisson_mc import *
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

# Load and prepare data
df = pd.read_parquet('data/matches_pre_match.parquet')
df, feat_cols, cat_cols = prepare_data(df)

# Walk-forward splits
df['YearMonth'] = df['Date'].dt.to_period('M')
months = sorted(df['YearMonth'].unique())
min_train_months = 12
n_splits = 1
test_months_per_split = max(1, (len(months) - min_train_months) // n_splits)
test_start_idx = min_train_months + 0 * test_months_per_split
test_end_idx = test_start_idx + test_months_per_split
train_end_idx = test_start_idx - 1
train_months = months[:train_end_idx]
test_months = months[test_start_idx:test_end_idx]
train_mask = df['YearMonth'].isin(train_months)
test_mask = df['YearMonth'].isin(test_months)

print(f'Train months: {len(train_months)} ({train_months[0]} to {train_months[-1]})', flush=True)
print(f'Test months: {len(test_months)} ({test_months[0]} to {test_months[-1]})', flush=True)
print(f'Train rows: {train_mask.sum()}, Test rows: {test_mask.sum()}', flush=True)

# Retrain models
fold_models = {}
fold_calibrators = {}
for target_name, target_info in TARGETS.items():
    target_col = f'target_{target_name}'
    X_train = df.loc[train_mask, feat_cols]
    y_train = df.loc[train_mask, target_col]
    X_val = df.loc[test_mask, feat_cols]
    y_val = df.loc[test_mask, target_col]

    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    lgb_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=lgb_train)

    model = lgb.train({
        'objective': 'multiclass' if target_info['type'] == 'multiclass' else 'binary',
        'metric': 'multi_logloss' if target_info['type'] == 'multiclass' else 'binary_logloss',
        'num_class': 3 if target_info['type'] == 'multiclass' else 1,
        'learning_rate': 0.03, 'num_leaves': 63, 'verbosity': -1, 'seed': 42,
    }, lgb_train, valid_sets=[lgb_val], num_boost_round=2000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

    val_probs = model.predict(X_val, num_iteration=model.best_iteration)
    if target_info['type'] == 'multiclass':
        cals = []
        for i in range(val_probs.shape[1]):
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(val_probs[:, i], (y_val == i).astype(int))
            cals.append(iso)
    else:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(val_probs, y_val)
        cals = iso

    fold_models[target_name] = model
    fold_calibrators[target_name] = cals

# Predict on test set
test_df = df.loc[test_mask].copy()
test_df = get_lgbm_probs(test_df, fold_models, fold_calibrators, feat_cols, cat_cols)

print(f'Test df shape: {test_df.shape}', flush=True)

# Run Poisson+MC for ALL test matches and count bets
total_bets = 0
matches_with_bets = 0
for idx, row in test_df.iterrows():
    result = run_poisson_mc_for_fixture(row, 100)
    if 'ev_bets' in result and len(result['ev_bets']) > 0:
        matches_with_bets += 1
        total_bets += len(result['ev_bets'])
        if matches_with_bets <= 10:
            print(f'  {row["HomeTeam"]} vs {row["AwayTeam"]}: {len(result["ev_bets"])} bets', flush=True)
            for _, bet in result['ev_bets'].iterrows():
                print(f'    {bet["Market"]}: EV={bet["EV%"]:.2f}% Stake={bet["Stake%"]:.2f}%', flush=True)

print(f'\nTotal matches with bets: {matches_with_bets}', flush=True)
print(f'Total bets: {total_bets}', flush=True)