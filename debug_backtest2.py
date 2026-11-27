import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from model_poisson_mc import *
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

def backtest_poisson_mc_debug(n_splits: int = 1, n_sims: int = 100) -> pd.DataFrame:
    """Walk-forward backtest of Poisson + MC vs pure LightGBM"""
    print(f"\n[BACKTEST] Poisson+MC vs LightGBM - {n_splits} folds, {n_sims} sims")

    df = pd.read_parquet(FEATURES_FILE)
    df, feat_cols, cat_cols = prepare_data(df)

    models, calibrators = load_models_and_calibrators()

    # Walk-forward splits
    df["YearMonth"] = df["Date"].dt.to_period("M")
    months = sorted(df["YearMonth"].unique())

    min_train_months = 12
    test_months_per_split = max(1, (len(months) - min_train_months) // n_splits)

    all_bets = []

    for fold in range(n_splits):
        test_start_idx = min_train_months + fold * test_months_per_split
        test_end_idx = test_start_idx + test_months_per_split
        if test_end_idx >= len(months):
            break

        train_end_idx = test_start_idx - 1
        train_months = months[:train_end_idx]
        test_months = months[test_start_idx:test_end_idx]

        train_mask = df["YearMonth"].isin(train_months)
        test_mask = df["YearMonth"].isin(test_months)

        if train_mask.sum() < 1000 or test_mask.sum() < 50:
            continue

        print(f"  Fold {fold+1}: train={train_mask.sum():,} test={test_mask.sum():,}")

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
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(val_probs[:, i], (y_val == i).astype(int))
                    cals.append(iso)
            else:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(val_probs, y_val)
                cals = iso

            fold_models[target_name] = model
            fold_calibrators[target_name] = cals

        # Predict on test set
        test_df = df.loc[test_mask].copy()
        test_df = get_lgbm_probs(test_df, fold_models, fold_calibrators, feat_cols, cat_cols)

        # Run Poisson+MC for each test match
        bet_count = 0
        for _, row in test_df.iterrows():
            result = run_poisson_mc_for_fixture(row, n_sims)

            if 'ev_bets' not in result or len(result['ev_bets']) == 0:
                continue

            bet_count += len(result['ev_bets'])
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

        print(f"    Fold {fold+1} generated {bet_count} bets")

    return pd.DataFrame(all_bets)

if __name__ == "__main__":
    results = backtest_poisson_mc_debug(n_splits=1, n_sims=100)
    print(f'\nTotal bets: {len(results)}')
    if len(results) > 0:
        stake = results['Stake%'].sum()
        pnl = results['PnL'].sum()
        roi = pnl / stake * 100 if stake > 0 else 0
        wr = results['Actual'].mean() * 100
        avg_ev = results['EV%'].mean()
        avg_clv = results['CLV%'].mean()
        print(f'Stake: {stake:.2f} | PnL: {pnl:.2f} | ROI: {roi:.2f}%')
        print(f'Win Rate: {wr:.1f}% | Avg EV: {avg_ev:.2f}% | Avg CLV: {avg_clv:.2f}%')