#!/usr/bin/env python
"""
Statistical Significance Testing for Betting Models
Bootstrap, permutation tests, and confidence intervals
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")

# Load all backtest results
files = {
    "LGB_only": DATA_DIR / "backtest_ml_clean.parquet",
    "Ensemble": DATA_DIR / "backtest_ensemble.parquet",
    "Poisson_xG": DATA_DIR / "backtest_results_xg.parquet",
}


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95, statistic=np.mean) -> tuple:
    """Bootstrap confidence interval for a statistic."""
    n = len(data)
    boot_stats = []

    for _ in range(n_bootstrap):
        sample = data[np.random.choice(n, n, replace=True)]
        boot_stats.append(statistic(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_stats, alpha * 100)
    upper = np.percentile(boot_stats, (1 - alpha) * 100)
    return lower, upper, np.mean(boot_stats)


def permutation_test(returns: np.ndarray, n_perm: int = 10000) -> float:
    """Permutation test: H0 = mean return = 0 (no edge)."""
    observed_mean = returns.mean()
    perm_means = []

    for _ in range(n_perm):
        # Randomly flip signs (null: no predictive power)
        perm_sample = returns * np.random.choice([-1, 1], size=len(returns))
        perm_means.append(perm_sample.mean())

    p_value = (np.sum(np.abs(perm_means) >= np.abs(observed_mean)) + 1) / (n_perm + 1)
    return p_value


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Sharpe ratio of returns."""
    excess = returns - risk_free
    if excess.std() == 0:
        return 0
    return excess.mean() / excess.std() * np.sqrt(252)  # Annualized


def max_drawdown(pnl_series: np.ndarray) -> float:
    """Maximum drawdown from peak."""
    cumsum = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cumsum)
    drawdown = cumsum - peak
    return drawdown.min()


def kelly_bet_sizing(p_model: np.ndarray, odds: np.ndarray, fraction: float = 0.25) -> np.ndarray:
    """Kelly criterion stake sizing."""
    kelly = (p_model * odds - 1) / (odds - 1)
    kelly = np.clip(kelly, 0, 1)
    return kelly * fraction


def analyze_model(name: str, filepath: Path):
    print(f"\n{'='*60}")
    print(f"STATISTICAL ANALYSIS: {name}")
    print(f"{'='*60}")

    df = pd.read_parquet(filepath)
    if len(df) == 0:
        print("No bets!")
        return

    n = len(df)
    stake = df["Stake"].sum()
    pnl = df["PnL"].sum()
    roi = pnl / stake * 100 if stake > 0 else 0
    wr = df["Actual"].mean() * 100
    avg_ev = df["EV"].mean() * 100
    avg_clv = df["CLV"].mean()

    print(f"\n## Basic Metrics")
    print(f"- Bets: {n:,}")
    print(f"- Total Stake: {stake:.2f}")
    print(f"- Total PnL: {pnl:.2f}")
    print(f"- ROI: {roi:.2f}%")
    print(f"- Win Rate: {wr:.1f}%")
    print(f"- Avg EV: {avg_ev:.2f}%")
    print(f"- Avg CLV: {avg_clv:.2f}%")

    # Per-bet returns (PnL / Stake)
    returns = df["PnL"] / df["Stake"]
    print(f"\n## Return Distribution")
    print(f"- Mean return per bet: {returns.mean():.4f}")
    print(f"- Std return per bet: {returns.std():.4f}")
    print(f"- Sharpe Ratio (annualized): {sharpe_ratio(returns.values):.2f}")
    print(f"- Max Drawdown: {max_drawdown(df['PnL'].values):.2f}")

    # Bootstrap CI for ROI
    roi_ci_lower, roi_ci_upper, _ = bootstrap_ci(returns.values, statistic=lambda x: x.mean() * 100)
    print(f"\n## Bootstrap 95% CI for ROI")
    print(f"- ROI 95% CI: [{roi_ci_lower:.2f}%, {roi_ci_upper:.2f}%]")
    print(f"- Significant (CI > 0): {roi_ci_lower > 0}")

    # Permutation test
    p_val = permutation_test(returns.values)
    print(f"\n## Permutation Test (H0: no edge)")
    print(f"- p-value: {p_val:.6f}")
    print(f"- Significant at 5%: {p_val < 0.05}")
    print(f"- Significant at 1%: {p_val < 0.01}")
    print(f"- Significant at 0.1%: {p_val < 0.001}")

    # CLV significance
    clv_ci_lower, clv_ci_upper, _ = bootstrap_ci(df["CLV"].values)
    print(f"\n## CLV Analysis")
    print(f"- Mean CLV: {avg_clv:.2f}%")
    print(f"- CLV 95% CI: [{clv_ci_lower:.2f}%, {clv_ci_upper:.2f}%]")
    print(f"- CLV > 0 significant: {clv_ci_lower > 0}")

    # By market
    print(f"\n## By Market")
    for mkt in sorted(df["Market"].unique()):
        d = df[df["Market"] == mkt]
        if len(d) < 20:
            continue
        r = d["PnL"].sum() / d["Stake"].sum() * 100 if d["Stake"].sum() > 0 else 0
        w = d["Actual"].mean() * 100
        c = d["CLV"].mean()
        # Significance
        p_val_mkt = permutation_test((d["PnL"] / d["Stake"]).values)
        sig = "***" if p_val_mkt < 0.001 else "**" if p_val_mkt < 0.01 else "*" if p_val_mkt < 0.05 else ""
        print(f"- {mkt}: n={len(d):,} ROI={r:.2f}% WR={w:.1f}% CLV={c:.2f}% {sig} (p={p_val_mkt:.4f})")

    # By season
    print(f"\n## By Season")
    for sea in sorted(df["Season"].unique()):
        d = df[df["Season"] == sea]
        r = d["PnL"].sum() / d["Stake"].sum() * 100 if d["Stake"].sum() > 0 else 0
        w = d["Actual"].mean() * 100
        p_val_s = permutation_test((d["PnL"] / d["Stake"]).values)
        sig = "***" if p_val_s < 0.001 else "**" if p_val_s < 0.01 else "*" if p_val_s < 0.05 else ""
        print(f"- {sea}: n={len(d):,} ROI={r:.2f}% WR={w:.1f}% {sig} (p={p_val_s:.4f})")

    # Kelly optimization check
    print(f"\n## Kelly Fraction Sensitivity")
    for frac in [0.1, 0.25, 0.5, 1.0]:
        stakes = kelly_bet_sizing(df["Model_Prob"].values, df["Odds"].values, frac)
        pnl_kelly = np.where(df["Actual"] == 1, stakes * (df["Odds"] - 1), -stakes)
        roi_kelly = pnl_kelly.sum() / stakes.sum() * 100 if stakes.sum() > 0 else 0
        dd_kelly = max_drawdown(pnl_kelly)
        print(f"- Kelly {frac:.0%}: ROI={roi_kelly:.2f}% MaxDD={dd_kelly:.2f} Stake={stakes.sum():.1f}")

    return {
        "name": name,
        "n": n,
        "roi": roi,
        "wr": wr,
        "clv": avg_clv,
        "sharpe": sharpe_ratio(returns.values),
        "max_dd": max_drawdown(df["PnL"].values),
        "p_value": p_val,
        "roi_ci": (roi_ci_lower, roi_ci_upper),
        "clv_ci": (clv_ci_lower, clv_ci_upper),
    }


def compare_models(results: list):
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Bets':>8} {'ROI%':>8} {'WR%':>6} {'CLV%':>7} {'Sharpe':>7} {'MaxDD':>8} {'p-value':>10} {'Sig'}")
    print(f"-" * 85)
    for r in results:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        print(f"{r['name']:<20} {r['n']:>8,} {r['roi']:>8.2f} {r['wr']:>6.1f} {r['clv']:>7.2f} {r['sharpe']:>7.2f} {r['max_dd']:>8.2f} {r['p_value']:>10.6f} {sig}")


def main():
    print("=" * 60)
    print("STATISTICAL VALIDATION OF BETTING MODELS")
    print("=" * 60)

    np.random.seed(42)
    all_results = []

    for name, filepath in files.items():
        if filepath.exists():
            try:
                result = analyze_model(name, filepath)
                all_results.append(result)
            except Exception as e:
                print(f"Error analyzing {name}: {e}")
        else:
            print(f"\nFile not found: {filepath}")

    if all_results:
        compare_models(all_results)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        status = "[OK] STATISTICALLY SIGNIFICANT" if r["p_value"] < 0.05 else "[FAIL] NOT SIGNIFICANT"
        print(f"- {r['name']}: {status} (p={r['p_value']:.6f}, ROI CI={r['roi_ci'][0]:.1f}%-{r['roi_ci'][1]:.1f}%)")


if __name__ == "__main__":
    main()