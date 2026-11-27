#!/usr/bin/env python
"""
Modelo Poisson + Dixon-Coles con backtest walk-forward (OPTIMIZADO)
Entrada: data/matches_clean.parquet
Salida: data/backtest_results.parquet + resumen
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
CLEAN_FILE = DATA_DIR / "matches_clean.parquet"
RESULTS_FILE = DATA_DIR / "backtest_results.parquet"
SUMMARY_FILE = DATA_DIR / "backtest_summary.md"

# ──────────────────────────────────────────────
# PARÁMETROS
# ──────────────────────────────────────────────
HALF_LIFE_DAYS = 150
MIN_MATCHES_FIT = 5
DIXON_COLES_RHO = 0.13
MIN_EV_THRESHOLD = 0.05
MAX_ODDS = 5.0
MIN_ODDS = 1.5
KELLY_FRACTION = 0.25
MAX_GOALS = 8  # Para matriz Poisson

MARKETS = {
    "1X2_H": {"target": "Result", "value": 1, "odds_col": "B365H", "prob_key": "pH"},
    "1X2_D": {"target": "Result", "value": 0, "odds_col": "B365D", "prob_key": "pD"},
    "1X2_A": {"target": "Result", "value": -1, "odds_col": "B365A", "prob_key": "pA"},
    "OVER_2.5": {"target": "total_goals", "value": "over", "odds_col": "B365>2.5", "prob_key": "pO25"},
    "UNDER_2.5": {"target": "total_goals", "value": "under", "odds_col": "B365<2.5", "prob_key": "pU25"},
    "BTTS_YES": {"target": "btts", "value": 1, "odds_col": None, "prob_key": "pBTTS"},
    "BTTS_NO": {"target": "btts", "value": 0, "odds_col": None, "prob_key": "pNBTTS"},
}

# ──────────────────────────────────────────────
# FUNCIONES VECTORIZADAS
# ──────────────────────────────────────────────
def time_weights(dates: pd.Series, ref_date: pd.Timestamp, half_life: float = HALF_LIFE_DAYS) -> np.ndarray:
    days_diff = (ref_date - dates).dt.days.clip(lower=0).values
    return np.power(0.5, days_diff / half_life)


def dixon_coles_matrix(lambda_h: float, lambda_a: float, rho: float = DIXON_COLES_RHO, max_g: int = MAX_GOALS) -> np.ndarray:
    """Matriz Poisson conjunta con corrección Dixon-Coles (vectorizada)"""
    x = np.arange(max_g + 1)
    y = np.arange(max_g + 1)
    pmf_x = poisson.pmf(x, lambda_h)
    pmf_y = poisson.pmf(y, lambda_a)
    mat = np.outer(pmf_x, pmf_y)

    # Corrección solo en esquinas 0-1
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
    # 1X2
    pH = mat[1:, :].sum() - np.diag(mat[1:, 1:]).sum()
    pD = np.diag(mat).sum()
    pA = mat[:, 1:].sum() - np.diag(mat[1:, 1:]).sum()
    s = pH + pD + pA
    pH, pD, pA = pH/s, pD/s, pA/s

    # Over/Under 2.5
    pO25 = sum(mat[i, j] for i in range(max_g+1) for j in range(max_g+1) if i + j > 2)
    pU25 = 1 - pO25

    # BTTS
    pBTTS = 1 - mat[0, :].sum() - mat[:, 0].sum() + mat[0, 0]
    pNBTTS = 1 - pBTTS

    return {"pH": pH, "pD": pD, "pA": pA, "pO25": pO25, "pU25": pU25, "pBTTS": pBTTS, "pNBTTS": pNBTTS}


def estimate_strengths_vectorized(df_hist: pd.DataFrame, ref_date: pd.Timestamp) -> tuple[dict, dict, float, float]:
    """Estima attack/defense/home_adv/league_avg de forma vectorizada"""
    df = df_hist[df_hist["Date"] < ref_date].copy()
    if len(df) < MIN_MATCHES_FIT * 2:
        return {}, {}, 1.0, 2.5

    w = time_weights(df["Date"], ref_date)
    df["w"] = w

    # League average ponderado
    total_g = (df["FTHG"] * w).sum() + (df["FTAG"] * w).sum()
    total_w = w.sum() * 2
    league_avg = total_g / total_w if total_w > 0 else 2.5

    # Home advantage
    hg = (df["FTHG"] * w).sum()
    ag = (df["FTAG"] * w).sum()
    home_adv = hg / ag if ag > 0 else 1.0

    # Equipos
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
    n_teams = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}

    # Matrices de diseño para regression ponderada
    # Home goals ~ attack_home * defense_away * home_adv * league_avg
    # Away goals ~ attack_away * defense_home * league_avg

    # Inicializa
    attack = np.ones(n_teams)
    defense = np.ones(n_teams)

    # Índices vectorizados
    h_idx = np.array([team_to_idx[t] for t in df["HomeTeam"]])
    a_idx = np.array([team_to_idx[t] for t in df["AwayTeam"]])

    # Iterative fitting (3-4 iteraciones bastan)
    for _ in range(4):
        # Attack: goles marcados / (league_avg * defense_opp * home_adv * weight)
        # Home
        num_h = np.bincount(h_idx, weights=df["FTHG"] * df["w"], minlength=n_teams)
        den_h = np.bincount(h_idx, weights=defense[a_idx] * home_adv * league_avg * df["w"], minlength=n_teams)
        # Away
        num_a = np.bincount(a_idx, weights=df["FTAG"] * df["w"], minlength=n_teams)
        den_a = np.bincount(a_idx, weights=defense[h_idx] * league_avg * df["w"], minlength=n_teams)

        num = num_h + num_a
        den = den_h + den_a
        mask = den > 0
        attack[mask] = num[mask] / den[mask]

        # Defense: goles recibidos / (league_avg * attack_opp * weight)
        num_h = np.bincount(h_idx, weights=df["FTAG"] * df["w"], minlength=n_teams)
        den_h = np.bincount(h_idx, weights=attack[a_idx] * league_avg * df["w"], minlength=n_teams)
        num_a = np.bincount(a_idx, weights=df["FTHG"] * df["w"], minlength=n_teams)
        den_a = np.bincount(a_idx, weights=attack[h_idx] * home_adv * league_avg * df["w"], minlength=n_teams)

        num = num_h + num_a
        den = den_h + den_a
        mask = den > 0
        defense[mask] = num[mask] / den[mask]

    # Normaliza
    attack = attack / attack.mean()
    defense = defense / defense.mean()

    attack_dict = {t: attack[i] for t, i in team_to_idx.items()}
    defense_dict = {t: defense[i] for t, i in team_to_idx.items()}

    return attack_dict, defense_dict, home_adv, league_avg


def predict_match_fast(home: str, away: str, attack: dict, defense: dict, home_adv: float, league_avg: float) -> dict:
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
# BACKTEST
# ──────────────────────────────────────────────
def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    df["YearMonth"] = df["Date"].dt.to_period("M")
    months = sorted(df["YearMonth"].unique())

    all_rows = []
    print(f"Backtest: {len(months)} meses")

    for i, month in enumerate(months):
        if i < 6:
            continue
        ref_date = month.to_timestamp()
        next_month = (month + 1).to_timestamp()

        train_df = df[df["Date"] < ref_date]
        test_df = df[(df["Date"] >= ref_date) & (df["Date"] < next_month)]
        if len(test_df) == 0:
            continue

        attack, defense, home_adv, league_avg = estimate_strengths_vectorized(train_df, ref_date)

        if not attack:
            continue

        print(f"  {month}: train={len(train_df):,} test={len(test_df):,} teams={len(attack)} avg={league_avg:.2f}")

        for _, row in test_df.iterrows():
            probs = predict_match_fast(row["HomeTeam"], row["AwayTeam"],
                                       attack, defense, home_adv, league_avg)

            for mkt_name, mkt in MARKETS.items():
                odds_col = mkt["odds_col"]
                if odds_col is None or odds_col not in row or pd.isna(row[odds_col]):
                    continue
                odds = row[odds_col]
                if odds < MIN_ODDS or odds > MAX_ODDS:
                    continue

                p_model = probs.get(mkt["prob_key"], 0)
                ev = p_model * odds - 1
                if ev <= MIN_EV_THRESHOLD:
                    continue

                kelly = (p_model * odds - 1) / (odds - 1) if odds > 1 else 0
                stake = max(0, kelly * KELLY_FRACTION)

                # Resultado real
                if mkt["target"] == "Result":
                    actual = int(row["Result"] == mkt["value"])
                elif mkt["target"] == "total_goals":
                    tot = row["FTHG"] + row["FTAG"]
                    actual = int((mkt["value"] == "over" and tot > 2.5) or (mkt["value"] == "under" and tot < 2.5))
                elif mkt["target"] == "btts":
                    btts = int(row["FTHG"] > 0 and row["FTAG"] > 0)
                    actual = int(btts == mkt["value"])
                else:
                    actual = 0

                pnl = stake * (odds - 1) if actual else -stake
                clv = (p_model - 1/odds) * 100

                all_rows.append({
                    "Date": row["Date"], "Season": row["Season"], "Div": row["Div"],
                    "HomeTeam": row["HomeTeam"], "AwayTeam": row["AwayTeam"],
                    "Market": mkt_name, "Model_Prob": p_model, "Odds": odds,
                    "EV": ev, "Stake": stake, "Actual": actual, "PnL": pnl,
                    "CLV": clv, "lambda_H": probs["lambda_h"], "lambda_A": probs["lambda_a"],
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
        f"# Backtest Poisson + Dixon-Coles",
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

    lines.append(f"\n## Top 10 by EV")
    for _, r in results.nlargest(10, "EV").iterrows():
        lines.append(f"- {r['Date'].date()} {r['HomeTeam']} vs {r['AwayTeam']} [{r['Market']}] "
                     f"p={r['Model_Prob']:.3f} o={r['Odds']:.2f} EV={r['EV']:.3f} "
                     f"{'W' if r['Actual'] else 'L'} PnL={r['PnL']:.2f}")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("POISSON + DIXON-COLES BACKTEST (VECTORIZADO)")
    print("=" * 60)

    df = pd.read_parquet(CLEAN_FILE)
    print(f"\n[LOAD] {len(df):,} matches")

    results = run_backtest(df)

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