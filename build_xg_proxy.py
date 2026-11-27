#!/usr/bin/env python
"""
Construye xG Proxy desde stats de football-data (HS, HST, HC, HF, etc.)
Calibra coeficientes con valores de literatura + validación interna.
Salida: data/matches_with_xg.parquet
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

DATA_DIR = Path("data")
CLEAN_FILE = DATA_DIR / "matches_clean.parquet"
OUTPUT_FILE = DATA_DIR / "matches_with_xg.parquet"

# ──────────────────────────────────────────────
# COEFICIENTES DE LITERATURA (puntos de partida)
# ──────────────────────────────────────────────
# Fuente: múltiples papers + modelos públicos (Opta, StatsBomb, etc.)
LIT_COEFFS = {
    # Ataque
    "HST": 0.30,   # Shot on target
    "HS": 0.04,    # Shot off target (HS - HST)
    "HC": 0.03,    # Corner
    "HF": 0.01,    # Foul (proxy tiros libres)
    "HY": -0.005,  # Amarillas (presión defensiva)
    "HR": -0.02,   # Rojas
    # Defensa (recibidos)
    "AST": 0.30,
    "AS": 0.04,
    "AC": 0.03,
    "AF": 0.01,
    "AY": -0.005,
    "AR": -0.02,
}

# Rangos plausibles para calibración
COEFF_BOUNDS = {
    "HST": (0.20, 0.40),
    "HS": (0.01, 0.08),
    "HC": (0.01, 0.06),
    "HF": (0.00, 0.03),
    "AST": (0.20, 0.40),
    "AS": (0.01, 0.08),
    "AC": (0.01, 0.06),
    "AF": (0.00, 0.03),
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye features de ataque/defensa por equipo (rolling)"""
    df = df.sort_values(["HomeTeam", "Date"]).copy()

    # Features base por partido
    df["home_shots_off"] = df["HS"] - df["HST"]
    df["away_shots_off"] = df["AS"] - df["AST"]

    # Rolling features (últimos 5 partidos, home/away separados)
    windows = [3, 5, 10]

    for w in windows:
        # Local como local
        for col in ["FTHG", "HST", "HS", "HC", "HF", "HY", "HR"]:
            df[f"home_{col}_roll{w}"] = df.groupby("HomeTeam")[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
        # Visitante como visitante
        for col in ["FTAG", "AST", "AS", "AC", "AF", "AY", "AR"]:
            df[f"away_{col}_roll{w}"] = df.groupby("AwayTeam")[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
        # Local como visitante (para defensa)
        for col in ["FTAG", "AST", "AS", "AC", "AF", "AY", "AR"]:
            df[f"home_{col}_roll{w}_away"] = df.groupby("HomeTeam")[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
        # Visitante como local (para defensa)
        for col in ["FTHG", "HST", "HS", "HC", "HF", "HY", "HR"]:
            df[f"away_{col}_roll{w}_home"] = df.groupby("AwayTeam")[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

    return df


def xg_proxy_formula(row: pd.Series, coeffs: dict, side: str = "home") -> float:
    """Calcula xG proxy para home o away"""
    if side == "home":
        return (
            coeffs.get("HST", 0.30) * row.get("HST", 0) +
            coeffs.get("HS", 0.04) * row.get("HS", 0) +
            coeffs.get("HC", 0.03) * row.get("HC", 0) +
            coeffs.get("HF", 0.01) * row.get("HF", 0) +
            coeffs.get("HY", -0.005) * row.get("HY", 0) +
            coeffs.get("HR", -0.02) * row.get("HR", 0)
        )
    else:
        return (
            coeffs.get("AST", 0.30) * row.get("AST", 0) +
            coeffs.get("AS", 0.04) * row.get("AS", 0) +
            coeffs.get("AC", 0.03) * row.get("AC", 0) +
            coeffs.get("AF", 0.01) * row.get("AF", 0) +
            coeffs.get("AY", -0.005) * row.get("AY", 0) +
            coeffs.get("AR", -0.02) * row.get("AR", 0)
        )


def calibrate_with_goals(df: pd.DataFrame) -> dict:
    """
    Calibra coeficientes usando goles reales como proxy de xG.
    Asume: goles ≈ xG a largo plazo (ley de grandes números).
    Usa regression: FTHG ~ HST + HS + HC + HF + HY + HR (ponderado por tiempo)
    """
    df = df.dropna(subset=["FTHG", "FTAG", "HST", "AST", "HS", "AS", "HC", "AC", "HF", "AF"]).copy()

    # Pesos temporales (recientes pesan más)
    ref_date = df["Date"].max()
    days_diff = (ref_date - df["Date"]).dt.days.clip(lower=0)
    weights = np.power(0.5, days_diff / 150)

    # Modelo local
    X_home = df[["HST", "HS", "HC", "HF", "HY", "HR"]].fillna(0)
    y_home = df["FTHG"]

    # Modelo visitante
    X_away = df[["AST", "AS", "AC", "AF", "AY", "AR"]].fillna(0)
    y_away = df["FTAG"]

    # Regresión ponderada
    reg_home = LinearRegression().fit(X_home, y_home, sample_weight=weights)
    reg_away = LinearRegression().fit(X_away, y_away, sample_weight=weights)

    coeffs = {
        "HST": max(0.15, min(0.45, reg_home.coef_[0])),
        "HS": max(0.00, min(0.10, reg_home.coef_[1])),
        "HC": max(0.00, min(0.08, reg_home.coef_[2])),
        "HF": max(0.00, min(0.05, reg_home.coef_[3])),
        "HY": min(0.00, max(-0.02, reg_home.coef_[4])),
        "HR": min(0.00, max(-0.05, reg_home.coef_[5])),
        "AST": max(0.15, min(0.45, reg_away.coef_[0])),
        "AS": max(0.00, min(0.10, reg_away.coef_[1])),
        "AC": max(0.00, min(0.08, reg_away.coef_[2])),
        "AF": max(0.00, min(0.05, reg_away.coef_[3])),
        "AY": min(0.00, max(-0.02, reg_away.coef_[4])),
        "AR": min(0.00, max(-0.05, reg_away.coef_[5])),
    }

    # R² scores
    r2_home = reg_home.score(X_home, y_home, sample_weight=weights)
    r2_away = reg_away.score(X_away, y_away, sample_weight=weights)

    return coeffs, r2_home, r2_away


def main():
    print("=" * 60)
    print("BUILDING xG PROXY FROM MATCH STATS")
    print("=" * 60)

    df = pd.read_parquet(CLEAN_FILE)
    print(f"\n[LOAD] {len(df):,} matches")

    # Calibra coeficientes
    print("\n[CALIBRATE] Regressing goals on shots/stats...")
    coeffs, r2_h, r2_a = calibrate_with_goals(df)

    print(f"\nCalibrated coefficients:")
    for k, v in coeffs.items():
        print(f"  {k}: {v:.4f} (lit: {LIT_COEFFS.get(k, 'N/A')})")
    print(f"R² Home: {r2_h:.3f}, R² Away: {r2_a:.3f}")

    # Aplica xG proxy a cada partido
    print("\n[APPLY] Computing xG proxy per match...")
    df["home_xg_proxy"] = df.apply(lambda r: xg_proxy_formula(r, coeffs, "home"), axis=1)
    df["away_xg_proxy"] = df.apply(lambda r: xg_proxy_formula(r, coeffs, "away"), axis=1)

    # xG concedido (xGA)
    df["home_xga_proxy"] = df["away_xg_proxy"]
    df["away_xga_proxy"] = df["home_xg_proxy"]

    # Clip a rangos razonables
    df["home_xg_proxy"] = df["home_xg_proxy"].clip(0.2, 3.5)
    df["away_xg_proxy"] = df["away_xg_proxy"].clip(0.2, 3.5)
    df["home_xga_proxy"] = df["home_xga_proxy"].clip(0.2, 3.5)
    df["away_xga_proxy"] = df["away_xga_proxy"].clip(0.2, 3.5)

    # Build rolling xG features
    print("\n[FEATURES] Building rolling xG features...")
    df = build_features(df)

    # Rolling xG (últimos 5)
    df["home_xg_roll5"] = df.groupby("HomeTeam")["home_xg_proxy"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["away_xg_roll5"] = df.groupby("AwayTeam")["away_xg_proxy"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["home_xga_roll5"] = df.groupby("HomeTeam")["home_xga_proxy"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["away_xga_roll5"] = df.groupby("AwayTeam")["away_xga_proxy"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Diff features (ataque vs defensa rival)
    df["xg_diff_roll5"] = df["home_xg_roll5"] - df["away_xga_roll5"]
    df["xga_diff_roll5"] = df["away_xg_roll5"] - df["home_xga_roll5"]

    # Guarda
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n[SAVE] {OUTPUT_FILE} ({len(df):,} matches, {len(df.columns)} cols)")

    # Stats
    print(f"\nxG Proxy stats:")
    print(f"  Home xG mean: {df['home_xg_proxy'].mean():.3f}")
    print(f"  Away xG mean: {df['away_xg_proxy'].mean():.3f}")
    print(f"  Home xGA mean: {df['home_xga_proxy'].mean():.3f}")
    print(f"  Away xGA mean: {df['away_xga_proxy'].mean():.3f}")
    print(f"  Correlation xG_proxy vs Goals (home): {df['home_xg_proxy'].corr(df['FTHG']):.3f}")
    print(f"  Correlation xG_proxy vs Goals (away): {df['away_xg_proxy'].corr(df['FTAG']):.3f}")

    # Sample
    print("\nSample (last 5):")
    cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "home_xg_proxy", "away_xg_proxy"]
    print(df[cols].tail())


if __name__ == "__main__":
    main()