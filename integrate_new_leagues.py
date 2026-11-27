#!/usr/bin/env python
"""
Integra ligas nuevas (N1, D2, B1) al dataset clean principal
Aplica misma limpieza que download_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

# Columnas objetivo (igual que matches_clean.parquet)
TARGET_COLS = [
    "Div", "Season", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5",
    "B365AHH", "B365AHA",
    "Result", "imp_prob_H", "imp_prob_D", "imp_prob_A", "overround",
    "imp_prob_O25", "imp_prob_U25",
]

def clean_raw_league(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza estándar a liga raw."""
    df = df_raw.copy()

    # 1. Fecha
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])

    # 2. Goles
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])

    # 3. Result (numérico: 1=Home, 0=Draw, -1=Away)
    if "Result" not in df.columns:
        df["Result"] = df["FTR"].map({"H": 1, "D": 0, "A": -1})

    # 4. Half-time stats (pueden no existir en ligas menores)
    for col in ["HTHG", "HTAG", "HTR"]:
        if col not in df.columns:
            df[col] = np.nan

    # 5. Asian handicap (pueden no existir)
    for col in ["B365AHH", "B365AHA"]:
        if col not in df.columns:
            df[col] = np.nan

    # 6. Stats de juego - fill NaN con 0
    stat_cols = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]
    for c in stat_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0

    # 7. Odds - quitar vigorish
    if all(c in df.columns for c in ["B365H", "B365D", "B365A"]):
        df = df.dropna(subset=["B365H", "B365D", "B365A"])
        inv = 1/df["B365H"] + 1/df["B365D"] + 1/df["B365A"]
        df["imp_prob_H"] = (1/df["B365H"]) / inv
        df["imp_prob_D"] = (1/df["B365D"]) / inv
        df["imp_prob_A"] = (1/df["B365A"]) / inv
        df["overround"] = inv - 1
    else:
        df["imp_prob_H"] = df["imp_prob_D"] = df["imp_prob_A"] = df["overround"] = np.nan

    # 8. Over/Under 2.5 odds
    if all(c in df.columns for c in ["B365>2.5", "B365<2.5"]):
        df = df.dropna(subset=["B365>2.5", "B365<2.5"])
        inv_o = 1/df["B365>2.5"] + 1/df["B365<2.5"]
        df["imp_prob_O25"] = (1/df["B365>2.5"]) / inv_o
        df["imp_prob_U25"] = (1/df["B365<2.5"]) / inv_o
    else:
        df["imp_prob_O25"] = df["imp_prob_U25"] = np.nan

    # 9. Columnas derivadas (no guardar, se recalculan en features)
    if "total_goals" in df.columns:
        df = df.drop(columns=["total_goals"])
    if "over_25" in df.columns:
        df = df.drop(columns=["over_25"])
    if "btts" in df.columns:
        df = df.drop(columns=["btts"])

    # 10. Seleccionar y ordenar columnas objetivo
    # Solo las que existen
    final_cols = [c for c in TARGET_COLS if c in df.columns]
    df = df[final_cols]

    return df

def main():
    print("=" * 60)
    print("INTEGRANDO LIGAS NUEVAS: N1 (Eredivisie), D2 (2. Bund), B1 (Bélgica)")
    print("=" * 60)

    # 1. Cargar dataset clean existente
    existing = pd.read_parquet(DATA_DIR / "matches_clean.parquet")
    print(f"\n[EXISTING] {len(existing):,} partidos | Ligas: {sorted(existing['Div'].unique())}")

    # 2. Cargar y limpiar ligas nuevas
    new_leagues = ["N1", "D2", "B1"]
    new_dfs = []

    for code in new_leagues:
        raw_file = DATA_DIR / "raw_all_leagues" / f"{code}_raw.parquet"
        df_raw = pd.read_parquet(raw_file)
        print(f"\n[RAW {code}] {len(df_raw):,} partidos")
        df_clean = clean_raw_league(df_raw)
        print(f"[CLEAN {code}] {len(df_clean):,} partidos")
        new_dfs.append(df_clean)

    # 3. Combinar todo
    combined = pd.concat([existing] + new_dfs, ignore_index=True)
    combined = combined.sort_values(["Div", "Date"]).reset_index(drop=True)

    print(f"\n[COMBINADO] {len(combined):,} partidos")
    print(f"Ligas: {sorted(combined['Div'].unique())}")
    print(f"Temporadas: {sorted(combined['Season'].unique())}")

    # 4. Verificar integridad
    print(f"\nColumnas finales: {len(combined.columns)}")
    print(f"NaN por columna:")
    nan_counts = combined.isnull().sum()
    print(nan_counts[nan_counts > 0])

    # 5. Guardar backup del original
    backup = DATA_DIR / "matches_clean_backup_top5.parquet"
    existing.to_parquet(backup, index=False)
    print(f"\n[BACKUP] Original guardado en {backup}")

    # 6. Guardar nuevo dataset unificado
    combined.to_parquet(DATA_DIR / "matches_clean.parquet", index=False)
    print(f"[SAVE] Nuevo matches_clean.parquet: {len(combined):,} partidos")

    # 7. Resumen por liga
    print("\n=== RESUMEN POR LIGA ===")
    for div in sorted(combined["Div"].unique()):
        sub = combined[combined["Div"] == div]
        print(f"  {div}: {len(sub):,} partidos | {sub['Season'].min()} a {sub['Season'].max()}")

    print("\n✅ INTEGRACIÓN COMPLETADA")

if __name__ == "__main__":
    main()