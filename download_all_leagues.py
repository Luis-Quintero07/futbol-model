#!/usr/bin/env python
"""
Descarga TODAS las ligas disponibles en football-data.co.uk (2018-2025)
Analiza patrones estadísticos: Over 2.5, BTTS, Home Win %, Draw %, Away Win %
Identifica ligas con edge explotable
"""
import os
import re
from pathlib import Path
import pandas as pd
import requests
from io import StringIO

# ──────────────────────────────────────────────
# TODAS LAS LIGAS DISPONIBLES EN FOOTBALL-DATA
# ──────────────────────────────────────────────
ALL_LEAGUES = {
    # Top 5 (ya tenemos)
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    # Segunda divisiones principales
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "SP2": "La Liga 2",
    "D2": "2. Bundesliga",
    "I2": "Serie B",
    "F2": "Ligue 2",
    # Otras europeas top
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
    "B1": "Jupiler Pro League",
    "T1": "Süper Lig",
    "G1": "Super League Greece",
    "A1": "Bundesliga Austria",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    # Resto del mundo
    "USA": "MLS",
    "ARG": "Primera División Argentina",
    "BRA": "Brasileirão",
    "MEX": "Liga MX",
    "JPN": "J1 League",
    "KOR": "K League 1",
    "CHN": "Chinese Super League",
    "CL": "Champions League",
    "EL": "Europa League",
    "EC": "European Championship",
    "WC": "World Cup",
}

SEASONS = [
    (2018, "2018-19"),
    (2019, "2019-20"),
    (2020, "2020-21"),
    (2021, "2021-22"),
    (2022, "2022-23"),
    (2023, "2023-24"),
    (2024, "2024-25"),
]

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_all_leagues"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def season_code(year):
    """2018 -> '1819'"""
    return f"{str(year)[2:]}{str(year+1)[2:]}"

def download_league_season(league_code, year):
    """Descarga una liga-temporada específica."""
    code = season_code(year)
    url = f"https://www.football-data.co.uk/mmz4281/{code}/{league_code}.csv"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        content = resp.content.decode("latin1")
        df = pd.read_csv(StringIO(content))
        df["Season"] = f"{year}-{year+1}"
        df["Div"] = league_code
        return df
    except Exception as e:
        print(f"  [ERR] {league_code} {year}: {e}")
        return None

def clean_dataframe(df):
    """Limpieza básica: columnas clave, quita vigorish."""
    # Columnas que nos importan
    keep = [
        "Div", "Season", "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
        "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5",
    ]
    # Solo columnas que existen
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Fecha
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])

    # Goles
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])

    # Total goles, BTTS, Over 2.5
    df["total_goals"] = df["FTHG"] + df["FTAG"]
    df["btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
    df["over_25"] = (df["total_goals"] > 2.5).astype(int)

    # Odds -> prob implícitas sin vigorish
    if all(c in df.columns for c in ["B365H", "B365D", "B365A"]):
        df = df.dropna(subset=["B365H", "B365D", "B365A"])
        inv = 1/df["B365H"] + 1/df["B365D"] + 1/df["B365A"]
        df["imp_prob_H"] = (1/df["B365H"]) / inv
        df["imp_prob_D"] = (1/df["B365D"]) / inv
        df["imp_prob_A"] = (1/df["B365A"]) / inv
        df["overround"] = inv - 1

    if all(c in df.columns for c in ["B365>2.5", "B365<2.5"]):
        df = df.dropna(subset=["B365>2.5", "B365<2.5"])
        inv_o = 1/df["B365>2.5"] + 1/df["B365<2.5"]
        df["imp_prob_O25"] = (1/df["B365>2.5"]) / inv_o
        df["imp_prob_U25"] = (1/df["B365<2.5"]) / inv_o

    return df

def analyze_league_patterns(df, league_code, league_name):
    """Analiza patrones estadísticos por liga y temporada."""
    results = []

    for season in sorted(df["Season"].unique()):
        sdf = df[df["Season"] == season]
        n = len(sdf)
        if n < 50:
            continue

        # Estadísticas básicas
        home_win = (sdf["FTR"] == "H").mean()
        draw = (sdf["FTR"] == "D").mean()
        away_win = (sdf["FTR"] == "A").mean()
        over25 = sdf["over_25"].mean()
        btts = sdf["btts"].mean()
        avg_goals = sdf["total_goals"].mean()
        avg_h_goals = sdf["FTHG"].mean()
        avg_a_goals = sdf["FTAG"].mean()

        # Odds implícitas (si disponibles)
        if "imp_prob_O25" in sdf.columns:
            mkt_over25 = sdf["imp_prob_O25"].mean()
            edge_over25 = over25 - mkt_over25
        else:
            mkt_over25 = edge_over25 = None

        if "imp_prob_H" in sdf.columns:
            mkt_home = sdf["imp_prob_H"].mean()
            edge_home = home_win - mkt_home
        else:
            mkt_home = edge_home = None

        results.append({
            "league_code": league_code,
            "league_name": league_name,
            "season": season,
            "matches": n,
            "home_win_pct": round(home_win * 100, 1),
            "draw_pct": round(draw * 100, 1),
            "away_win_pct": round(away_win * 100, 1),
            "over25_pct": round(over25 * 100, 1),
            "btts_pct": round(btts * 100, 1),
            "avg_goals": round(avg_goals, 2),
            "avg_h_goals": round(avg_h_goals, 2),
            "avg_a_goals": round(avg_a_goals, 2),
            "mkt_over25_pct": round(mkt_over25 * 100, 1) if mkt_over25 else None,
            "edge_over25_pct": round(edge_over25 * 100, 1) if edge_over25 else None,
            "mkt_home_pct": round(mkt_home * 100, 1) if mkt_home else None,
            "edge_home_pct": round(edge_home * 100, 1) if edge_home else None,
        })

    return results

def main():
    print("=" * 70)
    print("DESCARGA Y ANÁLISIS DE TODAS LAS LIGAS FOOTBALL-DATA.CO.UK")
    print("=" * 70)

    all_dfs = []
    all_patterns = []

    for league_code, league_name in ALL_LEAGUES.items():
        print(f"\n{'='*70}")
        print(f"LIGA: {league_name} ({league_code})")
        print(f"{'='*70}")

        league_dfs = []
        for year, season_name in SEASONS:
            print(f"  Descargando {season_name}...", end=" ")
            df = download_league_season(league_code, year)
            if df is not None and len(df) > 0:
                df = clean_dataframe(df)
                if len(df) > 0:
                    league_dfs.append(df)
                    print(f"[OK] {len(df)} partidos")
                else:
                    print("[VACÍO]")
            else:
                print("[NO DISPONIBLE]")

        if not league_dfs:
            print(f"  ⚠️  Sin datos para {league_name}")
            continue

        # Combina todas las temporadas de esta liga
        league_df = pd.concat(league_dfs, ignore_index=True)
        print(f"  Total partidos: {len(league_df)}")

        # Guarda raw por liga
        league_file = RAW_DIR / f"{league_code}_raw.parquet"
        league_df.to_parquet(league_file, index=False)
        print(f"  [SAVE] {league_file}")

        # Analiza patrones
        patterns = analyze_league_patterns(league_df, league_code, league_name)
        all_patterns.extend(patterns)

        # Resumen rápido por consola
        for p in patterns:
            edge_str = ""
            if p["edge_over25_pct"] is not None:
                edge_str += f" | Edge O2.5: {p['edge_over25_pct']:+.1f}%"
            if p["edge_home_pct"] is not None:
                edge_str += f" | Edge Home: {p['edge_home_pct']:+.1f}%"
            print(f"    {p['season']}: H:{p['home_win_pct']}% D:{p['draw_pct']}% A:{p['away_win_pct']}% | O2.5:{p['over25_pct']}% BTTS:{p['btts_pct']}% | AvgG:{p['avg_goals']}{edge_str}")

        all_dfs.append(league_df)

    # ──────────────────────────────────────────────
    # RESUMEN GLOBAL DE PATRONES
    # ──────────────────────────────────────────────
    if all_patterns:
        patterns_df = pd.DataFrame(all_patterns)
        patterns_file = DATA_DIR / "league_patterns_analysis.parquet"
        patterns_df.to_parquet(patterns_file, index=False)
        print(f"\n{'='*70}")
        print(f"[SAVE] Análisis de patrones: {patterns_file}")
        print(f"{'='*70}")

        # Top ligas por Over 2.5% (promedio últimas 3 temporadas)
        print("\n🏆 TOP 10 LIGAS POR OVER 2.5% (promedio 2022-2025):")
        recent = patterns_df[patterns_df["season"].isin(["2022-23", "2023-24", "2024-25"])]
        top_over25 = recent.groupby("league_name")["over25_pct"].mean().sort_values(ascending=False).head(10)
        for i, (league, val) in enumerate(top_over25.items(), 1):
            print(f"  {i:2d}. {league}: {val:.1f}%")

        # Top ligas por BTTS%
        print("\n🏆 TOP 10 LIGAS POR BTTS% (promedio 2022-2025):")
        top_btts = recent.groupby("league_name")["btts_pct"].mean().sort_values(ascending=False).head(10)
        for i, (league, val) in enumerate(top_btts.items(), 1):
            print(f"  {i:2d}. {league}: {val:.1f}%")

        # Top ligas por Home Win%
        print("\n🏆 TOP 10 LIGAS POR HOME WIN% (promedio 2022-2025):")
        top_home = recent.groupby("league_name")["home_win_pct"].mean().sort_values(ascending=False).head(10)
        for i, (league, val) in enumerate(top_home.items(), 1):
            print(f"  {i:2d}. {league}: {val:.1f}%")

        # Ligas con EDGE POSITIVO en Over 2.5 (mercado subestima)
        print("\n💰 LIGAS CON EDGE POSITIVO EN OVER 2.5 (promedio 2022-2025):")
        edge_over = recent[recent["edge_over25_pct"].notna()].groupby("league_name")["edge_over25_pct"].mean().sort_values(ascending=False)
        for league, val in edge_over.items():
            if val > 0:
                print(f"  {league}: {val:+.1f}% edge")

        # Ligas con EDGE POSITIVO en Home Win
        print("\n💰 LIGAS CON EDGE POSITIVO EN HOME WIN (promedio 2022-2025):")
        edge_home = recent[recent["edge_home_pct"].notna()].groupby("league_name")["edge_home_pct"].mean().sort_values(ascending=False)
        for league, val in edge_home.items():
            if val > 0:
                print(f"  {league}: {val:+.1f}% edge")

    # ──────────────────────────────────────────────
    # DATASET UNIFICADO
    # ──────────────────────────────────────────────
    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df = full_df.sort_values(["Div", "Date"]).reset_index(drop=True)
        full_file = DATA_DIR / "matches_all_leagues.parquet"
        full_df.to_parquet(full_file, index=False)
        print(f"\n[SAVE] Dataset unificado: {full_file} ({len(full_df):,} partidos)")

    print("\n✅ COMPLETADO")

if __name__ == "__main__":
    main()