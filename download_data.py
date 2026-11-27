#!/usr/bin/env python
"""
Descarga y limpieza de datos históricos de football-data.co.uk
Ligas: EPL, LaLiga, Bundesliga, Serie A, Ligue 1, UCL
Temporadas: 2018/19 a 2024/25
Salida: data/matches_raw.parquet, data/matches_clean.parquet
"""
import os
import re
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
BASE_URL = "https://www.football-data.co.uk/mmz4281/"

# (season_code, league_code, league_name)
# season_code: 1819 = 2018/19, 2425 = 2024/25
LEAGUES = [
    ("E0", "Premier League"),
    ("SP1", "La Liga"),
    ("D1", "Bundesliga"),
    ("I1", "Serie A"),
    ("F1", "Ligue 1"),
    # Champions League: código CL, pero football-data lo maneja distinto
]

SEASONS = [
    ("1819", "2018-2019"),
    ("1920", "2019-2020"),
    ("2021", "2020-2021"),
    ("2122", "2021-2022"),
    ("2223", "2022-2023"),
    ("2324", "2023-2024"),
    ("2425", "2024-2025"),  # en curso
]

# Columnas que nos importan (football-data tiene ~50, muchas vacías en ligas menores)
KEEP_COLS = [
    "Div", "Season", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",           # Full Time Home/Away Goals, Result (H/D/A)
    "HTHG", "HTAG", "HTR",           # Half Time
    "HS", "AS", "HST", "AST",        # Shots, Shots on Target
    "HF", "AF", "HC", "AC",          # Fouls, Corners
    "HY", "AY", "HR", "AR",          # Yellow/Red cards
    "B365H", "B365D", "B365A",       # Bet365 odds 1X2 (línea de cierre ~)
    "B365>2.5", "B365<2.5",          # Over/Under 2.5
    "B365AHH", "B365AHA",            # Asian Handicap (si existe)
]

# Estandarización de nombres de equipos (football-data usa abreviaciones)
TEAM_NAME_MAP = {
    # Premier League
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "Wolves": "Wolverhampton Wanderers",
    "Brighton": "Brighton & Hove Albion",
    "West Ham": "West Ham United",
    "Sheffield United": "Sheffield United",
    "Nott'm Forest": "Nottingham Forest",
    "Leicester": "Leicester City",
    # La Liga
    "Ath Madrid": "Atletico Madrid",
    "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona",
    "Valencia": "Valencia",
    "Villarreal": "Villarreal",
    "Real Betis": "Real Betis",
    "Sevilla": "Sevilla",
    "Ath Bilbao": "Athletic Bilbao",
    "Celta Vigo": "Celta Vigo",
    "Getafe": "Getafe",
    "Alaves": "Alaves",
    "Eibar": "Eibar",
    "Leganes": "Leganes",
    "Huesca": "Huesca",
    "Valladolid": "Valladolid",
    "Mallorca": "Mallorca",
    "Granada": "Granada",
    "Cadiz": "Cadiz",
    "Elche": "Elche",
    "Rayo Vallecano": "Rayo Vallecano",
    "Osasuna": "Osasuna",
    "Girona": "Girona",
    "Las Palmas": "Las Palmas",
    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Dortmund": "Borussia Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Leverkusen": "Bayer Leverkusen",
    "M'gladbach": "Borussia Monchengladbach",
    "Wolfsburg": "Wolfsburg",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Freiburg": "Freiburg",
    "Union Berlin": "Union Berlin",
    "Mainz": "Mainz",
    "Hoffenheim": "Hoffenheim",
    "Augsburg": "Augsburg",
    "Stuttgart": "Stuttgart",
    "Werder Bremen": "Werder Bremen",
    "Bochum": "Bochum",
    "Heidenheim": "Heidenheim",
    "Darmstadt": "Darmstadt",
    "St Pauli": "St Pauli",
    "Holstein Kiel": "Holstein Kiel",
    # Serie A
    "Juventus": "Juventus",
    "Inter": "Inter Milan",
    "AC Milan": "AC Milan",
    "Napoli": "Napoli",
    "Roma": "Roma",
    "Lazio": "Lazio",
    "Atalanta": "Atalanta",
    "Fiorentina": "Fiorentina",
    "Torino": "Torino",
    "Bologna": "Bologna",
    "Udinese": "Udinese",
    "Sassuolo": "Sassuolo",
    "Verona": "Verona",
    "Genoa": "Genoa",
    "Cagliari": "Cagliari",
    "Empoli": "Empoli",
    "Salernitana": "Salernitana",
    "Monza": "Monza",
    "Frosinone": "Frosinone",
    "Como": "Como",
    "Venezia": "Venezia",
    "Parma": "Parma",
    # Ligue 1
    "PSG": "Paris Saint-Germain",
    "Marseille": "Marseille",
    "Lyon": "Lyon",
    "Monaco": "Monaco",
    "Lille": "Lille",
    "Rennes": "Rennes",
    "Nice": "Nice",
    "Lens": "Lens",
    "Reims": "Reims",
    "Montpellier": "Montpellier",
    "Toulouse": "Toulouse",
    "Brest": "Brest",
    "Strasbourg": "Strasbourg",
    "Nantes": "Nantes",
    "Auxerre": "Auxerre",
    "Angers": "Angers",
    "Le Havre": "Le Havre",
    "Saint-Etienne": "Saint-Etienne",
}

DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "matches_raw.parquet"
CLEAN_FILE = DATA_DIR / "matches_clean.parquet"
DICT_FILE = DATA_DIR / "data_dictionary.md"


# ──────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────
def season_code_to_str(code: str) -> str:
    """1819 -> 2018-2019"""
    return f"20{code[:2]}-20{code[2:]}"


def standardize_team_name(name: str) -> str:
    """Normaliza nombres de equipos"""
    if pd.isna(name):
        return name
    name = str(name).strip()
    return TEAM_NAME_MAP.get(name, name)


def download_season(season_code: str, league_code: str) -> pd.DataFrame | None:
    """Descarga un CSV de una liga+temporada"""
    url = urljoin(BASE_URL, f"{season_code}/{league_code}.csv")
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"  [WARN] {league_code} {season_code}: HTTP {resp.status_code}")
            return None
        # football-data usa encoding latin1 a veces
        content = resp.content.decode("latin1")
        df = pd.read_csv(pd.io.common.StringIO(content))
        df["Season"] = season_code_to_str(season_code)
        df["Div"] = league_code
        print(f"  [OK] {league_code} {season_code}: {len(df)} partidos")
        return df
    except Exception as e:
        print(f"  [ERR] {league_code} {season_code}: {e}")
        return None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza y normalización"""
    df = df.copy()

    # 1. Solo columnas que existen + nos interesan
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].copy()

    # 2. Nombres de equipos
    for col in ["HomeTeam", "AwayTeam"]:
        if col in df.columns:
            df[col] = df[col].apply(standardize_team_name)

    # 3. Fecha
    if "Date" in df.columns:
        # football-data usa DD/MM/YY o DD/MM/YYYY
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # 4. Resultado numérico
    if "FTR" in df.columns:
        df["Result"] = df["FTR"].map({"H": 1, "D": 0, "A": -1})  # 1=Home, 0=Draw, -1=Away

    # 5. Odds a numérico (algunas vienen vacías o con strings raros)
    odds_cols = [c for c in df.columns if c.startswith("B365")]
    for col in odds_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Goles a entero
    for col in ["FTHG", "FTAG", "HTHG", "HTAG"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # 7. Elimina filas sin datos esenciales
    essential = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    df = df.dropna(subset=[c for c in essential if c in df.columns])

    # 8. Elimina partidos sin odds 1X2 (cancelados, etc.)
    if all(c in df.columns for c in ["B365H", "B365D", "B365A"]):
        df = df.dropna(subset=["B365H", "B365D", "B365A"])

    # 9. Quita vigorish (margen casa) a probabilidades implícitas "justas"
    if all(c in df.columns for c in ["B365H", "B365D", "B365A"]):
        inv = 1/df["B365H"] + 1/df["B365D"] + 1/df["B365A"]
        df["imp_prob_H"] = (1/df["B365H"]) / inv
        df["imp_prob_D"] = (1/df["B365D"]) / inv
        df["imp_prob_A"] = (1/df["B365A"]) / inv
        df["overround"] = inv - 1  # margen de la casa

    # 10. Ordena por fecha
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def create_data_dictionary(df: pd.DataFrame) -> str:
    """Genera diccionario de datos en markdown"""
    lines = ["# Data Dictionary", "", f"Total rows: {len(df):,}", f"Columns: {len(df.columns)}", "", "| Column | Type | Description |", "|--------|------|-------------|"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        pct = non_null / len(df) * 100
        # Descripción breve
        desc = {
            "Div": "Código de liga (E0, SP1, D1, I1, F1)",
            "Season": "Temporada (ej. 2023-2024)",
            "Date": "Fecha del partido",
            "HomeTeam": "Equipo local (nombre estandarizado)",
            "AwayTeam": "Equipo visitante",
            "FTHG": "Goles local tiempo completo",
            "FTAG": "Goles visitante tiempo completo",
            "FTR": "Resultado original (H/D/A)",
            "Result": "Resultado numérico (1/0/-1)",
            "HTHG": "Goles local medio tiempo",
            "HTAG": "Goles visitante medio tiempo",
            "HS": "Tiros local",
            "AS": "Tiros visitante",
            "HST": "Tiros a puerta local",
            "AST": "Tiros a puerta visitante",
            "HC": "Corners local",
            "AC": "Corners visitante",
            "HF": "Faltas local",
            "AF": "Faltas visitante",
            "HY": "Amarillas local",
            "AY": "Amarillas visitante",
            "HR": "Rojas local",
            "AR": "Rojas visitante",
            "B365H": "Odds Bet365 victoria local",
            "B365D": "Odds Bet365 empate",
            "B365A": "Odds Bet365 victoria visitante",
            "B365>2.5": "Odds Bet365 Over 2.5 goles",
            "B365<2.5": "Odds Bet365 Under 2.5 goles",
            "imp_prob_H": "Prob. implícita local (sin vigorish)",
            "imp_prob_D": "Prob. implícita empate (sin vigorish)",
            "imp_prob_A": "Prob. implícita visitante (sin vigorish)",
            "overround": "Margen casa (vigorish)",
        }.get(col, "")
        lines.append(f"| {col} | {dtype} | {desc} ({pct:.1f}% completado) |")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("DESCARGA DATOS FOOTBALL-DATA.CO.UK")
    print("=" * 60)

    all_dfs = []

    for season_code, season_name in SEASONS:
        print(f"\n[SEASON] Temporada {season_name} ({season_code})")
        for league_code, league_name in LEAGUES:
            print(f"  [DOWNLOAD]  {league_name} ({league_code})...", end=" ")
            df = download_season(season_code, league_code)
            if df is not None and len(df) > 0:
                if "Season" in df.columns:
                    all_dfs.append(df)
                else:
                    print(f"     [WARN] Sin columna Season, columnas: {list(df.columns)}")
            else:
                print(f"     [SKIP] Sin datos")

    if not all_dfs:
        print("\n[ERR] No se descargaron datos. Revisa conexion / URLs.")
        return

    # Concatena todo
    print("\n[MERGE] Unificando datasets...")
    raw_df = pd.concat(all_dfs, ignore_index=True)
    print(f"   Total partidos crudos: {len(raw_df):,}")
    print(f"   Columnas raw: {list(raw_df.columns)}")
    if "Season" in raw_df.columns:
        print(f"   Temporadas: {raw_df['Season'].unique()}")
    else:
        print(f"   [ERR] No hay columna Season en raw_df!")

    # Guarda raw
    raw_df.to_parquet(RAW_FILE, index=False)
    print(f"   [SAVE] Guardado: {RAW_FILE} ({RAW_FILE.stat().st_size / 1e6:.1f} MB)")

    # Limpieza
    print("\n[CLEAN] Limpiando y normalizando...")
    clean_df = clean_dataframe(raw_df)
    print(f"   Partidos tras limpieza: {len(clean_df):,}")
    print(f"   Columnas: {list(clean_df.columns)}")

    # Guarda clean
    clean_df.to_parquet(CLEAN_FILE, index=False)
    print(f"   [SAVE] Guardado: {CLEAN_FILE} ({CLEAN_FILE.stat().st_size / 1e6:.1f} MB)")

    # Diccionario de datos
    dict_md = create_data_dictionary(clean_df)
    DICT_FILE.write_text(dict_md, encoding="utf-8")
    print(f"   [DICT] Diccionario: {DICT_FILE}")

    # Resumen rápido
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Temporadas: {clean_df['Season'].nunique()} ({clean_df['Season'].min()} a {clean_df['Season'].max()})")
    print(f"Ligas: {clean_df['Div'].nunique()} ({', '.join(sorted(clean_df['Div'].unique()))})")
    print(f"Partidos: {len(clean_df):,}")
    print(f"Equipos únicos: {pd.concat([clean_df['HomeTeam'], clean_df['AwayTeam']]).nunique()}")
    print(f"Rango fechas: {clean_df['Date'].min().date()} a {clean_df['Date'].max().date()}")
    if "overround" in clean_df.columns:
        print(f"Overround medio: {clean_df['overround'].mean():.2%}")
    print("\n[DONE] ¡Listo para modelar!")


if __name__ == "__main__":
    main()