#!/usr/bin/env python
"""
Scraper xG de Understat.com (sin JS - usa requests + regex)
Ligas: EPL, LaLiga, Bundesliga, Serie A, Ligue 1
Temporadas: 2018/19 → 2024/25
"""
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path("data")
XG_FILE = DATA_DIR / "xg_matches.parquet"

LEAGUES = {
    "EPL": "epl",
    "La_liga": "la_liga",
    "Bundesliga": "bundesliga",
    "Serie_A": "serie_a",
    "Ligue_1": "ligue_1",
}

SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

UNDERSTAT_TO_STD = {
    # EPL
    "Manchester United": "Manchester United",
    "Manchester City": "Manchester City",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "Liverpool": "Liverpool",
    "Chelsea": "Chelsea",
    "Arsenal": "Arsenal",
    "Newcastle United": "Newcastle United",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "West Ham United": "West Ham United",
    "Aston Villa": "Aston Villa",
    "Crystal Palace": "Crystal Palace",
    "Fulham": "Fulham",
    "Brentford": "Brentford",
    "Nottingham Forest": "Nottingham Forest",
    "AFC Bournemouth": "Bournemouth",
    "Everton": "Everton",
    "Leicester City": "Leicester City",
    "Southampton": "Southampton",
    "Leeds United": "Leeds United",
    "Burnley": "Burnley",
    "Watford": "Watford",
    "Norwich City": "Norwich City",
    "Sheffield United": "Sheffield United",
    "Ipswich Town": "Ipswich Town",
    "Luton Town": "Luton Town",
    # La Liga
    "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona",
    "Atletico Madrid": "Atletico Madrid",
    "Sevilla": "Sevilla",
    "Villarreal": "Villarreal",
    "Real Betis": "Real Betis",
    "Athletic Bilbao": "Athletic Bilbao",
    "Real Sociedad": "Real Sociedad",
    "Valencia": "Valencia",
    "Celta Vigo": "Celta Vigo",
    "Getafe": "Getafe",
    "Osasuna": "Osasuna",
    "Rayo Vallecano": "Rayo Vallecano",
    "Espanyol": "Espanyol",
    "Mallorca": "Mallorca",
    "Alaves": "Alaves",
    "Granada": "Granada",
    "Cadiz": "Cadiz",
    "Elche": "Elche",
    "Levante": "Levante",
    "Huesca": "Huesca",
    "Valladolid": "Valladolid",
    "Girona": "Girona",
    "Las Palmas": "Las Palmas",
    "Almeria": "Almeria",
    "Leganes": "Leganes",
    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Borussia Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "Borussia Monchengladbach": "Borussia Monchengladbach",
    "VfL Wolfsburg": "Wolfsburg",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "SC Freiburg": "Freiburg",
    "Union Berlin": "Union Berlin",
    "FSV Mainz 05": "Mainz",
    "TSG Hoffenheim": "Hoffenheim",
    "FC Augsburg": "Augsburg",
    "VfB Stuttgart": "Stuttgart",
    "Werder Bremen": "Werder Bremen",
    "VfL Bochum": "Bochum",
    "FC Koln": "Koln",
    "Hertha Berlin": "Hertha Berlin",
    "Arminia Bielefeld": "Bielefeld",
    "Greuther Furth": "Greuther Furth",
    "Darmstadt 98": "Darmstadt",
    "Heidenheim": "Heidenheim",
    "St. Pauli": "St Pauli",
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
    "Hellas Verona": "Verona",
    "Genoa": "Genoa",
    "Cagliari": "Cagliari",
    "Empoli": "Empoli",
    "Salernitana": "Salernitana",
    "Monza": "Monza",
    "Frosinone": "Frosinone",
    "Como": "Como",
    "Venezia": "Venezia",
    "Parma": "Parma",
    "Spezia": "Spezia",
    "Benevento": "Benevento",
    "Crotone": "Crotone",
    "Sampdoria": "Sampdoria",
    # Ligue 1
    "Paris Saint-Germain": "Paris Saint-Germain",
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
    "Metz": "Metz",
    "Troyes": "Troyes",
    "Clermont Foot": "Clermont",
    "Lorient": "Lorient",
    "Ajaccio": "Ajaccio",
}


def standardize_team(name: str) -> str:
    return UNDERSTAT_TO_STD.get(name, name)


def scrape_league_season(league_key: str, season: int) -> list[dict]:
    url = f"https://understat.com/league/{league_key}/{season}"
    print(f"  Scraping {league_key} {season}/{season+1}...")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        # El JSON está en un script tag: datesData = {...}
        soup = BeautifulSoup(resp.text, "html.parser")
        scripts = soup.find_all("script")

        dates_data = None
        for script in scripts:
            if script.string and "datesData" in script.string:
                match = re.search(r"datesData\s*=\s*(\{.*?\});", script.string, re.DOTALL)
                if match:
                    dates_data = json.loads(match.group(1))
                    break

        if not dates_data:
            # Intenta con teamsData o matchesData
            for script in scripts:
                if script.string and "teamsData" in script.string:
                    match = re.search(r"teamsData\s*=\s*(\{.*?\});", script.string, re.DOTALL)
                    if match:
                        teams_data = json.loads(match.group(1))
                        # teamsData tiene estructura distinta, parsear
                        return parse_teams_data(teams_data, league_key, season)
            print(f"    [WARN] No datesData/teamsData found")
            return []

        matches = []
        for match_id, match_data in dates_data.items():
            if not isinstance(match_data, dict):
                continue

            h = match_data.get("h", {})
            a = match_data.get("a", {})
            g = match_data.get("goals", {})

            matches.append({
                "match_id": match_id,
                "date": match_data.get("datetime", "")[:10],
                "home_team": standardize_team(h.get("title", "")),
                "away_team": standardize_team(a.get("title", "")),
                "home_goals": int(g.get("h", 0)),
                "away_goals": int(g.get("a", 0)),
                "home_xg": float(h.get("xG", 0)),
                "away_xg": float(a.get("xG", 0)),
                "league": league_key,
                "season": f"{season}-{season+1}",
            })

        print(f"    [OK] {len(matches)} partidos")
        time.sleep(0.5)
        return matches

    except Exception as e:
        print(f"    [ERR] {e}")
        return []


def parse_teams_data(teams_data: dict, league_key: str, season: int) -> list[dict]:
    """Parse estructura teamsData (alternativa)"""
    matches = []
    # teamsData: {team_id: {"title": "...", "history": [...]}}
    for team_id, team_info in teams_data.items():
        team_name = standardize_team(team_info.get("title", ""))
        history = team_info.get("history", [])
        for m in history:
            if m.get("h_a") == "h":  # Solo local para evitar duplicados
                matches.append({
                    "match_id": m.get("id", ""),
                    "date": m.get("date", "")[:10],
                    "home_team": team_name,
                    "away_team": standardize_team(m.get("opponent", "")),
                    "home_goals": int(m.get("goals", {}).get("h", 0)),
                    "away_goals": int(m.get("goals", {}).get("a", 0)),
                    "home_xg": float(m.get("xG", {}).get("h", 0)),
                    "away_xg": float(m.get("xG", {}).get("a", 0)),
                    "league": league_key,
                    "season": f"{season}-{season+1}",
                })
    print(f"    [OK] {len(matches)} partidos (via teamsData)")
    return matches


def main():
    DATA_DIR.mkdir(exist_ok=True)

    all_matches = []

    for league_name, league_key in LEAGUES.items():
        print(f"\n{'='*50}")
        print(f"LIGA: {league_name} ({league_key})")
        print(f"{'='*50}")

        for season in SEASONS:
            matches = scrape_league_season(league_key, season)
            all_matches.extend(matches)

    if not all_matches:
        print("\n[ERR] No se scrapeó nada")
        return

    df = pd.DataFrame(all_matches)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])

    # Deduplica (teamsData puede duplicar)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team", "league"])

    df = df.sort_values(["league", "date"]).reset_index(drop=True)

    # Añade xGA (xG concedido)
    df["home_xga"] = df["away_xg"]
    df["away_xga"] = df["home_xg"]

    df.to_parquet(XG_FILE, index=False)
    print(f"\n[SAVE] {XG_FILE} ({len(df):,} partidos únicos)")

    print(f"\nResumen:")
    print(f"  Total: {len(df):,}")
    print(f"  Ligas: {df['league'].nunique()} ({', '.join(sorted(df['league'].unique()))})")
    print(f"  Temporadas: {df['season'].nunique()}")
    print(f"  Equipos: {pd.concat([df['home_team'], df['away_team']]).nunique()}")
    print(f"  xG medio local: {df['home_xg'].mean():.3f}")
    print(f"  xG medio visitante: {df['away_xg'].mean():.3f}")

    # Verifica merge
    print("\nVerificando merge con matches_clean...")
    clean = pd.read_parquet(DATA_DIR / "matches_clean.parquet")
    clean["DateOnly"] = clean["Date"].dt.date
    df["DateOnly"] = df["date"].dt.date

    merged = clean.merge(
        df[["DateOnly", "home_team", "away_team", "home_xg", "away_xg", "home_xga", "away_xga"]],
        left_on=["DateOnly", "HomeTeam", "AwayTeam"],
        right_on=["DateOnly", "home_team", "away_team"],
        how="left",
        indicator=True
    )
    match_rate = (merged["_merge"] == "both").mean() * 100
    print(f"  Merge rate: {match_rate:.1f}%")

    if match_rate < 90:
        print("  [WARN] Merge rate bajo")
        unmatched = merged[merged["_merge"] == "left_only"][["HomeTeam", "AwayTeam"]].drop_duplicates()
        print(f"  Ejemplos sin match: {len(unmatched)}")
        for _, r in unmatched.head(15).iterrows():
            print(f"    {r['HomeTeam']} vs {r['AwayTeam']}")


if __name__ == "__main__":
    main()