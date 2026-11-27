#!/usr/bin/env python
"""
OddsPortal Scraper for upcoming fixtures + odds
Uses Playwright to render JavaScript and parse match data from cuotasahora.com
Target leagues: E0, SP1, D1, I1, F1 (top 5 European leagues)
"""
import asyncio
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright

# Reuse team name mapping from scrape_xg.py
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
    "Hull": "Hull City",
    # La Liga
    "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona",
    "Atletico Madrid": "Atletico Madrid",
    "Atletico de Madrid": "Atletico Madrid",
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
    "Racing Santander": "Racing Santander",
    "R. Racing Club": "Racing Santander",
    "Malaga": "Malaga",
    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Bayern Múnich": "Bayern Munich",
    "Borussia Dortmund": "Borussia Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "Borussia Monchengladbach": "Borussia Monchengladbach",
    "VfL Wolfsburg": "Wolfsburg",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "SC Freiburg": "Freiburg",
    "Union Berlin": "Union Berlin",
    "FSV Mainz 05": "Mainz",
    "Mainz": "Mainz",
    "TSG Hoffenheim": "Hoffenheim",
    "FC Augsburg": "Augsburg",
    "Augsburgo": "Augsburg",
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
    "Hamburgo": "Hamburger SV",
    "Hamburger SV": "Hamburger SV",
    # Serie A
    "Juventus": "Juventus",
    "Inter": "Inter Milan",
    "Inter Milan": "Inter Milan",
    "AC Milan": "AC Milan",
    "Napoli": "Napoli",
    "Roma": "Roma",
    "Lazio": "Lazio",
    "Atalanta": "Atalanta",
    "Fiorentina": "Fiorentina",
    "Torino": "Torino",
    "Bologna": "Bologna",
    "Bolonia": "Bologna",
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
    "PSG": "Paris Saint-Germain",
    "Marseille": "Marseille",
    "Marsella": "Marseille",
    "Lyon": "Lyon",
    "Monaco": "Monaco",
    "Lille": "Lille",
    "Rennes": "Rennes",
    "Stade Rennais": "Rennes",
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
    "Paris FC": "Paris FC",
    "Le Mans": "Le Mans",
}

# OddsPortal league name -> our league code
ODDSPORTAL_TO_LEAGUE = {
    "Premier League": "E0",
    "LaLiga EA Sports": "SP1",
    "La Liga": "SP1",
    "Bundesliga": "D1",
    "Serie A": "I1",
    "Ligue 1": "F1",
}

TARGET_LEAGUES = {"E0", "SP1", "D1", "I1", "F1"}

# Exact league names we want (to avoid 2. Bundesliga, women's leagues, etc.)
TARGET_LEAGUE_NAMES = {
    "Premier League",
    "LaLiga EA Sports",
    "La Liga",
    "Bundesliga",  # exactly "Bundesliga" not "2. Bundesliga" or "Bundesliga - Femenina"
    "Serie A",
    "Ligue 1",
}

# Valid top-division teams per league (2026-2027 season) - includes all name variations from website
VALID_TEAMS_BY_LEAGUE = {
    "E0": {
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Brighton & Hove Albion",
        "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich", "Ipswich Town",
        "Leicester", "Leicester City", "Liverpool", "Manchester City", "Manchester United", "Man United",
        "Newcastle", "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham", "Tottenham Hotspur",
        "West Ham", "West Ham United", "Wolves", "Wolverhampton Wanderers"
    },
    "SP1": {
        "Alaves", "Athletic Bilbao", "Athletic Club", "Atlético Madrid", "Atletico Madrid", "Atlético de Madrid",
        "Barcelona", "Betis", "Real Betis", "Celta Vigo", "Espanyol", "Getafe", "Girona",
        "Las Palmas", "Leganes", "Mallorca", "Osasuna", "Rayo Vallecano",
        "Real Madrid", "Real Sociedad", "Sevilla", "Valencia", "Villarreal"
    },
    "D1": {
        "Augsburg", "Augsburgo", "Bayern Munich", "Bayern Múnich", "Bochum", "Dortmund", "Borussia Dortmund",
        "Eintracht Frankfurt", "Freiburg", "Heidenheim", "Hoffenheim", "Holstein Kiel", "Köln", "Koln", "FC Koln",
        "Leverkusen", "Bayer Leverkusen", "Mainz", "FSV Mainz 05", "Mönchengladbach", "Borussia Monchengladbach",
        "RB Leipzig", "St. Pauli", "St Pauli", "Stuttgart", "VfB Stuttgart", "Union Berlin", "Wolfsburg", "VfL Wolfsburg"
    },
    "I1": {
        "Atalanta", "Bologna", "Bolonia", "Cagliari", "Como", "Empoli", "Fiorentina", "Genoa",
        "Inter", "Inter Milan", "Juventus", "Lazio", "Lecce", "Milan", "AC Milan", "Monza", "Napoli",
        "Parma", "Roma", "Torino", "Udinese", "Venezia", "Verona", "Hellas Verona"
    },
    "F1": {
        "Angers", "Auxerre", "Brest", "Le Havre", "Lens", "Lille", "Lorient", "Lyon",
        "Marseille", "Marsella", "Monaco", "Montpellier", "Nantes", "Nice", "PSG", "Paris Saint-Germain",
        "Reims", "Rennes", "Stade Rennais", "Saint-Etienne", "Strasbourg", "Toulouse"
    },
    "UCL": {
        "Arsenal", "Aston Villa", "Atalanta", "Atlético Madrid", "Atletico Madrid", "Atlético de Madrid",
        "Barcelona", "Bayern Munich", "Bayern Múnich", "Bayer Leverkusen", "Borussia Dortmund",
        "Celtic", "Club Brujas", "Club Brugge", "Dortmund", "Eintracht Frankfurt", "Feyenoord",
        "Inter", "Inter Milan", "Juventus", "Lazio", "Leipzig", "RB Leipzig", "Leverkusen",
        "Liverpool", "Lille", "Manchester City", "Milan", "AC Milan", "Monaco", "Napoli",
        "Paris Saint-Germain", "PSG", "Porto", "Real Madrid", "Real Sociedad", "Roma",
        "Salzburg", "Shakhtar", "Shakhtar Donetsk", "Sporting", "Sporting CP", "Stuttgart",
        "Villarreal", "Viking", "Young Boys", "Sparta Praga", "Sparta Prague", "Slavia Praga",
        "Slavia Prague", "Dinamo Zagreb", "Red Star", "Crvena Zvezda", "Bologna", "Girona",
        "Brest", "Benfica", "Braga", "Lens", "PSV", "PSV Eindhoven"
    },
    "UEL": {
        "Athletic Bilbao", "Athletic Club", "Ajax", "Ajax Amsterdam", "AZ", "AZ Alkmaar",
        "Besiktas", "Bodø/Glimt", "Bodo Glimt", "Braga", "Elfsborg", "Fenerbahçe", "Fenerbahce",
        "Ferencvaros", "Galatasaray", "Glasgow Rangers", "Rangers", "Hoffenheim", "Lazio",
        "Lyon", "Maccabi Tel Aviv", "Malmo", "Malmö", "Manchester United", "Man United",
        "Midtjylland", "Nice", "Olympiacos", "PAOK", "Porto", "Qarabag", "Qarabağ",
        "Real Sociedad", "Roma", "Rigas FS", "Slavia Praga", "Slavia Prague", "Tottenham",
        "Tottenham Hotspur", "Twente", "Union Saint-Gilloise", "Victoria Plzen"
    },
    "UECL": {
        "Astana", "Basel", "Chelsea", "Copenhagen", "Djurgarden", "Djurgården", "Fiorentina",
        "Gent", "Hearts", "Heidenheim", "Istanbul Basaksehir", "Istanbul Basaksehir", "Jagiellonia",
        "Larne", "Legia", "Legia Warsaw", "LASK", "Lugano", "Molde", "Noah", "Omonia",
        "Pafos", "Panathinaikos", "Petrocub", "Puskas Akademia", "Puskás Akadémia",
        "Real Betis", "Shamrock Rovers", "St. Gallen", "St Gallen", "The New Saints",
        "Viking", "Vikingur", "Víkingur", "Vitoria", "Vitória Guimarães", "Vitoria Guimaraes"
    },
}


def standardize_team(name: str) -> str:
    """Normalize team name to match historical data."""
    import unicodedata
    name = name.strip()
    # Remove accents
    name_no_accents = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')

    # Direct match
    if name in UNDERSTAT_TO_STD:
        return UNDERSTAT_TO_STD[name]
    if name_no_accents in UNDERSTAT_TO_STD:
        return UNDERSTAT_TO_STD[name_no_accents]

    # Case-insensitive match
    name_lower = name.lower()
    name_no_accents_lower = name_no_accents.lower()
    for k, v in UNDERSTAT_TO_STD.items():
        k_lower = k.lower()
        k_no_accents = ''.join(c for c in unicodedata.normalize('NFD', k) if unicodedata.category(c) != 'Mn').lower()
        if k_lower == name_lower or k_no_accents == name_no_accents_lower:
            return v

    # Spanish name mappings for foreign teams
    spanish_to_std = {
        "Estrasburgo": "Strasbourg",
        "Niza": "Nice",
        "Marsella": "Marseille",
        "Lyon": "Lyon",  # same
        "Múnich": "Munich",
        "Bayern Múnich": "Bayern Munich",
        "Hamburgo": "Hamburger SV",
        "Colonia": "Koln",
        "Fráncfort": "Frankfurt",
        "Eintracht Fráncfort": "Eintracht Frankfurt",
        "Leipzig": "RB Leipzig",
        "Leverkusen": "Bayer Leverkusen",
        "Wolfsburgo": "Wolfsburg",
        "Mónchengladbach": "Borussia Monchengladbach",
        "Dortmund": "Borussia Dortmund",
        "Stuttgart": "Stuttgart",
        "Unión Berlín": "Union Berlin",
        "Heidenheim": "Heidenheim",
        "Hoffenheim": "Hoffenheim",
        "Augsburgo": "Augsburg",
        "Bochum": "Bochum",
        "Kiel": "Holstein Kiel",
        "San Pauli": "St Pauli",
        "Friburgo": "Freiburg",
        "Maguncia": "Mainz",
        "Paderborn": "Paderborn",  # 2. Bundesliga
        "Schalke": "Schalke",  # 2. Bundesliga
        "Elversberg": "Elversberg",  # 2. Bundesliga
        "Saarbrücken": "Saarbrucken",
        "Aachen": "Aachen",
    }

    if name in spanish_to_std:
        return spanish_to_std[name]
    if name_no_accents in spanish_to_std:
        return spanish_to_std[name_no_accents]

    return name


def parse_oddsportal_time(time_str: str, ref_date: datetime) -> datetime:
    """Parse OddsPortal time string (HH:MM) with reference date."""
    try:
        time_part = time_str.strip()
        if ":" in time_part:
            hour, minute = map(int, time_part.split(":")[:2])
            return ref_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    except Exception:
        pass
    return ref_date


class OddsPortalScraper:
    def __init__(self, headless: bool = True, timeout: int = 60000):
        self.headless = headless
        self.timeout = timeout
        self.browser = None
        self.page = None

    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.page = await self.browser.new_page()
        await self.page.set_viewport_size({"width": 1920, "height": 1080})
        await self.page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def scrape_date(self, date_str: str) -> list[dict]:
        """
        Scrape fixtures for a specific date.
        date_str: 'YYYY-MM-DD' or 'YYYYMMDD'
        Returns list of fixture dicts compatible with fetch_live.py
        """
        if "-" in date_str:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        else:
            dt = datetime.strptime(date_str, "%Y%m%d")

        url_date = dt.strftime("%Y-%m-%d")
        main_url = f"https://www.cuotasahora.com/football/{url_date}/"

        all_fixtures = []
        fixture_id = 1000

        print(f"[SCRAPE] Loading main page: {main_url}")
        await self.page.goto(main_url, wait_until="networkidle", timeout=self.timeout)
        await self.page.wait_for_timeout(5000)

        # Click "Ocultar eventos finalizados" to hide finished matches
        try:
            hide_finished = await self.page.query_selector('text="Ocultar eventos finalizados"')
            if hide_finished:
                print("[SCRAPE] Clicking Ocultar eventos finalizados...")
                await hide_finished.click()
                await self.page.wait_for_timeout(2000)
        except Exception:
            pass

        # Scroll down to trigger lazy loading of more leagues
        print("[SCRAPE] Scrolling to load more content...")
        for _ in range(15):
            await self.page.evaluate("window.scrollBy(0, 1000)")
            await self.page.wait_for_timeout(500)

        body = await self.page.query_selector("body")
        all_text = await body.inner_text()

        Path("debug_page_main.txt").write_text(all_text, encoding="utf-8")

        lines = [line.strip() for line in all_text.split('\n') if line.strip()]
        fixtures = self._parse_lines(lines, dt)

        for f in fixtures:
            f["fixture"]["id"] = fixture_id
            fixture_id += 1

        all_fixtures.extend(fixtures)
        print(f"[SCRAPE] Found {len(fixtures)} fixtures from main page")

        print(f"[SCRAPE] Total: {len(all_fixtures)} fixtures in target leagues")
        return all_fixtures

    def _parse_lines(self, lines: list[str], ref_date: datetime) -> list[dict]:
        """Parse non-empty lines from page text."""
        fixtures = []
        fixture_id = 1000
        current_league_code = None
        i = 0

        while i < len(lines):
            line = lines[i]

            # Detect league header: Fútbol / Country / League / Mañana (domestic leagues)
            if line == "Fútbol" and i + 4 < len(lines):
                if lines[i+1] == "/" and lines[i+3] == "/":
                    country = lines[i+2]
                    league_name = lines[i+4]

                    # Check if next line is "Mañana", "Hoy", or date format (DD MMM. YYYY)
                    next_line = lines[i+5] if i + 5 < len(lines) else ""
                    is_today = ("Mañana" in next_line or "Hoy" in next_line or
                                re.match(r'\d{1,2} \w{3}\. \d{4}', next_line))
                    if i + 5 < len(lines) and is_today:
                        # Validate country + league exact match (avoid 2. Bundesliga, foreign leagues)
                        # Website uses Spanish country names
                        current_league_code = None
                        valid = False
                        if country == "Inglaterra" and league_name == "Premier League":
                            current_league_code = "E0"
                            valid = True
                        elif country == "España" and league_name in ["LaLiga EA Sports", "La Liga"]:
                            current_league_code = "SP1"
                            valid = True
                        elif country == "Alemania" and league_name == "Bundesliga":
                            current_league_code = "D1"
                            valid = True
                        elif country == "Italia" and league_name == "Serie A":
                            current_league_code = "I1"
                            valid = True
                        elif country == "Francia" and league_name == "Ligue 1":
                            current_league_code = "F1"
                            valid = True
                        # European competitions (country = "Europa")
                        elif country == "Europa" and league_name == "Champions League":
                            current_league_code = "UCL"
                            valid = True
                        elif country == "Europa" and league_name == "Europa League":
                            current_league_code = "UEL"
                            valid = True
                        elif country == "Europa" and league_name == "Conference League":
                            current_league_code = "UECL"
                            valid = True

                        if not valid:
                            current_league_code = None

                        i += 6  # Skip header lines
                        continue

            # Detect Champions League / Europa League / Conference League (no country prefix)
            # Format: "Champions League" -> "Mañana, 25 nov. - Fase de liga" -> "1" "X" "2"
            if line in ["Champions League", "Europa League", "Conference League"] and i + 2 < len(lines):
                league_name = line
                next_line = lines[i+1] if i + 1 < len(lines) else ""
                next_next = lines[i+2] if i + 2 < len(lines) else ""
                is_today = ("Mañana" in next_line or "Hoy" in next_line or
                            re.match(r'\d{1,2} \w{3}\. \d{4}', next_line) or
                            "Fase de liga" in next_line)
                if is_today:
                    if league_name == "Champions League":
                        current_league_code = "UCL"
                    elif league_name == "Europa League":
                        current_league_code = "UEL"
                    elif league_name == "Conference League":
                        current_league_code = "UECL"
                    else:
                        current_league_code = None

                    if current_league_code is None:
                        i += 3
                        continue

                    i += 5  # Skip league name, date line, "1", "X", "2" (3 separate lines)
                    continue

            # Parse match: time -> home -> - -> away -> odds...
            if current_league_code and re.match(r'^\d{1,2}:\d{2}$', line):
                time_str = line
                i += 1

                # Collect home team (may span multiple lines until "-")
                home_parts = []
                while i < len(lines):
                    l = lines[i]
                    if l == "-":
                        break
                    if l and not re.match(r'^\d{1,2}:\d{2}$', l):
                        home_parts.append(l)
                    i += 1

                if i >= len(lines) or lines[i] != "-":
                    continue
                i += 1  # skip "-"

                # Collect away team (next non-empty line that's not a time or odds)
                away_team = ""
                while i < len(lines):
                    l = lines[i]
                    if l and not re.match(r'^\d{1,2}:\d{2}$', l):
                        try:
                            float(l)  # skip odds
                        except ValueError:
                            away_team = l
                            break
                    i += 1

                if not away_team:
                    continue

                # Collect odds (next 3 numeric lines)
                odds = {}
                odds_vals = []
                while i < len(lines) and len(odds_vals) < 3:
                    l = lines[i]
                    try:
                        val = float(l)
                        odds_vals.append(val)
                    except ValueError:
                        pass
                    i += 1

                if len(odds_vals) >= 3:
                    odds = {"home": odds_vals[0], "draw": odds_vals[1], "away": odds_vals[2]}

                home_team = " ".join(home_parts)
                home_team = standardize_team(home_team)
                away_team = standardize_team(away_team)

                # Validate teams: for domestic leagues, both must be in valid set
                # For European competitions (UCL/UEL/UECL), be more permissive - at least one team known
                valid_teams = VALID_TEAMS_BY_LEAGUE.get(current_league_code, set())
                if valid_teams:
                    if current_league_code in ("UCL", "UEL", "UECL"):
                        # European comps: teams from different domestic leagues face each other
                        # Accept if at least one team is recognized
                        if home_team not in valid_teams and away_team not in valid_teams:
                            continue
                    else:
                        # Domestic leagues: both teams must be in top division
                        if home_team not in valid_teams or away_team not in valid_teams:
                            continue

                match_dt = parse_oddsportal_time(time_str, ref_date)

                fixture = {
                    "fixture": {"id": fixture_id, "date": match_dt.isoformat()},
                    "teams": {"home": {"name": home_team}, "away": {"name": away_team}},
                    "league": {"code": current_league_code},
                    "odds": odds,
                }
                fixtures.append(fixture)
                fixture_id += 1
                continue

            i += 1

        return fixtures


async def get_fixtures_for_date(date_str: str) -> list[dict]:
    """Main entry point - scrape fixtures for a date."""
    async with OddsPortalScraper() as scraper:
        return await scraper.scrape_date(date_str)


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    fixtures = asyncio.run(get_fixtures_for_date(date_str))

    print(f"\n{'='*60}")
    print(f"ODDSPORTAL FIXTURES - {date_str}")
    print(f"{'='*60}")

    for f in fixtures:
        odds_str = ""
        if f["odds"]:
            odds_str = f" | 1:{f['odds'].get('home', 'N/A')} X:{f['odds'].get('draw', 'N/A')} 2:{f['odds'].get('away', 'N/A')}"
        # Safe print for Unicode
        home = f['teams']['home']['name']
        away = f['teams']['away']['name']
        print(f"  {f['league']['code']} | {home} vs {away} | {f['fixture']['date']}{odds_str}")

    import pandas as pd
    df = pd.DataFrame([{
        "date": f["fixture"]["date"],
        "league": f["league"]["code"],
        "home": f["teams"]["home"]["name"],
        "away": f["teams"]["away"]["name"],
        "odds_home": f["odds"].get("home"),
        "odds_draw": f["odds"].get("draw"),
        "odds_away": f["odds"].get("away"),
    } for f in fixtures])

    output_file = Path("data") / f"oddsportal_fixtures_{date_str.replace('-', '')}.parquet"
    df.to_parquet(output_file, index=False)
    print(f"\n[SAVE] {output_file}")


if __name__ == "__main__":
    main()