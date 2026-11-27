#!/usr/bin/env python
"""
Scrape confirmed lineups 60-90 minutes before kickoff.
Sources: Flashscore, Sofascore, official club sites.
"""
import asyncio
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from playwright.async_api import async_playwright

# ──────────────────────────────────────────────
# TEAM NAME MAPPING (API-football → Flashscore/Sofascore)
# ──────────────────────────────────────────────
API_TO_FLASHSCORE = {
    # Premier League
    "Manchester City": "Manchester City",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Aston Villa": "Aston Villa",
    "Tottenham": "Tottenham Hotspur",
    "Manchester United": "Manchester United",
    "Chelsea": "Chelsea",
    "Newcastle": "Newcastle United",
    "Brighton": "Brighton & Hove Albion",
    "West Ham": "West Ham United",
    "Crystal Palace": "Crystal Palace",
    "Brentford": "Brentford",
    "Fulham": "Fulham",
    "Wolves": "Wolverhampton Wanderers",
    "Everton": "Everton",
    "Nottingham Forest": "Nottingham Forest",
    "Bournemouth": "AFC Bournemouth",
    "Luton": "Luton Town",
    "Burnley": "Burnley",
    "Sheffield United": "Sheffield United",

    # La Liga
    "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona",
    "Atletico Madrid": "Atletico Madrid",
    "Girona": "Girona",
    "Real Sociedad": "Real Sociedad",
    "Athletic Bilbao": "Athletic Bilbao",
    "Real Betis": "Real Betis",
    "Villarreal": "Villarreal",
    "Valencia": "Valencia",
    "Sevilla": "Sevilla",
    "Osasuna": "Osasuna",
    "Getafe": "Getafe",
    "Rayo Vallecano": "Rayo Vallecano",
    "Mallorca": "Mallorca",
    "Celta Vigo": "Celta de Vigo",
    "Alaves": "Alaves",
    "Las Palmas": "Las Palmas",
    "Granada": "Granada CF",
    "Cadiz": "Cadiz",
    "Almeria": "Almeria",

    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "Stuttgart": "Stuttgart",
    "RB Leipzig": "RB Leipzig",
    "Borussia Dortmund": "Borussia Dortmund",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Freiburg": "Freiburg",
    "Wolfsburg": "Wolfsburg",
    "Mainz": "Mainz 05",
    "Augsburg": "Augsburg",
    "Monchengladbach": "Borussia Monchengladbach",
    "Union Berlin": "Union Berlin",
    "Hoffenheim": "Hoffenheim",
    "Cologne": "FC Koln",
    "Bochum": "Bochum",
    "Heidenheim": "Heidenheim",
    "Darmstadt": "Darmstadt",

    # Serie A
    "Inter": "Inter",
    "AC Milan": "AC Milan",
    "Juventus": "Juventus",
    "Bologna": "Bologna",
    "Roma": "Roma",
    "Lazio": "Lazio",
    "Fiorentina": "Fiorentina",
    "Napoli": "Napoli",
    "Atalanta": "Atalanta",
    "Torino": "Torino",
    "Monza": "Monza",
    "Genoa": "Genoa",
    "Sassuolo": "Sassuolo",
    "Udinese": "Udinese",
    "Lecce": "Lecce",
    "Verona": "Verona",
    "Cagliari": "Cagliari",
    "Empoli": "Empoli",
    "Salernitana": "Salernitana",
    "Frosinone": "Frosinone",

    # Ligue 1
    "PSG": "PSG",
    "Monaco": "Monaco",
    "Brest": "Brest",
    "Lille": "Lille",
    "Lyon": "Lyon",
    "Nice": "Nice",
    "Lens": "Lens",
    "Marseille": "Marseille",
    "Rennes": "Rennes",
    "Reims": "Reims",
    "Montpellier": "Montpellier",
    "Toulouse": "Toulouse",
    "Strasbourg": "Strasbourg",
    "Nantes": "Nantes",
    "Le Havre": "Le Havre",
    "Metz": "Metz",
    "Lorient": "Lorient",
    "Clermont": "Clermont Foot",
}


def map_team_name(api_name: str) -> str:
    """Map API-football team name to Flashscore/Sofascore name."""
    return API_TO_FLASHSCORE.get(api_name, api_name)


async def get_lineups_flashscore(page, home_team: str, away_team: str, match_date: datetime) -> Optional[dict]:
    """
    Get confirmed lineups from Flashscore.
    """
    try:
        # Flashscore match URL format: https://www.flashscore.com/match/{match_id}/#/lineups
        # We need to find the match first
        search_url = f"https://www.flashscore.com/search/?q={home_team}%20{away_team}"
        await page.goto(search_url, wait_until="networkidle", timeout=30000)

        # Wait for search results
        await page.wait_for_selector(".searchResult", timeout=10000)

        # Find the correct match link
        match_links = await page.query_selector_all(".searchResult a")
        for link in match_links:
            href = await link.get_attribute("href")
            text = await link.inner_text()
            if home_team.lower() in text.lower() and away_team.lower() in text.lower():
                match_url = f"https://www.flashscore.com{href}#/lineups"
                await page.goto(match_url, wait_until="networkidle", timeout=30000)
                break
        else:
            return None

        # Wait for lineups to load
        await page.wait_for_selector(".lineup", timeout=15000)

        # Extract lineups
        home_players = await page.query_selector_all(".lineup__home .lineup__player")
        away_players = await page.query_selector_all(".lineup__away .lineup__player")

        home_xi = []
        for p in home_players[:11]:  # Starting XI
            name_el = await p.query_selector(".lineup__player-name")
            pos_el = await p.query_selector(".lineup__player-position")
            if name_el:
                name = await name_el.inner_text()
                pos = await pos_el.inner_text() if pos_el else ""
                home_xi.append({"name": name.strip(), "position": pos.strip()})

        away_xi = []
        for p in away_players[:11]:
            name_el = await p.query_selector(".lineup__player-name")
            pos_el = await p.query_selector(".lineup__player-position")
            if name_el:
                name = await name_el.inner_text()
                pos = await pos_el.inner_text() if pos_el else ""
                away_xi.append({"name": name.strip(), "position": pos.strip()})

        # Formation
        home_form = await page.query_selector(".lineup__home .lineup__formation")
        away_form = await page.query_selector(".lineup__away .lineup__formation")
        home_formation = await home_form.inner_text() if home_form else ""
        away_formation = await away_form.inner_text() if away_form else ""

        return {
            "source": "flashscore",
            "home": {"formation": home_formation.strip(), "players": home_xi},
            "away": {"formation": away_formation.strip(), "players": away_xi},
            "confirmed": True,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"[Flashscore] Error: {e}")
        return None


async def get_lineups_sofascore(page, home_team: str, away_team: str, match_date: datetime) -> Optional[dict]:
    """
    Get confirmed lineups from Sofascore.
    """
    try:
        # Sofascore match URL format: https://www.sofascore.com/{home}-{away}/{match_id}
        search_url = f"https://www.sofascore.com/api/v1/search/all?q={home_team}%20{away_team}"
        await page.goto(search_url, wait_until="networkidle", timeout=30000)

        # Parse JSON response (wrapped in <pre> tag)
        content = await page.content()
        import json, re
        try:
            # Extract JSON from <pre> tag
            match = re.search(r'<pre[^>]*>(.*?)</pre>', content, re.DOTALL)
            json_str = match.group(1) if match else content
            data = json.loads(json_str)

            # Search results are in "results" array with nested "entity"
            results = data.get("results", [])
            match_id = None
            for result in results:
                entity = result.get("entity", result)
                if (home_team.lower() in entity.get("homeTeam", {}).get("name", "").lower() and
                        away_team.lower() in entity.get("awayTeam", {}).get("name", "").lower()):
                    match_id = entity.get("id")
                    break

            if not match_id:
                return None

            lineup_url = f"https://www.sofascore.com/api/v1/event/{match_id}/lineups"
            await page.goto(lineup_url, wait_until="networkidle", timeout=30000)
        except json.JSONDecodeError:
            return None

        # Parse lineup JSON (also wrapped in <pre> tag)
        content = await page.content()
        match = re.search(r'<pre[^>]*>(.*?)</pre>', content, re.DOTALL)
        json_str = match.group(1) if match else content
        data = json.loads(json_str)

        home_players = data.get("home", {}).get("players", [])
        away_players = data.get("away", {}).get("players", [])

        # Map Sofascore position codes (G, D, M, F) to standard
        pos_map = {"G": "GK", "D": "CB", "M": "CM", "F": "ST"}

        home_xi = [{"name": p["player"]["name"], "position": pos_map.get(p["position"], p["position"])} for p in home_players[:11]]
        away_xi = [{"name": p["player"]["name"], "position": pos_map.get(p["position"], p["position"])} for p in away_players[:11]]

        home_form = data.get("home", {}).get("formation", "")
        away_form = data.get("away", {}).get("formation", "")
        confirmed = data.get("confirmed", False)

        return {
            "source": "sofascore",
            "home": {"formation": home_form, "players": home_xi},
            "away": {"formation": away_form, "players": away_xi},
            "confirmed": confirmed,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"[Sofascore] Error: {e}")
        return None


async def get_lineups_for_match(home_team: str, away_team: str, match_date: datetime) -> Optional[dict]:
    """
    Get confirmed lineups for a match from Sofascore.
    Flashscore disabled due to bot detection.
    """
    home_mapped = map_team_name(home_team)
    away_mapped = map_team_name(away_team)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Use Sofascore only (Flashscore blocked by bot detection)
        result = await get_lineups_sofascore(page, home_mapped, away_mapped, match_date)
        await browser.close()

        return result


async def get_lineups_for_fixtures(fixtures: list, hours_before: float = 1.5) -> dict:
    """
    Get lineups for multiple fixtures.
    Only fetches for matches starting within `hours_before` hours.
    """
    now = datetime.now()
    cutoff = now + timedelta(hours=hours_before)

    results = {}
    for f in fixtures:
        fixture = f["fixture"]
        teams = f["teams"]
        home = teams["home"]["name"]
        away = teams["away"]["name"]
        match_time = pd.Timestamp(fixture["date"]).to_pydatetime()

        if now <= match_time <= cutoff:
            print(f"[LINEUPS] Fetching for {home} vs {away}...")
            lineup = await get_lineups_for_match(home, away, match_time)
            if lineup:
                results[f"{home} vs {away}"] = lineup
                print(f"  ✅ Got lineups from {lineup['source']}")
            else:
                print(f"  ⚠️  No lineups available yet")

    return results


# ──────────────────────────────────────────────
# PLAYER IMPORTANCE WEIGHTS (for adjusting team strength)
# ──────────────────────────────────────────────
# Key players by team (simplified - in production use market value / minutes / stats)
KEY_PLAYERS = {
    # Premier League
    "Manchester City": ["Haaland", "De Bruyne", "Rodri", "Ruben Dias", "Ederson"],
    "Arsenal": ["Saka", "Odegaard", "Rice", "Saliba", "Raya"],
    "Liverpool": ["Salah", "Van Dijk", "Alisson", "Alexander-Arnold", "Nunez"],
    "Aston Villa": ["Watkins", "Tielemans", "Martinez", "Pau Torres", "McGinn"],
    "Tottenham": ["Son", "Maddison", "Romero", "Van de Ven", "Vicario"],
    "Manchester United": ["Bruno Fernandes", "Rashford", "Casemiro", "Martinez", "Onana"],
    "Chelsea": ["Palmer", "Jackson", "Caicedo", "Colwill", "Sanchez"],

    # La Liga
    "Real Madrid": ["Vinicius", "Bellingham", "Rodrygo", "Rudiger", "Courtois"],
    "Barcelona": ["Yamal", "Lewandowski", "Pedri", "Cubarsi", "Ter Stegen"],
    "Atletico Madrid": ["Griezmann", "Morata", "Koke", "Gimenez", "Oblak"],
    "Girona": ["Dovbyk", "Tsygankov", "Stuani", "Blind", "Gazzaniga"],
    "Real Sociedad": ["Kubo", "Oyarzabal", "Merino", "Zubimendi", "Remiro"],
    "Athletic Bilbao": ["Williams", "Sancet", "Berenguer", "Vivian", "Unai Simon"],
    "Real Betis": ["Isco", "Fekir", "Guido", "Pezzella", "Rui Silva"],
    "Fiorentina": ["Gonzalez", "Beltran", "Mandragora", "Ranieri", "De Gea"],

    # Bundesliga
    "Bayern Munich": ["Kane", "Musiala", "Kimmich", "Davies", "Neuer"],
    "Bayer Leverkusen": ["Wirtz", "Boniface", "Xhaka", "Hincapie", "Hradecky"],
    "Stuttgart": ["Guirassy", "Mittlestadt", "Undav", "Anton", "Nubel"],
    "RB Leipzig": ["Sesko", "Olmo", "Simons", "Orban", "Gulacsi"],

    # Serie A
    "Inter": ["Lautaro", "Thuram", "Barella", "Bastoni", "Sommer"],
    "AC Milan": ["Leao", "Pulisic", "Reijnders", "Tomori", "Maingnan"],
    "Juventus": ["Vlahovic", "Koopmeiners", "Locatelli", "Bremer", "Perin"],
    "Bologna": ["Zirkzee", "Orsolini", "Freuler", "Calafiori", "Skov Olsen"],
    "Roma": ["Dybala", "Pellegrini", "Dovbyk", "Mancini", "Svilar"],
    "Lazio": ["Immobile", "Zaccagni", "Guendouzi", "Romagnoli", "Provedel"],
    "Fiorentina": ["Gonzalez", "Beltran", "Mandragora", "Ranieri", "De Gea"],

    # Ligue 1
    "PSG": ["Mbappe", "Dembele", "Barcola", "Vitinha", "Donnarumma"],
    "Monaco": ["Balogun", "Minamino", "Ben Seghir", "Singh", "Majecki"],
    "Brest": ["Le Douaron", "Mounie", "Magnetti", "Chardonnet", "Bizot"],
    "Lille": ["David", "Zhegrova", "Andre", "Diakite", "Chevalier"],
}

PLAYER_POSITION_WEIGHTS = {
    "GK": 1.2,   # Goalkeeper very important
    "CB": 1.0,
    "LB": 0.8,
    "RB": 0.8,
    "DM": 1.0,
    "CM": 0.9,
    "AM": 0.9,
    "LW": 1.0,
    "RW": 1.0,
    "ST": 1.1,   # Striker most important for goals
}


def calculate_lineup_strength(lineup: dict, team_name: str) -> float:
    """
    Calculate lineup strength adjustment factor (0.8 - 1.2).
    1.0 = expected/strongest XI, <1.0 = weakened, >1.0 = stronger than expected.
    """
    if not lineup or not lineup.get("players"):
        return 1.0

    key_players = set(KEY_PLAYERS.get(team_name, []))
    starting_names = {p["name"].split()[-1] for p in lineup["players"]}  # Last name matching

    # Count key players in starting XI
    key_in_xi = len(key_players & starting_names)
    total_key = len(key_players)

    if total_key == 0:
        return 1.0

    # Base strength from key player presence
    key_ratio = key_in_xi / total_key

    # Position-weighted strength
    pos_weight = 0
    for p in lineup["players"]:
        pos = p.get("position", "").upper()
        weight = PLAYER_POSITION_WEIGHTS.get(pos, 0.9)
        name_key = p["name"].split()[-1]
        if name_key in key_players:
            weight *= 1.3  # Key player bonus
        pos_weight += weight

    avg_pos_weight = pos_weight / max(len(lineup["players"]), 1)

    # Combine: 70% key player ratio, 30% positional weight
    strength = 0.7 * key_ratio + 0.3 * avg_pos_weight

    # Normalize to 0.7-1.3 range
    return max(0.7, min(1.3, strength))


def adjust_team_strength_for_lineups(
    home_strength: dict,
    away_strength: dict,
    home_lineup: Optional[dict],
    away_lineup: Optional[dict],
    home_team: str,
    away_team: str
) -> tuple:
    """
    Adjust attack/defense strength based on confirmed lineups.
    Returns adjusted (home_strength, away_strength).
    """
    home_adj = calculate_lineup_strength(home_lineup.get("home") if home_lineup else None, home_team)
    away_adj = calculate_lineup_strength(away_lineup.get("away") if away_lineup else None, away_team)

    # Adjust attack/defense rolls
    for w in [5, 10, 20]:
        for prefix, adj in [("home_", home_adj), ("away_", away_adj)]:
            for key in [f"{prefix}attack_roll{w}", f"{prefix}defense_roll{w}", f"{prefix}gd_roll{w}"]:
                if key in home_strength:
                    home_strength[key] *= adj
                if key in away_strength:
                    away_strength[key] *= adj

    return home_strength, away_strength


# ──────────────────────────────────────────────
# MAIN (for testing)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    # Test lineup strength calculation
    test_lineup = {
        "home": {
            "formation": "4-3-3",
            "players": [
                {"name": "De Gea", "position": "GK"},
                {"name": "Ranieri", "position": "CB"},
                {"name": "Martinez Quarta", "position": "CB"},
                {"name": "Biraghi", "position": "LB"},
                {"name": "Dodo", "position": "RB"},
                {"name": "Mandragora", "position": "CM"},
                {"name": "Arthur", "position": "CM"},
                {"name": "Bonaventura", "position": "AM"},
                {"name": "Gonzalez", "position": "LW"},
                {"name": "Beltran", "position": "ST"},
                {"name": "Kouame", "position": "RW"},
            ]
        }
    }

    strength = calculate_lineup_strength(test_lineup["home"], "Fiorentina")
    print(f"Fiorentina lineup strength: {strength:.2f}")

    # Test with missing key players
    test_lineup_weak = {
        "home": {
            "formation": "4-3-3",
            "players": [
                {"name": "Terracciano", "position": "GK"},
                {"name": "Ranieri", "position": "CB"},
                {"name": "Martinez Quarta", "position": "CB"},
                {"name": "Biraghi", "position": "LB"},
                {"name": "Dodo", "position": "RB"},
                {"name": "Mandragora", "position": "CM"},
                {"name": "Arthur", "position": "CM"},
                {"name": "Bonaventura", "position": "AM"},
                {"name": "Ikone", "position": "LW"},
                {"name": "Nzola", "position": "ST"},
                {"name": "Kouame", "position": "RW"},
            ]
        }
    }

    strength_weak = calculate_lineup_strength(test_lineup_weak["home"], "Fiorentina")
    print(f"Fiorentina weakened lineup strength: {strength_weak:.2f}")