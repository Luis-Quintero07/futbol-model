#!/usr/bin/env python
"""
Fetch live fixtures + odds from API-football or OddsPortal, build features, run predictions
"""
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
API_KEY = "1620116392c66ab66ef031e88e01ed03"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
FEATURES_FILE = DATA_DIR / "matches_pre_match.parquet"

# Leagues we care about (API-football IDs)
TARGET_LEAGUES = {
    39: "E0",   # Premier League
    140: "SP1", # La Liga
    78: "D1",   # Bundesliga
    135: "I1",  # Serie A
    61: "F1",   # Ligue 1
}

# Colombian timezone (UTC-5)
COL_TZ = timezone(timedelta(hours=-5))

EXCLUDE_FEATURES = {
    "Season", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Result",
    "imp_prob_H", "imp_prob_D", "imp_prob_A", "overround",
    "imp_prob_O25", "imp_prob_U25", "ou25_favorite",
    "favorite", "favorite_prob", "odds_ratio_H_A", "draw_prob",
    "home_win_prob", "away_win_prob", "market_entropy", "market_max_prob",
    "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5",
    "B365AHH", "B365AHA",
    # "Div" REMOVED - now used as categorical feature to distinguish league dynamics
}

TARGETS = {
    "result": {"type": "multiclass", "classes": [-1, 0, 1]},
    "over_25": {"type": "binary", "threshold": 2.5},
    "btts": {"type": "binary", "threshold": None},
}

MARKETS = {
    "1X2_H": {"target": "result", "class_idx": 2, "odds_key": "home"},
    "1X2_D": {"target": "result", "class_idx": 1, "odds_key": "draw"},
    "1X2_A": {"target": "result", "class_idx": 0, "odds_key": "away"},
    "OVER_2.5": {"target": "over_25", "class_idx": 1, "odds_key": "over_2_5"},
    "UNDER_2.5": {"target": "over_25", "class_idx": 0, "odds_key": "under_2_5"},
    "BTTS_YES": {"target": "btts", "class_idx": 1, "odds_key": "btts_yes"},
    "BTTS_NO": {"target": "btts", "class_idx": 0, "odds_key": "btts_no"},
}

MIN_EV = 0.05
MAX_ODDS = 5.0
MIN_ODDS = 1.5
KELLY_FRAC = 0.25

# ──────────────────────────────────────────────
# TRIPLE-SOURCE CONSENSUS (Model + Bet365 + Polymarket)
# ──────────────────────────────────────────────
POLYMARKET_API_KEY = ""  # Set via environment or config
USE_POLYMARKET = False   # Enable when API key available

MIN_KL_DIVERGENCE = 0.02  # Minimum KL divergence to consider signal
CONSENSUS_THRESHOLD = 0.52  # Probability threshold for "agreement"


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D_KL(P || Q) with clipping for numerical stability."""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0, ln2])."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def three_way_kl(p_model: np.ndarray, p_bet365: np.ndarray, p_poly: np.ndarray) -> float:
    """Average pairwise KL divergence between three sources."""
    return (js_divergence(p_model, p_bet365) +
            js_divergence(p_model, p_poly) +
            js_divergence(p_bet365, p_poly)) / 3


def get_consensus_decision(p_model: np.ndarray, p_bet365: np.ndarray, p_poly: np.ndarray) -> tuple:
    """
    Returns (decision, confidence) where decision is:
    - 1: majority says Home/Over/Yes
    - 0: majority says Draw (for 1X2) / no clear majority
    - -1: majority says Away/Under/No
    - None: no consensus (3-way split)
    """
    # For binary markets (over_25, btts): compare p(1) vs 0.5
    # For 1X2: compare all three
    pass


def fetch_polymarket_odds(fixture_id: int, home_team: str, away_team: str, match_date: datetime) -> dict:
    """
    Fetch odds from Polymarket.
    Placeholder - integrate with odds-api.io or Polymarket API when key available.
    Returns dict with 'home', 'draw', 'away', 'over_2_5', 'under_2_5', 'btts_yes', 'btts_no'
    """
    if not USE_POLYMARKET or not POLYMARKET_API_KEY:
        return {}

    # TODO: Implement actual API call
    # Example using odds-api.io:
    # import requests
    # url = f"https://api.odds-api.io/v3/odds/polymarket/{fixture_id}"
    # headers = {"Authorization": f"Bearer {POLYMARKET_API_KEY}"}
    # response = requests.get(url, headers=headers)
    # return parse_polymarket_response(response.json())

    return {}


def odds_to_probs(odds: dict, market: str) -> np.ndarray:
    """Convert odds dict to probability array for a specific market."""
    if market == "result":
        h, d, a = odds.get("home"), odds.get("draw"), odds.get("away")
        if h and d and a:
            inv = np.array([1/h, 1/d, 1/a])
            return inv / inv.sum()
    elif market == "over_25":
        o, u = odds.get("over_2_5"), odds.get("under_2_5")
        if o and u:
            inv = np.array([1/u, 1/o])  # [Under, Over]
            return inv / inv.sum()
    elif market == "btts":
        y, n = odds.get("btts_yes"), odds.get("btts_no")
        if y and n:
            inv = np.array([1/n, 1/y])  # [No, Yes]
            return inv / inv.sum()
    return np.array([])

# ──────────────────────────────────────────────
# EUROPEAN COMPETITION TEAMS (must match features_pre_match.py)
# ──────────────────────────────────────────────
EUROPEAN_COMPETITION_TEAMS = {
    "2026-2027": {
        "UCL": {
            "E0": ["Manchester City", "Arsenal", "Liverpool", "Aston Villa"],
            "SP1": ["Real Madrid", "Barcelona", "Atletico Madrid", "Girona"],
            "D1": ["Bayern Munich", "Bayer Leverkusen", "Stuttgart", "RB Leipzig"],
            "I1": ["Inter", "AC Milan", "Juventus", "Bologna"],
            "F1": ["PSG", "Monaco", "Brest", "Lille"],
        },
        "UEL": {
            "E0": ["Tottenham", "Manchester United"],
            "SP1": ["Real Sociedad", "Athletic Bilbao", "Real Betis"],
            "D1": ["Eintracht Frankfurt", "Freiburg"],
            "I1": ["Roma", "Lazio", "Fiorentina"],
            "F1": ["Lyon", "Nice", "Lens"],
        },
        "UECL": {
            "E0": ["Chelsea"],
            "SP1": ["Villarreal"],
            "D1": ["Heidenheim"],
            "I1": ["Fiorentina"],
            "F1": ["Lens"],
        },
    },
    "2025-2026": {
        "UCL": {
            "E0": ["Manchester City", "Arsenal", "Liverpool", "Aston Villa"],
            "SP1": ["Real Madrid", "Barcelona", "Atletico Madrid", "Girona"],
            "D1": ["Bayern Munich", "Bayer Leverkusen", "Stuttgart", "RB Leipzig"],
            "I1": ["Inter", "AC Milan", "Juventus", "Bologna"],
            "F1": ["PSG", "Monaco", "Brest", "Lille"],
        },
        "UEL": {
            "E0": ["Tottenham", "Manchester United"],
            "SP1": ["Real Sociedad", "Athletic Bilbao", "Real Betis"],
            "D1": ["Eintracht Frankfurt", "Freiburg"],
            "I1": ["Roma", "Lazio", "Fiorentina"],
            "F1": ["Lyon", "Nice", "Lens"],
        },
        "UECL": {
            "E0": ["Chelsea"],
            "SP1": ["Villarreal"],
            "D1": ["Heidenheim"],
            "I1": ["Fiorentina"],
            "F1": ["Lens"],
        },
    },
}

def get_european_comp(team: str, league_code: str, season: str) -> str:
    """Get European competition for a team."""
    season_data = EUROPEAN_COMPETITION_TEAMS.get(season, EUROPEAN_COMPETITION_TEAMS["2026-2027"])
    for comp, leagues in season_data.items():
        if league_code in leagues and team in leagues[league_code]:
            return comp
    return "None"


def estimate_euro_rest(match_date: pd.Timestamp, comp: str) -> tuple:
    """Estimate days since/until European match and rotation risk."""
    if comp == "None":
        return 99, 99, 0.0

    weekday = match_date.weekday()

    if comp == "UCL":
        days_since = {6: 4, 5: 3, 4: 2, 3: 6, 2: 7, 1: 8, 0: 6}.get(weekday, 7)
        days_until = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}.get(weekday, 3)
    else:
        days_since = {6: 3, 0: 4, 1: 5, 2: 6, 3: 7, 4: 1, 5: 2}.get(weekday, 4)
        days_until = {0: 3, 1: 2, 2: 1, 3: 7, 4: 6, 5: 5, 6: 4}.get(weekday, 3)

    rotation_risk = 0.0
    if days_since < 4:
        rotation_risk += 0.4
    if days_until < 4:
        rotation_risk += 0.3
    if days_since < 3:
        rotation_risk += 0.3
    rotation_risk = min(rotation_risk, 1.0)

    return days_since, days_until, rotation_risk


# ──────────────────────────────────────────────
# LINEUP INTEGRATION (Post-lineup prediction adjustment)
# ──────────────────────────────────────────────
async def fetch_lineups_for_fixtures(fixtures: list, hours_before: float = 1.5) -> dict:
    """
    Fetch confirmed lineups for fixtures starting within hours_before.
    Returns dict: {"Home vs Away": lineup_dict, ...}
    """
    try:
        from scrape_lineups import get_lineups_for_fixtures
        return await get_lineups_for_fixtures(fixtures, hours_before)
    except ImportError:
        print("[LINEUPS] scrape_lineups module not available")
        return {}
    except Exception as e:
        print(f"[LINEUPS] Error fetching lineups: {e}")
        return {}


def calculate_lineup_strength(lineup: dict, team_name: str) -> float:
    """
    Calculate lineup strength adjustment factor (0.7 - 1.3).
    1.0 = expected/strongest XI, <1.0 = weakened, >1.0 = stronger than expected.
    """
    if not lineup or not lineup.get("players"):
        return 1.0

    # Key players by team (last names for matching)
    KEY_PLAYERS = {
        "Manchester City": ["Haaland", "De Bruyne", "Rodri", "Dias", "Ederson"],
        "Arsenal": ["Saka", "Odegaard", "Rice", "Saliba", "Raya"],
        "Liverpool": ["Salah", "Van Dijk", "Alisson", "Alexander-Arnold", "Nunez"],
        "Aston Villa": ["Watkins", "Tielemans", "Martinez", "Torres", "McGinn"],
        "Tottenham": ["Son", "Maddison", "Romero", "Van de Ven", "Vicario"],
        "Manchester United": ["Fernandes", "Rashford", "Casemiro", "Martinez", "Onana"],
        "Chelsea": ["Palmer", "Jackson", "Caicedo", "Colwill", "Sanchez"],
        "Real Madrid": ["Vinicius", "Bellingham", "Rodrygo", "Rudiger", "Courtois"],
        "Barcelona": ["Yamal", "Lewandowski", "Pedri", "Cubarsi", "Ter Stegen"],
        "Atletico Madrid": ["Griezmann", "Morata", "Koke", "Gimenez", "Oblak"],
        "Girona": ["Dovbyk", "Tsygankov", "Stuani", "Blind", "Gazzaniga"],
        "Real Sociedad": ["Kubo", "Oyarzabal", "Merino", "Zubimendi", "Remiro"],
        "Athletic Bilbao": ["Williams", "Sancet", "Berenguer", "Vivian", "Simon"],
        "Real Betis": ["Isco", "Fekir", "Guido", "Pezzella", "Rui Silva"],
        "Fiorentina": ["Gonzalez", "Beltran", "Mandragora", "Ranieri", "De Gea"],
        "Bayern Munich": ["Kane", "Musiala", "Kimmich", "Davies", "Neuer"],
        "Bayer Leverkusen": ["Wirtz", "Boniface", "Xhaka", "Hincapie", "Hradecky"],
        "Stuttgart": ["Guirassy", "Mittlestadt", "Undav", "Anton", "Nubel"],
        "RB Leipzig": ["Sesko", "Olmo", "Simons", "Orban", "Gulacsi"],
        "Inter": ["Lautaro", "Thuram", "Barella", "Bastoni", "Sommer"],
        "AC Milan": ["Leao", "Pulisic", "Reijnders", "Tomori", "Maingnan"],
        "Juventus": ["Vlahovic", "Koopmeiners", "Locatelli", "Bremer", "Perin"],
        "Bologna": ["Zirkzee", "Orsolini", "Freuler", "Calafiori", "Skov Olsen"],
        "Roma": ["Dybala", "Pellegrini", "Dovbyk", "Mancini", "Svilar"],
        "Lazio": ["Immobile", "Zaccagni", "Guendouzi", "Romagnoli", "Provedel"],
        "PSG": ["Mbappe", "Dembele", "Barcola", "Vitinha", "Donnarumma"],
        "Monaco": ["Balogun", "Minamino", "Ben Seghir", "Singh", "Majecki"],
        "Brest": ["Le Douaron", "Mounie", "Magnetti", "Chardonnet", "Bizot"],
        "Lille": ["David", "Zhegrova", "Andre", "Diakite", "Chevalier"],
    }

    PLAYER_POSITION_WEIGHTS = {
        "GK": 1.2, "CB": 1.0, "LB": 0.8, "RB": 0.8,
        "DM": 1.0, "CM": 0.9, "AM": 0.9,
        "LW": 1.0, "RW": 1.0, "ST": 1.1,
    }

    key_players = set(KEY_PLAYERS.get(team_name, []))
    starting_names = {p["name"].split()[-1] for p in lineup["players"]}

    key_in_xi = len(key_players & starting_names)
    total_key = len(key_players)

    if total_key == 0:
        return 1.0

    key_ratio = key_in_xi / total_key

    pos_weight = 0
    for p in lineup["players"]:
        pos = p.get("position", "").upper()
        weight = PLAYER_POSITION_WEIGHTS.get(pos, 0.9)
        name_key = p["name"].split()[-1]
        if name_key in key_players:
            weight *= 1.3
        pos_weight += weight

    avg_pos_weight = pos_weight / max(len(lineup["players"]), 1)
    strength = 0.7 * key_ratio + 0.3 * avg_pos_weight

    return max(0.7, min(1.3, strength))


def adjust_features_for_lineups(live_df: pd.DataFrame, lineups: dict) -> pd.DataFrame:
    """
    Adjust live_df features based on confirmed lineups.
    Modifies attack/defense strength and Elo for teams with confirmed lineups.
    """
    df = live_df.copy()

    for idx, row in df.iterrows():
        match_key = f"{row['HomeTeam']} vs {row['AwayTeam']}"
        alt_key = f"{row['AwayTeam']} vs {row['HomeTeam']}"

        lineup_data = lineups.get(match_key) or lineups.get(alt_key)
        if not lineup_data:
            continue

        home_lineup = lineup_data.get("home")
        away_lineup = lineup_data.get("away")

        home_adj = calculate_lineup_strength(home_lineup, row["HomeTeam"])
        away_adj = calculate_lineup_strength(away_lineup, row["AwayTeam"])

        print(f"  [LINEUP ADJ] {row['HomeTeam']}: {home_adj:.2f} | {row['AwayTeam']}: {away_adj:.2f}")

        # Adjust attack/defense rolls
        for w in [5, 10, 20]:
            for prefix, adj in [("home_", home_adj), ("away_", away_adj)]:
                for key in [f"{prefix}attack_roll{w}", f"{prefix}defense_roll{w}", f"{prefix}gd_roll{w}"]:
                    if key in df.columns:
                        df.at[idx, key] = df.at[idx, key] * adj

        # Adjust Elo slightly based on lineup strength
        elo_adj_home = (home_adj - 1.0) * 50  # Max ±15 Elo points
        elo_adj_away = (away_adj - 1.0) * 50
        df.at[idx, "home_elo"] = df.at[idx, "home_elo"] + elo_adj_home
        df.at[idx, "away_elo"] = df.at[idx, "away_elo"] + elo_adj_away
        df.at[idx, "elo_diff"] = df.at[idx, "home_elo"] - df.at[idx, "away_elo"]
        df.at[idx, "elo_sum"] = df.at[idx, "home_elo"] + df.at[idx, "away_elo"]

        # Recompute Elo expected probs
        h_elo = df.at[idx, "home_elo"]
        a_elo = df.at[idx, "away_elo"]
        h_exp = 1 / (1 + 10 ** ((a_elo - h_elo - 65) / 400))
        a_exp = 1 / (1 + 10 ** ((h_elo + 65 - a_elo) / 400))
        d_exp = 1 - h_exp - a_exp
        s = h_exp + d_exp + a_exp
        df.at[idx, "elo_exp_home_win"] = h_exp / s
        df.at[idx, "elo_exp_draw"] = d_exp / s
        df.at[idx, "elo_exp_away_win"] = a_exp / s

        # Mark as post-lineup
        df.at[idx, "post_lineup"] = 1

    return df


# ──────────────────────────────────────────────
# ODDS PORTAL SCRAPER INTEGRATION
# ──────────────────────────────────────────────
async def get_fixtures_from_oddsportal(target_date: str) -> list:
    """Get fixtures from OddsPortal scraper."""
    from scrape_oddsportal import get_fixtures_for_date
    return await get_fixtures_for_date(target_date)


# ──────────────────────────────────────────────
# API-FOOTBALL CALLS
# ──────────────────────────────────────────────
def api_get(endpoint: str, params: dict = None):
    """Call API-football with rate limit handling."""
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if response.status_code == 429:
        raise Exception("Rate limited - wait and retry")
    response.raise_for_status()
    data = response.json()
    if data.get("errors"):
        raise Exception(f"API error: {data['errors']}")
    return data.get("response", [])


def get_fixtures_for_date(target_date: str) -> list:
    """Get fixtures for a specific date (YYYY-MM-DD) in COL timezone."""
    # Convert COL date to UTC range
    col_start = datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=COL_TZ)
    col_end = col_start + timedelta(days=1)
    utc_start = col_start.astimezone(timezone.utc).strftime("%Y-%m-%d")
    utc_end = col_end.astimezone(timezone.utc).strftime("%Y-%m-%d")

    # Determine season year (season starts in August)
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    season_year = target_dt.year if target_dt.month >= 8 else target_dt.year - 1

    all_fixtures = []
    for league_id, league_code in TARGET_LEAGUES.items():
        params = {
            "league": league_id,
            "season": season_year,
            "from": utc_start,
            "to": utc_end,
            "timezone": "America/Bogota"
        }
        fixtures = api_get("fixtures", params)
        for f in fixtures:
            f["league_code"] = league_code
        all_fixtures.extend(fixtures)

    return all_fixtures


def get_odds_for_fixture(fixture_id: int) -> dict:
    """Get pre-match odds for a fixture from multiple bookmakers."""
    params = {"fixture": fixture_id, "bookmaker": "8"}  # Bet365 = 8
    for attempt in range(3):
        try:
            odds_data = api_get("odds", params)
            if not odds_data:
                return {}

            # API returns list of bookmakers, each with bets
            # We want Bet365 (bookmaker 8) main markets
            for bookie in odds_data:
                if bookie.get("bookmaker", {}).get("id") == 8:
                    bets = bookie.get("bets", [])
                    odds = {}
                    for bet in bets:
                        name = bet.get("name", "").lower()
                        values = bet.get("values", [])

                        if name == "match winner":
                            for v in values:
                                if v["value"] == "Home":
                                    odds["home"] = float(v["odd"])
                                elif v["value"] == "Draw":
                                    odds["draw"] = float(v["odd"])
                                elif v["value"] == "Away":
                                    odds["away"] = float(v["odd"])
                        elif "over/under" in name and "2.5" in name:
                            for v in values:
                                if v["value"] == "Over":
                                    odds["over_2_5"] = float(v["odd"])
                                elif v["value"] == "Under":
                                    odds["under_2_5"] = float(v["odd"])
                        elif "both teams to score" in name or "btts" in name:
                            for v in values:
                                if v["value"] == "Yes":
                                    odds["btts_yes"] = float(v["odd"])
                                elif v["value"] == "No":
                                    odds["btts_no"] = float(v["odd"])
                    return odds

            return {}
        except Exception as e:
            if "Rate limited" in str(e) and attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
                continue
            raise
    return {}


# ──────────────────────────────────────────────
# FEATURE ENGINEERING (match historical structure - full 162 features)
# ──────────────────────────────────────────────
def load_historical_data() -> pd.DataFrame:
    """Load historical matches for feature computation."""
    df = pd.read_parquet(FEATURES_FILE)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_elo_ratings(df: pd.DataFrame, ref_date: pd.Timestamp) -> dict:
    """Compute current Elo ratings up to ref_date (walk-forward)."""
    INITIAL_ELO = 1500
    HOME_ADV_ELO = 65
    K_FACTOR = 20

    hist = df[df["Date"] < ref_date].sort_values("Date")
    teams = pd.unique(pd.concat([hist["HomeTeam"], hist["AwayTeam"]]))
    elo = {t: INITIAL_ELO for t in teams}

    for _, row in hist.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        h_elo, a_elo = elo[h], elo[a]

        h_exp = 1 / (1 + 10 ** ((a_elo - h_elo - HOME_ADV_ELO) / 400))
        a_exp = 1 / (1 + 10 ** ((h_elo + HOME_ADV_ELO - a_elo) / 400))

        if row["Result"] == 1:
            h_score, a_score = 1.0, 0.0
        elif row["Result"] == 0:
            h_score, a_score = 0.5, 0.5
        else:
            h_score, a_score = 0.0, 1.0

        elo[h] += K_FACTOR * (h_score - h_exp)
        elo[a] += K_FACTOR * (a_score - a_exp)

    return elo


def compute_team_form_all(df: pd.DataFrame, team: str, ref_date: pd.Timestamp, windows=[3, 5, 10], is_home: bool = True) -> dict:
    """Compute rolling form features for a team matching training feature names exactly."""
    # Build team match history
    home_matches = df[(df["HomeTeam"] == team) & (df["Date"] < ref_date)].copy()
    away_matches = df[(df["AwayTeam"] == team) & (df["Date"] < ref_date)].copy()

    # Home perspective
    home_matches = home_matches.sort_values("Date")
    home_matches["Points"] = home_matches["Result"].map({1: 3, 0: 1, -1: 0})
    home_matches["GF"] = home_matches["FTHG"]
    home_matches["GA"] = home_matches["FTAG"]
    home_matches["CleanSheet"] = (home_matches["FTAG"] == 0).astype(int)
    home_matches["FailedToScore"] = (home_matches["FTHG"] == 0).astype(int)
    home_matches["TotalGoals"] = home_matches["FTHG"] + home_matches["FTAG"]

    # Away perspective
    away_matches = away_matches.sort_values("Date")
    away_matches["Points"] = away_matches["Result"].map({-1: 3, 0: 1, 1: 0})
    away_matches["GF"] = away_matches["FTAG"]
    away_matches["GA"] = away_matches["FTHG"]
    away_matches["CleanSheet"] = (away_matches["FTHG"] == 0).astype(int)
    away_matches["FailedToScore"] = (away_matches["FTAG"] == 0).astype(int)
    away_matches["TotalGoals"] = away_matches["FTHG"] + away_matches["FTAG"]

    # Overall (all matches) - from this team's perspective
    all_matches = pd.concat([home_matches, away_matches]).sort_values("Date")
    all_matches["Points"] = all_matches["Result"].apply(lambda x: 3 if x == 1 else (1 if x == 0 else 0))
    all_matches["GF"] = np.where(all_matches.index.isin(home_matches.index), all_matches["FTHG"], all_matches["FTAG"])
    all_matches["GA"] = np.where(all_matches.index.isin(home_matches.index), all_matches["FTAG"], all_matches["FTHG"])
    all_matches["CleanSheet"] = np.where(all_matches.index.isin(home_matches.index),
                                          (all_matches["FTAG"] == 0).astype(int),
                                          (all_matches["FTHG"] == 0).astype(int))
    all_matches["FailedToScore"] = np.where(all_matches.index.isin(home_matches.index),
                                             (all_matches["FTHG"] == 0).astype(int),
                                             (all_matches["FTAG"] == 0).astype(int))
    all_matches["TotalGoals"] = all_matches["FTHG"] + all_matches["FTAG"]

    prefix = "home_" if is_home else "away_"
    features = {}

    # Venue-specific (H and A)
    for venue, venue_df in [("H", home_matches), ("A", away_matches)]:
        for w in windows:
            if len(venue_df) >= w:
                recent = venue_df.tail(w)
                for col in ["Points", "GF", "GA", "CleanSheet", "FailedToScore", "TotalGoals"]:
                    key = f"{prefix}{col}_roll{w}_{venue}"
                    features[key] = recent[col].mean()
            else:
                for col in ["Points", "GF", "GA", "CleanSheet", "FailedToScore", "TotalGoals"]:
                    key = f"{prefix}{col}_roll{w}_{venue}"
                    features[key] = np.nan

            # Form trends (last 3 vs previous 3) - only for Points, GF, GA
            if len(venue_df) >= 6:
                last3 = venue_df.tail(3)
                prev3 = venue_df.tail(6).head(3)
                for col in ["Points", "GF", "GA"]:
                    key = f"{prefix}{col}_trend{w}_{venue}"
                    features[key] = last3[col].mean() - prev3[col].mean()
            else:
                for col in ["Points", "GF", "GA"]:
                    key = f"{prefix}{col}_trend{w}_{venue}"
                    features[key] = np.nan

    # Overall (no venue suffix) - this is what the model expects as home_Points_roll3, away_Points_roll3
    for w in windows:
        if len(all_matches) >= w:
            recent = all_matches.tail(w)
            for col in ["Points", "GF", "GA", "CleanSheet", "FailedToScore", "TotalGoals"]:
                key = f"{prefix}{col}_roll{w}"
                features[key] = recent[col].mean()
        else:
            for col in ["Points", "GF", "GA", "CleanSheet", "FailedToScore", "TotalGoals"]:
                key = f"{prefix}{col}_roll{w}"
                features[key] = np.nan

        # Form trends for overall
        if len(all_matches) >= 6:
            last3 = all_matches.tail(3)
            prev3 = all_matches.tail(6).head(3)
            for col in ["Points", "GF", "GA"]:
                key = f"{prefix}{col}_trend{w}"
                features[key] = last3[col].mean() - prev3[col].mean()
        else:
            for col in ["Points", "GF", "GA"]:
                key = f"{prefix}{col}_trend{w}"
                features[key] = np.nan

    return features


def compute_h2h_features(df: pd.DataFrame, home: str, away: str, ref_date: pd.Timestamp, max_matches=10) -> dict:
    """Compute H2H features."""
    past = df[
        ((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
        ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
    ][df["Date"] < ref_date].tail(max_matches)

    if len(past) == 0:
        return {
            "h2h_n": 0, "h2h_home_wins": 0, "h2h_away_wins": 0, "h2h_draws": 0,
            "h2h_home_goals_avg": 0, "h2h_away_goals_avg": 0, "h2h_total_goals_avg": 0
        }

    h_as_home = past[past["HomeTeam"] == home]
    h_as_away = past[past["AwayTeam"] == home]

    h_wins = len(h_as_home[h_as_home["Result"] == 1]) + len(h_as_away[h_as_away["Result"] == -1])
    a_wins = len(h_as_home[h_as_home["Result"] == -1]) + len(h_as_away[h_as_away["Result"] == 1])
    draws = len(past[past["Result"] == 0])

    return {
        "h2h_n": len(past),
        "h2h_home_wins": h_wins,
        "h2h_away_wins": a_wins,
        "h2h_draws": draws,
        "h2h_home_goals_avg": h_as_home["FTHG"].mean() if len(h_as_home) else 0,
        "h2h_away_goals_avg": h_as_away["FTAG"].mean() if len(h_as_away) else 0,
        "h2h_total_goals_avg": (past["FTHG"] + past["FTAG"]).mean(),
    }


def compute_schedule_features(df: pd.DataFrame, team: str, ref_date: pd.Timestamp, league_code: str, season: str) -> tuple:
    """
    Rest days, fixture congestion, midweek/weekend, season phase, European competition.
    Returns (team_features, match_features) where match_features are match-level (not prefixed).
    """
    team_matches = df[
        ((df["HomeTeam"] == team) | (df["AwayTeam"] == team)) & (df["Date"] < ref_date)
    ].sort_values("Date")

    if len(team_matches) == 0:
        team_feats = {
            "rest_days": 7, "long_rest": 0, "short_rest": 0,
            "games_last_7": 0, "games_last_14": 0, "games_last_21": 0, "games_last_30": 0,
            "euro_comp": "None", "euro_matchday": 0, "euro_days_since": 99,
            "euro_days_until": 99, "rotation_risk": 0.0
        }
        match_feats = {
            "season_day": 0, "season_month": ref_date.month,
            "season_phase": 0 if ref_date.month in [8,9,10] else (1 if ref_date.month in [11,12,1] else (2 if ref_date.month in [2,3,4] else 3)),
            "is_midweek": 1 if ref_date.dayofweek in [1, 2, 3] else 0,
        }
        return team_feats, match_feats

    last_match = team_matches.iloc[-1]["Date"]
    rest_days = (ref_date - last_match).days
    rest_days = np.clip(rest_days, 1, 21)

    team_feats = {
        "rest_days": rest_days,
        "long_rest": 1 if rest_days > 10 else 0,
        "short_rest": 1 if rest_days < 4 else 0,
        "games_last_7": len(team_matches[team_matches["Date"] >= ref_date - timedelta(days=7)]),
        "games_last_14": len(team_matches[team_matches["Date"] >= ref_date - timedelta(days=14)]),
        "games_last_21": len(team_matches[team_matches["Date"] >= ref_date - timedelta(days=21)]),
        "games_last_30": len(team_matches[team_matches["Date"] >= ref_date - timedelta(days=30)]),
    }

    # Season day (days since first match of season for this team)
    season_start = team_matches[team_matches["Date"].dt.year == ref_date.year]["Date"].min()
    if pd.notna(season_start):
        team_feats["season_day"] = (ref_date - season_start).days
    else:
        team_feats["season_day"] = 0

    # European competition features
    comp = get_european_comp(team, league_code, season)
    team_feats["euro_comp"] = comp
    team_feats["euro_matchday"] = 1 if comp != "None" else 0
    days_since, days_until, rot_risk = estimate_euro_rest(ref_date, comp)
    team_feats["euro_days_since"] = days_since
    team_feats["euro_days_until"] = days_until
    team_feats["rotation_risk"] = rot_risk

    # Match-level features (same for both teams)
    month = ref_date.month
    match_feats = {
        "season_day": team_feats["season_day"],  # Use home team's season_day as match-level
        "season_month": ref_date.month,
        "season_phase": 0 if month in [8, 9, 10] else (1 if month in [11, 12, 1] else (2 if month in [2, 3, 4] else 3)),
        "is_midweek": 1 if ref_date.dayofweek in [1, 2, 3] else 0,
    }

    return team_feats, match_feats


def compute_market_features(odds: dict) -> dict:
    """Compute implied probabilities and market features from odds."""
    features = {}
    if not odds:
        return features

    h = odds.get("home")
    d = odds.get("draw")
    a = odds.get("away")
    o25 = odds.get("over_2_5")
    u25 = odds.get("under_2_5")

    if h and d and a:
        inv_h, inv_d, inv_a = 1/h, 1/d, 1/a
        total_inv = inv_h + inv_d + inv_a
        imp_h, imp_d, imp_a = inv_h/total_inv, inv_d/total_inv, inv_a/total_inv

        features["imp_prob_H"] = imp_h
        features["imp_prob_D"] = imp_d
        features["imp_prob_A"] = imp_a
        features["overround"] = total_inv
        features["imp_prob_total"] = 1.0

        favorite_map = {"imp_prob_H": 1, "imp_prob_D": 0, "imp_prob_A": -1}
        fav_key = max([("imp_prob_H", imp_h), ("imp_prob_D", imp_d), ("imp_prob_A", imp_a)], key=lambda x: x[1])[0]
        features["favorite"] = favorite_map[fav_key]
        features["favorite_prob"] = max(imp_h, imp_d, imp_a)
        features["odds_ratio_H_A"] = h / a if a else 1
        features["draw_prob"] = imp_d
        features["home_win_prob"] = imp_h
        features["away_win_prob"] = imp_a

        # Market entropy
        eps = 1e-10
        p_h, p_d, p_a = max(imp_h, eps), max(imp_d, eps), max(imp_a, eps)
        features["market_entropy"] = -(p_h * np.log(p_h) + p_d * np.log(p_d) + p_a * np.log(p_a))
        features["market_max_prob"] = max(imp_h, imp_d, imp_a)

    if o25 and u25:
        inv_o, inv_u = 1/o25, 1/u25
        total_inv_ou = inv_o + inv_u
        features["imp_prob_O25"] = inv_o / total_inv_ou
        features["imp_prob_U25"] = inv_u / total_inv_ou
        features["ou25_favorite"] = 1 if features["imp_prob_O25"] > features["imp_prob_U25"] else 0

    return features


def compute_strength_features(df: pd.DataFrame, team: str, ref_date: pd.Timestamp, venue: str, windows=[5, 10, 20]) -> dict:
    """Compute attack/defense strength for a team at a venue."""
    if venue == "H":
        team_df = df[(df["HomeTeam"] == team) & (df["Date"] < ref_date)].copy()
        gf_col, ga_col = "FTHG", "FTAG"
    else:
        team_df = df[(df["AwayTeam"] == team) & (df["Date"] < ref_date)].copy()
        gf_col, ga_col = "FTAG", "FTHG"

    team_df = team_df.sort_values("Date")
    prefix = "home_" if venue == "H" else "away_"
    features = {}

    for w in windows:
        if len(team_df) >= w:
            recent = team_df.tail(w)
            features[f"{prefix}attack_roll{w}"] = recent[gf_col].mean()
            features[f"{prefix}defense_roll{w}"] = recent[ga_col].mean()
            features[f"{prefix}gd_roll{w}"] = features[f"{prefix}attack_roll{w}"] - features[f"{prefix}defense_roll{w}"]
        else:
            features[f"{prefix}attack_roll{w}"] = np.nan
            features[f"{prefix}defense_roll{w}"] = np.nan
            features[f"{prefix}gd_roll{w}"] = np.nan

    return features


def compute_league_avg_goals(df: pd.DataFrame, league_code: str, ref_date: pd.Timestamp) -> float:
    """League average goals from historical data (expanding mean)."""
    league_hist = df[(df["Div"] == league_code) & (df["Date"] < ref_date)].sort_values("Date")
    if len(league_hist) == 0:
        return 2.5
    return (league_hist["FTHG"] + league_hist["FTAG"]).mean()


def build_live_features(fixtures: list, historical_df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """Build feature rows for live fixtures matching historical structure (162 features)."""
    print(f"[FEATURES] Building features for {len(fixtures)} fixtures...")

    # Pre-compute Elo
    elo_ratings = compute_elo_ratings(historical_df, ref_date)

    # League averages
    league_avg_goals = {}
    for league_code in ["E0", "SP1", "D1", "I1", "F1"]:
        league_avg_goals[league_code] = compute_league_avg_goals(historical_df, league_code, ref_date)

    rows = []
    for f in fixtures:
        fixture = f["fixture"]
        teams = f["teams"]
        league = f["league"]
        odds = f.get("odds", {})

        fixture_id = fixture["id"]
        home_team = teams["home"]["name"]
        away_team = teams["away"]["name"]
        league_code = league.get("code", f.get("league_code", "UNK"))
        match_date = pd.Timestamp(fixture["date"])
        if match_date.tz is not None:
            match_date = match_date.tz_convert(COL_TZ).tz_localize(None)

        # Compute all feature groups
        season = "2026-2027"  # Could be derived from match_date
        home_form = compute_team_form_all(historical_df, home_team, match_date, is_home=True)
        away_form = compute_team_form_all(historical_df, away_team, match_date, is_home=False)
        h2h = compute_h2h_features(historical_df, home_team, away_team, match_date)
        home_sched_team, home_sched_match = compute_schedule_features(historical_df, home_team, match_date, league_code, season)
        away_sched_team, away_sched_match = compute_schedule_features(historical_df, away_team, match_date, league_code, season)
        market = compute_market_features(odds)
        home_strength = compute_strength_features(historical_df, home_team, match_date, "H")
        away_strength = compute_strength_features(historical_df, away_team, match_date, "A")

        # Combined rotation risk features (match-level)
        home_rot = home_sched_team.get("rotation_risk", 0.0)
        away_rot = away_sched_team.get("rotation_risk", 0.0)
        match_rotation_risk = (home_rot + away_rot) / 2
        home_rotation_advantage = 1 if away_rot > home_rot else 0
        away_rotation_advantage = 1 if home_rot > away_rot else 0

        home_elo = elo_ratings.get(home_team, 1500)
        away_elo = elo_ratings.get(away_team, 1500)
        elo_diff = home_elo - away_elo
        elo_sum = home_elo + away_elo

        # Elo expected probabilities
        h_exp = 1 / (1 + 10 ** ((away_elo - home_elo - 65) / 400))
        a_exp = 1 / (1 + 10 ** ((home_elo + 65 - away_elo) / 400))
        d_exp = 1 - h_exp - a_exp
        s = h_exp + d_exp + a_exp
        elo_exp_home = h_exp / s
        elo_exp_draw = d_exp / s
        elo_exp_away = a_exp / s

        row = {
            "Date": match_date,
            "Season": "2026-2027",
            "Div": league_code,  # Will be converted to category in generate_predictions
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "Fixture_ID": fixture_id,
            # Elo
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": elo_diff,
            "elo_sum": elo_sum,
            "elo_exp_home_win": elo_exp_home,
            "elo_exp_draw": elo_exp_draw,
            "elo_exp_away_win": elo_exp_away,
            # Home form (venue-specific + overall + trends)
            **home_form,
            # Away form (venue-specific + overall + trends)
            **away_form,
            # H2H
            **h2h,
            # Schedule (home) - team-level features prefixed
            **{f"home_{k}": v for k, v in home_sched_team.items()},
            # Schedule (away) - team-level features prefixed
            **{f"away_{k}": v for k, v in away_sched_team.items()},
            # Schedule (match-level) - no prefix
            **home_sched_match,  # is_midweek, season_day, season_month, season_phase
            # Combined rotation risk (match-level)
            "match_rotation_risk": match_rotation_risk,
            "home_rotation_advantage": home_rotation_advantage,
            "away_rotation_advantage": away_rotation_advantage,
            # Market features (from odds)
            **market,
            # Strength (home)
            **home_strength,
            # Strength (away)
            **away_strength,
            # League avg
            "league_avg_goals": league_avg_goals.get(league_code, 2.5),
            # Odds (for evaluation only, not features)
            # Both formats: historical (B365*) and live (home/draw/away) for MARKETS compatibility
            "home": odds.get("home"),
            "draw": odds.get("draw"),
            "away": odds.get("away"),
            "over_2_5": odds.get("over_2_5"),
            "under_2_5": odds.get("under_2_5"),
            "btts_yes": odds.get("btts_yes"),
            "btts_no": odds.get("btts_no"),
            "B365H": odds.get("home"),
            "B365D": odds.get("draw"),
            "B365A": odds.get("away"),
            "B365>2.5": odds.get("over_2_5"),
            "B365<2.5": odds.get("under_2_5"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add interaction features
    df = add_interaction_features(df)

    # Fill NaN with historical medians
    for col in df.columns:
        if col in EXCLUDE_FEATURES or col.startswith("target_"):
            continue
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            median_val = historical_df[col].median() if col in historical_df.columns else 0
            df[col] = df[col].fillna(median_val)

    # Ensure Div is category (match training data)
    if "Div" in df.columns:
        df["Div"] = df["Div"].astype("category")

    return df

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between home/away team stats."""
    df = df.copy()

    # Form differentials
    for w in [5, 10]:
        # Points diff
        h_pts = df.get(f"home_Points_roll{w}", 0)
        a_pts = df.get(f"away_Points_roll{w}", 0)
        df[f"form_pts_diff_roll{w}"] = h_pts - a_pts

        # Attack diff
        h_att = df.get(f"home_attack_roll{w}", 0)
        a_att = df.get(f"away_attack_roll{w}", 0)
        df[f"form_att_diff_roll{w}"] = h_att - a_att

        # Defense diff
        h_def = df.get(f"home_defense_roll{w}", 0)
        a_def = df.get(f"away_defense_roll{w}", 0)
        df[f"form_def_diff_roll{w}"] = h_def - a_def

    # Rest advantage
    df["rest_diff"] = df.get("home_rest_days", 0) - df.get("away_rest_days", 0)
    df["rest_advantage"] = (df["rest_diff"] > 2).astype(int) - (df["rest_diff"] < -2).astype(int)

    # Congestion differential
    for w in [7, 14]:
        h_cong = df.get(f"home_games_last_{w}", 0)
        a_cong = df.get(f"away_games_last_{w}", 0)
        df[f"congestion_diff_{w}"] = h_cong - a_cong

    return df


# ──────────────────────────────────────────────
# MODEL PREDICTION
# ──────────────────────────────────────────────
def load_models():
    """Load trained models and calibrators."""
    models = {}
    calibrators = {}

    for target_name in TARGETS:
        model_path = MODEL_DIR / f"{target_name}_model.txt"
        cal_path = MODEL_DIR / f"{target_name}_calibrator.pkl"

        if model_path.exists():
            models[target_name] = lgb.Booster(model_file=str(model_path))
            with open(cal_path, "rb") as f:
                calibrators[target_name] = joblib.load(f)
            print(f"  Loaded {target_name}")
        else:
            print(f"  WARNING: {target_name} model not found!")

    return models, calibrators


def predict_calibrated(model, X, calibrators, task_type):
    probs = model.predict(X, num_iteration=model.best_iteration)
    if task_type == "multiclass":
        cal_probs = np.column_stack([
            calibrators[i].predict(probs[:, i]) for i in range(probs.shape[1])
        ])
        cal_probs = cal_probs / cal_probs.sum(axis=1, keepdims=True)
        return cal_probs
    else:
        return calibrators.predict(probs)


def generate_predictions(live_df: pd.DataFrame, models: dict, calibrators: dict) -> pd.DataFrame:
    """Run models on live fixtures."""
    print("\n[PREDICT] Generating predictions...")

    # Get feature columns (same as training)
    historical_df = pd.read_parquet(FEATURES_FILE)
    historical_df, feat_cols, cat_cols = prepare_data(historical_df)

    # Ensure live_df has same columns
    for c in feat_cols:
        if c not in live_df.columns:
            live_df[c] = historical_df[c].median()

    X_live = live_df[feat_cols].copy()
    for c in cat_cols:
        if c in X_live.columns:
            X_live[c] = X_live[c].astype("category")

    # Predict each target
    for target_name, target_info in TARGETS.items():
        if target_name not in models:
            continue
        model = models[target_name]
        cal = calibrators[target_name]
        probs = predict_calibrated(model, X_live, cal, target_info["type"])

        if target_info["type"] == "multiclass":
            live_df[f"pred_{target_name}_A"] = probs[:, 0]
            live_df[f"pred_{target_name}_D"] = probs[:, 1]
            live_df[f"pred_{target_name}_H"] = probs[:, 2]
        else:
            live_df[f"pred_{target_name}_1"] = probs
            live_df[f"pred_{target_name}_0"] = 1 - probs

    return live_df


def prepare_data(df: pd.DataFrame) -> tuple:
    """Same prep as training."""
    df = df.sort_values("Date").reset_index(drop=True)
    df["target_result"] = df["Result"].map({-1: 0, 0: 1, 1: 2}) if "Result" in df.columns else 0
    df["target_over_25"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int) if "FTHG" in df.columns else 0
    df["target_btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int) if "FTHG" in df.columns else 0

    feat_cols = [c for c in df.columns if c not in EXCLUDE_FEATURES and not c.startswith("target_")]
    cat_cols = df[feat_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype("category")

    num_cols = [c for c in feat_cols if c not in cat_cols]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    return df, feat_cols, cat_cols


def evaluate_bets(live_df: pd.DataFrame) -> pd.DataFrame:
    """Find EV+ bets with triple-source consensus (Model + Bet365 + Polymarket)."""
    print("\n[BETS] Evaluating betting opportunities (triple-source)...")
    all_rows = []

    for _, row in live_df.iterrows():
        fixture_id = row["Fixture_ID"]
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        match_date = row["Date"]

        # Fetch Polymarket odds for this fixture
        poly_odds = fetch_polymarket_odds(fixture_id, home_team, away_team, match_date)

        for mkt_name, mkt in MARKETS.items():
            odds_key = mkt["odds_key"]
            odds_bet365 = row.get(odds_key)
            target = mkt["target"]
            class_idx = mkt["class_idx"]

            # Get model probability
            if target == "result":
                p_model = row.get(f"pred_result_{['A','D','H'][class_idx]}", 0)
                model_probs = np.array([
                    row.get("pred_result_A", 0),
                    row.get("pred_result_D", 0),
                    row.get("pred_result_H", 0)
                ])
            elif target == "over_25":
                p_model = row.get(f"pred_over_25_{class_idx}", 0)
                model_probs = np.array([
                    row.get("pred_over_25_0", 0),
                    row.get("pred_over_25_1", 0)
                ])
            elif target == "btts":
                p_model = row.get(f"pred_btts_{class_idx}", 0)
                model_probs = np.array([
                    row.get("pred_btts_0", 0),
                    row.get("pred_btts_1", 0)
                ])
            else:
                continue

            if p_model <= 0 or odds_bet365 is None or pd.isna(odds_bet365):
                continue
            if odds_bet365 < MIN_ODDS or odds_bet365 > MAX_ODDS:
                continue

            # Bet365 implied probabilities
            bet365_probs = odds_to_probs(row, target)
            if len(bet365_probs) == 0:
                continue

            # Polymarket implied probabilities
            poly_probs = odds_to_probs(poly_odds, target) if poly_odds else np.array([])

            # Triple-source analysis
            kl_score = 0.0
            consensus = None
            poly_agree = False
            bet365_agree = False

            if len(poly_probs) > 0 and len(model_probs) > 0:
                # KL divergence between three sources
                kl_score = three_way_kl(model_probs, bet365_probs, poly_probs)

                # Consensus check (2 vs 1)
                consensus, confidence = get_consensus_decision(model_probs, bet365_probs, poly_probs)

                # Check if Polymarket agrees with model on the predicted outcome
                model_pred = np.argmax(model_probs)
                poly_pred = np.argmax(poly_probs)
                bet365_pred = np.argmax(bet365_probs)
                poly_agree = (model_pred == poly_pred)
                bet365_agree = (model_pred == bet365_pred)

                # Skip if all three agree (no edge per tweet strategy)
                if poly_agree and bet365_agree:
                    continue  # No divergence = no edge

                # Require minimum KL divergence for signal
                if kl_score < MIN_KL_DIVERGENCE:
                    continue

                # Require 2-vs-1 consensus (model + one other)
                if consensus is None or consensus != model_pred - 1:  # model_pred: 0=A, 1=D, 2=H
                    continue

            # EV calculation using model probability
            ev = p_model * odds_bet365 - 1
            if ev <= MIN_EV:
                continue

            kelly = (p_model * odds_bet365 - 1) / (odds_bet365 - 1) if odds_bet365 > 1 else 0
            stake = max(0, kelly * KELLY_FRAC)
            clv = (p_model - 1/odds_bet365) * 100

            all_rows.append({
                "Date": row["Date"].strftime("%Y-%m-%d %H:%M"),
                "League": row["Div"],
                "Home": home_team,
                "Away": away_team,
                "Market": mkt_name,
                "Model_Prob": round(p_model, 4),
                "Bet365_Prob": round(bet365_probs[class_idx], 4) if len(bet365_probs) > class_idx else 0,
                "Poly_Prob": round(poly_probs[class_idx], 4) if len(poly_probs) > class_idx else 0,
                "Odds": odds_bet365,
                "EV%": round(ev * 100, 2),
                "Stake%": round(stake * 100, 2),
                "CLV%": round(clv, 2),
                "KL_Score": round(kl_score, 4),
                "Consensus": "2-vs-1" if consensus is not None else "None",
                "Poly_Agree": poly_agree,
                "Fixture_ID": fixture_id,
            })

    return pd.DataFrame(all_rows)


def get_consensus_decision(p_model: np.ndarray, p_bet365: np.ndarray, p_poly: np.ndarray) -> tuple:
    """
    Returns (decision, confidence) where decision is the majority class index.
    For 1X2: 0=Away, 1=Draw, 2=Home
    For binary: 0=Under/No, 1=Over/Yes
    Returns (None, 0) if 3-way split.
    """
    n_classes = len(p_model)
    preds = [
        int(np.argmax(p_model)),
        int(np.argmax(p_bet365)),
        int(np.argmax(p_poly))
    ]

    # Count votes
    from collections import Counter
    votes = Counter(preds)
    majority_class, count = votes.most_common(1)[0]

    if count >= 2:
        # 2-vs-1 or 3-vs-0 consensus
        confidence = count / 3.0
        return majority_class, confidence
    else:
        # 3-way split (1-1-1)
        return None, 0.0


def print_bets(bets_df: pd.DataFrame):
    """Pretty print betting recommendations."""
    if len(bets_df) == 0:
        print("\n[NO BETS] NO EV+ BETS FOUND")
        return

    print(f"\n{'='*120}")
    print(f"[BETS] BETTING RECOMMENDATIONS - {len(bets_df)} EV+ opportunities (Triple-Source)")
    print(f"{'='*120}")
    print(f"{'Date':<18} {'League':<5} {'Match':<32} {'Market':<10} {'M%':>5} {'B%':>5} {'P%':>5} {'Odds':>6} {'EV%':>7} {'Stake%':>7} {'KL':>5} {'Cons':<6}")
    print(f"{'-'*120}")

    for _, row in bets_df.sort_values("EV%", ascending=False).iterrows():
        match = f"{row['Home'][:15]:<15} vs {row['Away'][:15]:<15}"
        m_prob = row.get('Model_Prob', 0)
        b_prob = row.get('Bet365_Prob', 0)
        p_prob = row.get('Poly_Prob', 0)
        kl = row.get('KL_Score', 0)
        cons = row.get('Consensus', 'N/A')
        print(f"{row['Date']:<18} {row['League']:<5} {match:<32} {row['Market']:<10} "
              f"{m_prob:>5.1%} {b_prob:>5.1%} {p_prob:>5.1%} {row['Odds']:>6.2f} {row['EV%']:>7.2f}% "
              f"{row['Stake%']:>7.2f}% {kl:>5.3f} {cons:<6}")

    # Summary
    total_stake = bets_df["Stake%"].sum()
    avg_ev = bets_df["EV%"].mean()
    avg_clv = bets_df["CLV%"].mean()
    avg_kl = bets_df["KL_Score"].mean() if "KL_Score" in bets_df.columns else 0
    print(f"{'-'*120}")
    print(f"TOTAL: {len(bets_df)} bets | Avg EV: {avg_ev:.2f}% | Avg CLV: {avg_clv:.2f}% | Avg KL: {avg_kl:.4f} | Total Stake: {total_stake:.2f}% bankroll")
    print(f"{'='*120}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main(target_date: str = None, source: str = "api-football", pass_mode: str = "pre"):
    """
    Analyze fixtures for target date (YYYY-MM-DD in COL timezone).

    Args:
        target_date: Date in YYYY-MM-DD format (COL timezone)
        source: "api-football" or "oddsportal"
        pass_mode: "pre" (pre-lineup, early) or "post" (post-lineup, 60-90 min before kickoff)
    """
    if target_date is None:
        # Tomorrow in COL
        target_date = (datetime.now(COL_TZ) + timedelta(days=1)).strftime("%Y-%m-%d")

    print("=" * 60)
    print(f"LIVE ANALYSIS - {target_date} (COL time) [source: {source}] [pass: {pass_mode}]")
    print("=" * 60)

    # 1. Load historical data
    print("\n[1/6] Loading historical data...")
    historical_df = load_historical_data()
    print(f"       {len(historical_df):,} matches loaded")

    # 2. Fetch fixtures
    print(f"\n[2/6] Fetching fixtures for {target_date} from {source}...")
    if source == "oddsportal":
        import asyncio
        fixtures = asyncio.run(get_fixtures_from_oddsportal(target_date))
    else:
        fixtures = get_fixtures_for_date(target_date)
    print(f"       Found {len(fixtures)} fixtures in target leagues")

    if len(fixtures) == 0:
        print("[NO BETS] No fixtures found for this date")
        return

    # 3. Fetch odds (only for API-football; OddsPortal already has odds)
    if source == "api-football":
        print("\n[3/6] Fetching odds (Bet365)...")
        for i, f in enumerate(fixtures):
            fixture_id = f["fixture"]["id"]
            try:
                odds = get_odds_for_fixture(fixture_id)
                f["odds"] = odds
                if odds:
                    print(f"       [OK] {f['teams']['home']['name']} vs {f['teams']['away']['name']}")
                else:
                    print(f"       [WARN] No odds for {f['teams']['home']['name']} vs {f['teams']['away']['name']}")
            except Exception as e:
                if "Rate limited" in str(e):
                    print(f"       [RATE LIMIT] Stopping odds fetch, continuing without odds for remaining fixtures")
                    for j in range(i, len(fixtures)):
                        fixtures[j]["odds"] = {}
                    break
                else:
                    raise
            if i < len(fixtures) - 1:
                time.sleep(1.5)
    else:
        print("\n[3/6] Using odds from OddsPortal (already included)")
        for f in fixtures:
            odds = f.get("odds", {})
            if odds:
                print(f"       [OK] {f['teams']['home']['name']} vs {f['teams']['away']['name']} - 1X2 odds available")
            else:
                print(f"       [WARN] {f['teams']['home']['name']} vs {f['teams']['away']['name']} - no odds")

    # 4. Build features (pre-lineup)
    print("\n[4/6] Building features (pre-lineup)...")
    ref_date = pd.Timestamp(target_date)  # timezone-naive to match historical data
    live_df = build_live_features(fixtures, historical_df, ref_date)

    # 5. Post-lineup adjustment (if pass_mode == "post")
    if pass_mode == "post":
        print("\n[5/6] Fetching confirmed lineups (60-90 min before kickoff)...")
        import asyncio
        lineups = asyncio.run(fetch_lineups_for_fixtures(fixtures, hours_before=2.0))
        if lineups:
            print(f"       Got lineups for {len(lineups)} matches")
            live_df = adjust_features_for_lineups(live_df, lineups)
        else:
            print("       No lineups available yet (too early or source unavailable)")
    else:
        print("\n[5/6] Pre-lineup pass - skipping lineup fetch")

    # 6. Load models and predict
    print("\n[6/6] Loading models & predicting...")
    models, calibrators = load_models()
    live_df = generate_predictions(live_df, models, calibrators)

    # 7. Evaluate bets
    bets_df = evaluate_bets(live_df)

    # 8. Output
    print_bets(bets_df)

    # Save
    suffix = "_post" if pass_mode == "post" else "_pre"
    output_file = DATA_DIR / f"live_analysis_{target_date}{suffix}.parquet"
    live_df.to_parquet(output_file, index=False)
    print(f"\n[SAVE] Saved full analysis to {output_file}")

    bets_file = DATA_DIR / f"live_bets_{target_date}{suffix}.parquet"
    bets_df.to_parquet(bets_file, index=False)
    print(f"[SAVE] Saved bets to {bets_file}")

    return bets_df


def run_dual_pass(target_date: str = None, source: str = "api-football"):
    """
    Run both pre-lineup and post-lineup passes.
    Pre-pass: Run early (day before / morning of)
    Post-pass: Run 60-90 min before kickoff
    """
    print("=" * 60)
    print("DUAL-PASS PREDICTION")
    print("=" * 60)

    # Pass 1: Pre-lineup
    print("\n>>> PASS 1: PRE-LINEUP (early prediction)")
    pre_bets = main(target_date, source, pass_mode="pre")

    # Pass 2: Post-lineup
    print("\n>>> PASS 2: POST-LINEUP (confirmed XIs)")
    post_bets = main(target_date, source, pass_mode="post")

    # Compare
    if pre_bets is not None and post_bets is not None and len(pre_bets) > 0 and len(post_bets) > 0:
        print("\n" + "=" * 60)
        print("COMPARISON: PRE vs POST LINEUP")
        print("=" * 60)

        # Merge on match + market
        pre_bets["key"] = pre_bets["Home"] + " vs " + pre_bets["Away"] + " | " + pre_bets["Market"]
        post_bets["key"] = post_bets["Home"] + " vs " + post_bets["Away"] + " | " + post_bets["Market"]

        merged = pre_bets.merge(post_bets, on="key", suffixes=("_pre", "_post"))
        for _, row in merged.iterrows():
            prob_diff = row["Model_Prob_post"] - row["Model_Prob_pre"]
            ev_diff = row["EV%_post"] - row["EV%_pre"]
            stake_diff = row["Stake%_post"] - row["Stake%_pre"]
            print(f"  {row['key']}")
            print(f"    Prob: {row['Model_Prob_pre']:.1%} → {row['Model_Prob_post']:.1%} ({prob_diff:+.1%})")
            print(f"    EV:   {row['EV%_pre']:.1f}% → {row['EV%_post']:.1f}% ({ev_diff:+.1f}%)")
            print(f"    Stake: {row['Stake%_pre']:.2f}% → {row['Stake%_post']:.2f}% ({stake_diff:+.2f}%)")

    return pre_bets, post_bets


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("date", nargs="?", help="Target date (YYYY-MM-DD)")
    parser.add_argument("--source", choices=["api-football", "oddsportal"], default="api-football",
                        help="Data source for fixtures and odds")
    parser.add_argument("--pass", dest="pass_mode", choices=["pre", "post", "dual"], default="pre",
                        help="Prediction pass: pre-lineup, post-lineup, or dual (both)")
    args = parser.parse_args()

    if args.pass_mode == "dual":
        run_dual_pass(args.date, args.source)
    else:
        main(args.date, args.source, args.pass_mode)