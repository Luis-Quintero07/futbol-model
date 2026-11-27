#!/usr/bin/env python
"""
Feature Engineering v2 - Advanced free features for football betting model
Adds: form, H2H, rest days, Elo, market features, venue splits, shot quality, etc.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
CLEAN_FILE = DATA_DIR / "matches_clean.parquet"
OUTPUT_FILE = DATA_DIR / "matches_features_v2.parquet"

# ──────────────────────────────────────────────
# ELO RATINGS
# ──────────────────────────────────────────────
INITIAL_ELO = 1500
HOME_ADV_ELO = 65
K_FACTOR = 20

def calculate_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Elo ratings for each team over time (walk-forward)."""
    df = df.sort_values("Date").copy()
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
    elo = {t: INITIAL_ELO for t in teams}

    home_elo_pre = []
    away_elo_pre = []
    home_elo_post = []
    away_elo_post = []

    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        h_elo, a_elo = elo[h], elo[a]

        home_elo_pre.append(h_elo)
        away_elo_pre.append(a_elo)

        # Expected scores
        h_exp = 1 / (1 + 10 ** ((a_elo - h_elo - HOME_ADV_ELO) / 400))
        a_exp = 1 / (1 + 10 ** ((h_elo + HOME_ADV_ELO - a_elo) / 400))

        # Actual result
        if row["Result"] == 1:
            h_score, a_score = 1.0, 0.0
        elif row["Result"] == 0:
            h_score, a_score = 0.5, 0.5
        else:
            h_score, a_score = 0.0, 1.0

        # Update
        elo[h] += K_FACTOR * (h_score - h_exp)
        elo[a] += K_FACTOR * (a_score - a_exp)

        home_elo_post.append(elo[h])
        away_elo_post.append(elo[a])

    df["home_elo_pre"] = home_elo_pre
    df["away_elo_pre"] = away_elo_pre
    df["home_elo_post"] = home_elo_post
    df["away_elo_post"] = away_elo_post
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]

    return df


# ──────────────────────────────────────────────
# TEAM FORM (Rolling windows, home/away specific)
# ──────────────────────────────────────────────
def add_form_features(df: pd.DataFrame, windows=[3, 5, 10]) -> pd.DataFrame:
    """Add form features: points, goals, xG, clean sheets, etc."""
    df = df.sort_values(["Date"]).copy()

    # Create long format for easier rolling
    rows = []
    for _, r in df.iterrows():
        # Home perspective
        rows.append({
            "Date": r["Date"], "Team": r["HomeTeam"], "Opponent": r["AwayTeam"],
            "Venue": "H", "GF": r["FTHG"], "GA": r["FTAG"],
            "xGF": r.get("home_xg_proxy", np.nan), "xGA": r.get("home_xga_proxy", np.nan),
            "Result": 1 if r["Result"] == 1 else (0 if r["Result"] == 0 else -1),
            "Points": 3 if r["Result"] == 1 else (1 if r["Result"] == 0 else 0),
            "CleanSheet": 1 if r["FTAG"] == 0 else 0,
            "FailedToScore": 1 if r["FTHG"] == 0 else 0,
            "BTTS": 1 if r["FTHG"] > 0 and r["FTAG"] > 0 else 0,
        })
        # Away perspective
        rows.append({
            "Date": r["Date"], "Team": r["AwayTeam"], "Opponent": r["HomeTeam"],
            "Venue": "A", "GF": r["FTAG"], "GA": r["FTHG"],
            "xGF": r.get("away_xg_proxy", np.nan), "xGA": r.get("away_xga_proxy", np.nan),
            "Result": -1 if r["Result"] == 1 else (0 if r["Result"] == 0 else 1),
            "Points": 3 if r["Result"] == -1 else (1 if r["Result"] == 0 else 0),
            "CleanSheet": 1 if r["FTHG"] == 0 else 0,
            "FailedToScore": 1 if r["FTAG"] == 0 else 0,
            "BTTS": 1 if r["FTHG"] > 0 and r["FTAG"] > 0 else 0,
        })

    team_df = pd.DataFrame(rows).sort_values(["Team", "Date"])

    # Rolling features per team+venue
    for w in windows:
        for col in ["Points", "GF", "GA", "xGF", "xGA", "CleanSheet", "FailedToScore", "BTTS"]:
            team_df[f"{col}_roll{w}"] = team_df.groupby(["Team", "Venue"])[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
        # Overall (venue agnostic)
        for col in ["Points", "GF", "GA", "xGF", "xGA"]:
            team_df[f"{col}_roll{w}_overall"] = team_df.groupby("Team")[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

    # Merge back
    home_cols = {c: f"home_{c}" for c in team_df.columns if c.endswith(("_roll3", "_roll5", "_roll10"))}
    away_cols = {c: f"away_{c}" for c in team_df.columns if c.endswith(("_roll3", "_roll5", "_roll10"))}

    home_feats = team_df[team_df["Venue"] == "H"][["Date", "Team"] + list(home_cols.keys())].rename(
        columns={"Team": "HomeTeam", **home_cols})
    away_feats = team_df[team_df["Venue"] == "A"][["Date", "Team"] + list(away_cols.keys())].rename(
        columns={"Team": "AwayTeam", **away_cols})

    df = df.merge(home_feats, on=["Date", "HomeTeam"], how="left")
    df = df.merge(away_feats, on=["Date", "AwayTeam"], how="left")

    return df


# ──────────────────────────────────────────────
# H2H (Head-to-Head) HISTORY
# ──────────────────────────────────────────────
def add_h2h_features(df: pd.DataFrame, max_matches=10) -> pd.DataFrame:
    """Add H2H features: last N meetings results, goals, venue-specific."""
    df = df.sort_values("Date").copy()

    h2h_rows = []
    for _, row in df.iterrows():
        h, a, d = row["HomeTeam"], row["AwayTeam"], row["Date"]

        # Historical H2H (both venues)
        past = df[
            ((df["HomeTeam"] == h) & (df["AwayTeam"] == a)) |
            ((df["HomeTeam"] == a) & (df["AwayTeam"] == h))
        ][df["Date"] < d].tail(max_matches)

        if len(past) == 0:
            h2h_rows.append({
                "Date": d, "HomeTeam": h, "AwayTeam": a,
                "h2h_n": 0, "h2h_home_wins": 0, "h2h_away_wins": 0, "h2h_draws": 0,
                "h2h_home_goals_avg": 0, "h2h_away_goals_avg": 0,
                "h2h_home_xg_avg": 0, "h2h_away_xg_avg": 0,
            })
            continue

        # Home team perspective in H2H
        h_as_home = past[past["HomeTeam"] == h]
        h_as_away = past[past["AwayTeam"] == h]

        h_wins = len(h_as_home[h_as_home["Result"] == 1]) + len(h_as_away[h_as_away["Result"] == -1])
        a_wins = len(h_as_home[h_as_home["Result"] == -1]) + len(h_as_away[h_as_away["Result"] == 1])
        draws = len(past[past["Result"] == 0])

        h2h_rows.append({
            "Date": d, "HomeTeam": h, "AwayTeam": a,
            "h2h_n": len(past),
            "h2h_home_wins": h_wins,
            "h2h_away_wins": a_wins,
            "h2h_draws": draws,
            "h2h_home_goals_avg": h_as_home["FTHG"].mean() if len(h_as_home) else 0,
            "h2h_away_goals_avg": h_as_away["FTAG"].mean() if len(h_as_away) else 0,
            "h2h_home_xg_avg": h_as_home["home_xg_proxy"].mean() if "home_xg_proxy" in past.columns and len(h_as_home) else 0,
            "h2h_away_xg_avg": h_as_away["away_xg_proxy"].mean() if "away_xg_proxy" in past.columns and len(h_as_away) else 0,
        })

    h2h_df = pd.DataFrame(h2h_rows)
    return df.merge(h2h_df, on=["Date", "HomeTeam", "AwayTeam"], how="left")


# ──────────────────────────────────────────────
# REST DAYS, SCHEDULE CONGESTION & EUROPEAN COMPETITION
# ──────────────────────────────────────────────
# European competition participants by season (update each season)
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
            "I1": ["Fiorentina"],  # If dropped from UEL
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

def get_european_competition(team: str, league_code: str, season: str) -> dict:
    """Determine European competition for a team in a given season."""
    season_data = EUROPEAN_COMPETITION_TEAMS.get(season, EUROPEAN_COMPETITION_TEAMS["2026-2027"])
    for comp, leagues in season_data.items():
        if league_code in leagues and team in leagues[league_code]:
            return {"comp": comp, "matchday": None}  # matchday would need fixture data
    return {"comp": "None", "matchday": None}

def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rest days, fixture congestion, midweek vs weekend, European competition."""
    df = df.sort_values(["Date"]).copy()

    # Rest days for each team
    for venue, team_col in [("home", "HomeTeam"), ("away", "AwayTeam")]:
        df[f"{venue}_rest_days"] = df.groupby(team_col)["Date"].diff().dt.days
        df[f"{venue}_rest_days"] = df[f"{venue}_rest_days"].fillna(7).clip(1, 14)

        # Games in last 7/14/21 days - use date-based count
        for window in [7, 14, 21]:
            def count_recent(dates):
                return dates.apply(lambda d: ((dates < d) & (dates >= d - pd.Timedelta(days=window))).sum())
            df[f"{venue}_games_last_{window}"] = df.groupby(team_col)["Date"].transform(count_recent)

    # Midweek (Tue/Wed/Thu) vs Weekend
    df["is_midweek"] = df["Date"].dt.weekday.isin([1, 2, 3]).astype(int)

    # Days since season start (fitness/fatigue curve)
    df["season_day"] = df.groupby("Season")["Date"].transform(
        lambda x: (x - x.min()).dt.days
    )

    # ─── EUROPEAN COMPETITION FEATURES ───
    # For each team, determine if they play in UCL (Tue/Wed), UEL (Thu), UECL (Thu)
    # This affects rest days and rotation risk
    for venue, team_col in [("home", "HomeTeam"), ("away", "AwayTeam")]:
        # Initialize
        df[f"{venue}_euro_comp"] = "None"
        df[f"{venue}_euro_matchday"] = 0
        df[f"{venue}_euro_days_since"] = 99  # Days since last Euro match
        df[f"{venue}_euro_days_until"] = 99  # Days until next Euro match
        df[f"{venue}_rotation_risk"] = 0.0  # 0-1 probability of rotation

    # Vectorized approach: map teams to their European competition
    def map_euro_comp(row):
        season = row["Season"]
        home_comp = get_european_competition(row["HomeTeam"], row["Div"], season)
        away_comp = get_european_competition(row["AwayTeam"], row["Div"], season)
        return pd.Series({
            "home_euro_comp": home_comp["comp"],
            "away_euro_comp": away_comp["comp"],
        })

    euro_features = df.apply(map_euro_comp, axis=1)
    df["home_euro_comp"] = euro_features["home_euro_comp"]
    df["away_euro_comp"] = euro_features["away_euro_comp"]

    # Compute days since/until European match for each team
    # We need to find European match dates - approximate from fixture list
    # For now, use heuristic: UCL plays Tue/Wed, UEL/UECL play Thu
    # Matchday typically every 2 weeks during group stage (Sep-Dec, Feb-Mar)

    for venue in ["home", "away"]:
        comp_col = f"{venue}_euro_comp"

        # UCL teams: matchdays typically Tue/Wed
        ucl_mask = df[comp_col] == "UCL"
        df.loc[ucl_mask, f"{venue}_euro_matchday"] = 1

        # UEL/UECL teams: matchdays typically Thu
        uel_mask = df[comp_col].isin(["UEL", "UECL"])
        df.loc[uel_mask, f"{venue}_euro_matchday"] = 1

        # Estimate days since last Euro match (heuristic based on matchday schedule)
        # Group stage: Sep, Oct, Nov, Dec (matchdays 1-6), Knockout: Feb, Mar, Apr
        # Simplified: if team in Euro comp, estimate rest impact
        def estimate_euro_rest(row, venue):
            comp = row[comp_col]
            if comp == "None":
                return 99, 99, 0.0

            match_date = row["Date"]
            weekday = match_date.weekday()  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun

            if comp == "UCL":
                # UCL plays Tue(1)/Wed(2)
                # If match is Sun(6), last UCL was Wed(2) -> 4 days rest
                # If match is Sat(5), last UCL was Wed(2) -> 3 days rest
                # If match is Fri(4), last UCL was Wed(2) -> 2 days rest (HIGH rotation risk)
                days_since = {6: 4, 5: 3, 4: 2, 3: 6, 2: 7, 1: 8, 0: 6}.get(weekday, 7)
                days_until = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}.get(weekday, 3)
            else:
                # UEL/UECL plays Thu(3)
                # If match is Sun(6), last UEL was Thu(3) -> 3 days rest
                # If match is Mon(0), last UEL was Thu(3) -> 4 days rest
                days_since = {6: 3, 0: 4, 1: 5, 2: 6, 3: 7, 4: 1, 5: 2}.get(weekday, 4)
                days_until = {0: 3, 1: 2, 2: 1, 3: 7, 4: 6, 5: 5, 6: 4}.get(weekday, 3)

            # Rotation risk: high if <4 days rest, or if next Euro match in <4 days
            rotation_risk = 0.0
            if days_since < 4:
                rotation_risk += 0.4
            if days_until < 4:
                rotation_risk += 0.3
            if days_since < 3:
                rotation_risk += 0.3  # Very short turnaround
            rotation_risk = min(rotation_risk, 1.0)

            return days_since, days_until, rotation_risk

        rest_data = df.apply(lambda r: estimate_euro_rest(r, venue), axis=1)
        df[f"{venue}_euro_days_since"] = [x[0] for x in rest_data]
        df[f"{venue}_euro_days_until"] = [x[1] for x in rest_data]
        df[f"{venue}_rotation_risk"] = [x[2] for x in rest_data]

    # Combined rotation risk for the match
    df["match_rotation_risk"] = (df["home_rotation_risk"] + df["away_rotation_risk"]) / 2
    df["home_rotation_advantage"] = (df["away_rotation_risk"] > df["home_rotation_risk"]).astype(int)
    df["away_rotation_advantage"] = (df["home_rotation_risk"] > df["away_rotation_risk"]).astype(int)

    return df


# ──────────────────────────────────────────────
# MARKET FEATURES (Odds-based)
# ──────────────────────────────────────────────
def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Implied probabilities, overround, market consensus proxies."""
    df = df.copy()

    # Already have imp_prob_H/D/A and overround from download_data.py
    # Add derived features
    df["imp_prob_total"] = df["imp_prob_H"] + df["imp_prob_D"] + df["imp_prob_A"]
    df["favorite"] = df[["imp_prob_H", "imp_prob_D", "imp_prob_A"]].idxmax(axis=1).map({
        "imp_prob_H": 1, "imp_prob_D": 0, "imp_prob_A": -1
    })
    df["favorite_prob"] = df[["imp_prob_H", "imp_prob_D", "imp_prob_A"]].max(axis=1)
    df["odds_ratio_H_A"] = df["B365H"] / df["B365A"]
    df["draw_prob"] = df["imp_prob_D"]

    # Over/Under 2.5 implied
    if "B365>2.5" in df.columns and "B365<2.5" in df.columns:
        mask = df["B365>2.5"].notna() & df["B365<2.5"].notna()
        inv_o = 1 / df.loc[mask, "B365>2.5"]
        inv_u = 1 / df.loc[mask, "B365<2.5"]
        total_inv = inv_o + inv_u
        df.loc[mask, "imp_prob_O25"] = inv_o / total_inv
        df.loc[mask, "imp_prob_U25"] = inv_u / total_inv

    return df


# ──────────────────────────────────────────────
# SHOT QUALITY & ADVANCED STATS
# ──────────────────────────────────────────────
def add_shot_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """xG per shot, conversion rates, shot dominance."""
    df = df.copy()

    # Shot conversion (goals per shot on target)
    df["home_shot_conv"] = np.where(df["HST"] > 0, df["FTHG"] / df["HST"], 0)
    df["away_shot_conv"] = np.where(df["AST"] > 0, df["FTAG"] / df["AST"], 0)

    # xG per shot (proxy quality)
    df["home_xg_per_shot"] = np.where(df["HS"] > 0, df["home_xg_proxy"] / df["HS"], 0)
    df["away_xg_per_shot"] = np.where(df["AS"] > 0, df["away_xg_proxy"] / df["AS"], 0)
    df["home_xg_per_sot"] = np.where(df["HST"] > 0, df["home_xg_proxy"] / df["HST"], 0)
    df["away_xg_per_sot"] = np.where(df["AST"] > 0, df["away_xg_proxy"] / df["AST"], 0)

    # Shot dominance
    df["shot_dom"] = df["HS"] / (df["HS"] + df["AS"]).replace(0, np.nan)
    df["sot_dom"] = df["HST"] / (df["HST"] + df["AST"]).replace(0, np.nan)
    df["xg_dom"] = df["home_xg_proxy"] / (df["home_xg_proxy"] + df["away_xg_proxy"]).replace(0, np.nan)

    # Corners & set pieces
    df["corner_dom"] = df["HC"] / (df["HC"] + df["AC"]).replace(0, np.nan)
    df["home_corner_rate"] = df["HC"] / df["HS"].replace(0, np.nan)
    df["away_corner_rate"] = df["AC"] / df["AS"].replace(0, np.nan)

    # Discipline
    df["home_card_rate"] = (df["HY"] + 2*df["HR"]) / df["HF"].replace(0, np.nan)
    df["away_card_rate"] = (df["AY"] + 2*df["AR"]) / df["AF"].replace(0, np.nan)

    return df


# ──────────────────────────────────────────────
# VENUE-SPECIFIC STRENGTHS
# ──────────────────────────────────────────────
def add_venue_splits(df: pd.DataFrame, window=10) -> pd.DataFrame:
    """Home/away specific rolling xG, goals, points."""
    df = df.sort_values("Date").copy()

    # Home team at home
    for col in ["FTHG", "FTAG", "home_xg_proxy", "home_xga_proxy", "Points_H"]:
        pass  # Already in form features

    # Additional: home team performance at home vs away team performance away
    # Already captured in form features with venue split

    return df


# ──────────────────────────────────────────────
# GOAL TIMING (First half vs Second half)
# ──────────────────────────────────────────────
def add_goal_timing_features(df: pd.DataFrame, window=10) -> pd.DataFrame:
    """Rolling first half/second half scoring patterns."""
    df = df.copy()

    # FH/2H goals
    df["home_fh_goals"] = df["HTHG"]
    df["home_sh_goals"] = df["FTHG"] - df["HTHG"]
    df["away_fh_goals"] = df["HTAG"]
    df["away_sh_goals"] = df["FTAG"] - df["HTAG"]

    # Rolling FH/SH performance
    for venue, team_col, fh_col, sh_col in [
        ("home", "HomeTeam", "home_fh_goals", "home_sh_goals"),
        ("away", "AwayTeam", "away_fh_goals", "away_sh_goals"),
    ]:
        for w in [5, 10]:
            df[f"{venue}_fh_goals_roll{w}"] = df.groupby(team_col)[fh_col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            df[f"{venue}_sh_goals_roll{w}"] = df.groupby(team_col)[sh_col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

    return df


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("FEATURE ENGINEERING v2")
    print("=" * 60)

    df = pd.read_parquet(CLEAN_FILE)
    print(f"\n[LOAD] {len(df):,} matches from {CLEAN_FILE}")

    # Need xG proxy data - merge from matches_with_xg
    xg_df = pd.read_parquet(DATA_DIR / "matches_with_xg.parquet")
    xg_cols = [c for c in xg_df.columns if c not in df.columns]
    xg_cols = ["Date", "HomeTeam", "AwayTeam"] + xg_cols
    df = df.merge(xg_df[xg_cols], on=["Date", "HomeTeam", "AwayTeam"], how="left")
    print(f"[MERGE] Added {len(xg_cols)-3} xG features")

    print("\n[1/7] Calculating Elo ratings...")
    df = calculate_elo(df)

    print("[2/7] Adding team form features...")
    df = add_form_features(df, windows=[3, 5, 10])

    print("[3/7] Adding H2H features...")
    df = add_h2h_features(df, max_matches=10)

    print("[4/7] Adding schedule/congestion features...")
    df = add_schedule_features(df)

    print("[5/7] Adding market features...")
    df = add_market_features(df)

    print("[6/7] Adding shot quality features...")
    df = add_shot_quality_features(df)

    print("[7/7] Adding goal timing features...")
    df = add_goal_timing_features(df, window=10)

    # Fill NaNs for rolling features (early season)
    rolling_cols = [c for c in df.columns if "_roll" in c.lower()]
    for c in rolling_cols:
        df[c] = df[c].fillna(df[c].median())

    # Save
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n[SAVE] {OUTPUT_FILE} ({len(df):,} rows, {len(df.columns)} cols)")

    # Summary
    print(f"\nFeature categories:")
    cats = {
        "Elo": [c for c in df.columns if "elo" in c.lower()],
        "Form": [c for c in df.columns if "_roll" in c.lower()],
        "H2H": [c for c in df.columns if "h2h" in c.lower()],
        "Schedule": [c for c in df.columns if "rest" in c.lower() or "games_last" in c.lower() or "midweek" in c.lower() or "season_day" in c.lower()],
        "Market": [c for c in df.columns if "imp_prob" in c.lower() or "favorite" in c.lower() or "odds_ratio" in c.lower() or "draw_prob" in c.lower()],
        "Shot Quality": [c for c in df.columns if "conv" in c.lower() or "xg_per" in c.lower() or "_dom" in c.lower() or "corner_rate" in c.lower() or "card_rate" in c.lower()],
        "Goal Timing": [c for c in df.columns if "fh_goals" in c.lower() or "sh_goals" in c.lower()],
    }
    for cat, cols in cats.items():
        if cols:
            print(f"  {cat}: {len(cols)} features")


if __name__ == "__main__":
    main()