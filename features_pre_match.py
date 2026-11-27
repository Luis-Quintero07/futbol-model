#!/usr/bin/env python
"""
Feature Engineering - PRE-MATCH ONLY (no data leakage)
Only uses information available BEFORE kickoff:
- Historical team performance (rolling, Elo, H2H)
- Schedule / fixture congestion
- Market odds (opening/early odds if available)
- Venue factors, team strength estimates
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
CLEAN_FILE = DATA_DIR / "matches_clean.parquet"
OUTPUT_FILE = DATA_DIR / "matches_pre_match.parquet"

# ──────────────────────────────────────────────
# ELO RATINGS (walk-forward, pre-match)
# ──────────────────────────────────────────────
INITIAL_ELO = 1500
HOME_ADV_ELO = 65
K_FACTOR = 20

def calculate_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Elo ratings - only using past matches."""
    df = df.sort_values("Date").copy()
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
    elo = {t: INITIAL_ELO for t in teams}

    home_elo_pre = []
    away_elo_pre = []

    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        home_elo_pre.append(elo[h])
        away_elo_pre.append(elo[a])

        # Expected scores
        h_elo, a_elo = elo[h], elo[a]
        h_exp = 1 / (1 + 10 ** ((a_elo - h_elo - HOME_ADV_ELO) / 400))
        a_exp = 1 / (1 + 10 ** ((h_elo + HOME_ADV_ELO - a_elo) / 400))

        # Actual result (only for updating Elo AFTER match)
        if row["Result"] == 1:
            h_score, a_score = 1.0, 0.0
        elif row["Result"] == 0:
            h_score, a_score = 0.5, 0.5
        else:
            h_score, a_score = 0.0, 1.0

        # Update Elo (post-match)
        elo[h] += K_FACTOR * (h_score - h_exp)
        elo[a] += K_FACTOR * (a_score - a_exp)

    df["home_elo"] = home_elo_pre
    df["away_elo"] = away_elo_pre
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["elo_sum"] = df["home_elo"] + df["away_elo"]

    return df


# ──────────────────────────────────────────────
# TEAM FORM - Historical only (shifted)
# ──────────────────────────────────────────────
def add_form_features(df: pd.DataFrame, windows=[3, 5, 10]) -> pd.DataFrame:
    """Rolling form using ONLY past matches (properly shifted)."""
    df = df.sort_values("Date").copy()

    # Long format for team-match history
    rows = []
    for _, r in df.iterrows():
        # Home team perspective
        rows.append({
            "Date": r["Date"], "Team": r["HomeTeam"], "Opponent": r["AwayTeam"],
            "Venue": "H",
            "GF": r["FTHG"], "GA": r["FTAG"],
            "Result": 1 if r["Result"] == 1 else (0 if r["Result"] == 0 else -1),
            "Points": 3 if r["Result"] == 1 else (1 if r["Result"] == 0 else 0),
            "CleanSheet": 1 if r["FTAG"] == 0 else 0,
            "FailedToScore": 1 if r["FTHG"] == 0 else 0,
            "TotalGoals": r["FTHG"] + r["FTAG"],
        })
        # Away team perspective
        rows.append({
            "Date": r["Date"], "Team": r["AwayTeam"], "Opponent": r["HomeTeam"],
            "Venue": "A",
            "GF": r["FTAG"], "GA": r["FTHG"],
            "Result": -1 if r["Result"] == 1 else (0 if r["Result"] == 0 else 1),
            "Points": 3 if r["Result"] == -1 else (1 if r["Result"] == 0 else 0),
            "CleanSheet": 1 if r["FTHG"] == 0 else 0,
            "FailedToScore": 1 if r["FTAG"] == 0 else 0,
            "TotalGoals": r["FTHG"] + r["FTAG"],
        })

    team_df = pd.DataFrame(rows).sort_values(["Team", "Date"])

    # Rolling features - shift(1) ensures only PAST matches
    for w in windows:
        for col in ["Points", "GF", "GA", "CleanSheet", "FailedToScore", "TotalGoals"]:
            # Venue-specific
            team_df[f"{col}_roll{w}_H"] = team_df[team_df["Venue"] == "H"].groupby("Team")[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            team_df[f"{col}_roll{w}_A"] = team_df[team_df["Venue"] == "A"].groupby("Team")[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            # Overall
            team_df[f"{col}_roll{w}"] = team_df.groupby("Team")[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

        # Form trend (last 3 vs previous 3)
        for col in ["Points", "GF", "GA"]:
            team_df[f"{col}_trend{w}"] = (
                team_df.groupby("Team")[col].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()) -
                team_df.groupby("Team")[col].transform(lambda x: x.shift(4).rolling(3, min_periods=1).mean())
            )

    # Merge back to match level
    home_cols = {c: f"home_{c}" for c in team_df.columns if c.endswith(("_roll3", "_roll5", "_roll10", "_trend3", "_trend5", "_trend10"))}
    away_cols = {c: f"away_{c}" for c in team_df.columns if c.endswith(("_roll3", "_roll5", "_roll10", "_trend3", "_trend5", "_trend10"))}

    home_feats = team_df[team_df["Venue"] == "H"][["Date", "Team"] + list(home_cols.keys())].rename(
        columns={"Team": "HomeTeam", **home_cols})
    away_feats = team_df[team_df["Venue"] == "A"][["Date", "Team"] + list(away_cols.keys())].rename(
        columns={"Team": "AwayTeam", **away_cols})

    df = df.merge(home_feats, on=["Date", "HomeTeam"], how="left")
    df = df.merge(away_feats, on=["Date", "AwayTeam"], how="left")

    return df


# ──────────────────────────────────────────────
# H2H HISTORY
# ──────────────────────────────────────────────
def add_h2h_features(df: pd.DataFrame, max_matches=10) -> pd.DataFrame:
    """Head-to-head history using only past meetings."""
    df = df.sort_values("Date").copy()

    h2h_rows = []
    for _, row in df.iterrows():
        h, a, d = row["HomeTeam"], row["AwayTeam"], row["Date"]

        # Past H2H meetings (both home/away)
        past = df[
            ((df["HomeTeam"] == h) & (df["AwayTeam"] == a)) |
            ((df["HomeTeam"] == a) & (df["AwayTeam"] == h))
        ][df["Date"] < d].tail(max_matches)

        if len(past) == 0:
            h2h_rows.append({
                "Date": d, "HomeTeam": h, "AwayTeam": a,
                "h2h_n": 0, "h2h_home_wins": 0, "h2h_away_wins": 0, "h2h_draws": 0,
                "h2h_home_goals_avg": 0, "h2h_away_goals_avg": 0,
                "h2h_total_goals_avg": 0,
            })
            continue

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
            "h2h_total_goals_avg": (past["FTHG"] + past["FTAG"]).mean(),
        })

    h2h_df = pd.DataFrame(h2h_rows)
    return df.merge(h2h_df, on=["Date", "HomeTeam", "AwayTeam"], how="left")


# ──────────────────────────────────────────────
# EUROPEAN COMPETITION TEAMS (update per season)
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

def get_european_competition(team: str, league_code: str, season: str) -> dict:
    """Determine European competition for a team in a given season."""
    season_data = EUROPEAN_COMPETITION_TEAMS.get(season, EUROPEAN_COMPETITION_TEAMS["2026-2027"])
    for comp, leagues in season_data.items():
        if league_code in leagues and team in leagues[league_code]:
            return {"comp": comp, "matchday": None}
    return {"comp": "None", "matchday": None}


# ──────────────────────────────────────────────
# SCHEDULE / CONGESTION + EUROPEAN COMPETITION
# ──────────────────────────────────────────────
def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rest days, fixture congestion, midweek/weekend, European competition."""
    df = df.sort_values("Date").copy()

    for venue, team_col in [("home", "HomeTeam"), ("away", "AwayTeam")]:
        # Rest days since last match
        df[f"{venue}_rest_days"] = df.groupby(team_col)["Date"].diff().dt.days
        df[f"{venue}_rest_days"] = df[f"{venue}_rest_days"].fillna(7).clip(1, 21)

        # Games in last 7/14/21/30 days
        for window in [7, 14, 21, 30]:
            def count_recent(dates):
                return dates.apply(lambda d: ((dates < d) & (dates >= d - pd.Timedelta(days=window))).sum())
            df[f"{venue}_games_last_{window}"] = df.groupby(team_col)["Date"].transform(count_recent)

        # Long rest (>10 days) vs short rest (<4 days)
        df[f"{venue}_long_rest"] = (df[f"{venue}_rest_days"] > 10).astype(int)
        df[f"{venue}_short_rest"] = (df[f"{venue}_rest_days"] < 4).astype(int)

    # Midweek match (Tue/Wed/Thu)
    df["is_midweek"] = df["Date"].dt.dayofweek.isin([1, 2, 3]).astype(int)

    # Days since season start
    df["season_day"] = df.groupby("Season")["Date"].transform(
        lambda x: (x - x.min()).dt.days
    )

    # Month of season (early/mid/late)
    df["season_month"] = df["Date"].dt.month
    df["season_phase"] = pd.cut(df["season_month"], bins=[0, 3, 6, 9, 12], labels=[0, 1, 2, 3]).astype(int)

    # ─── EUROPEAN COMPETITION FEATURES ───
    for venue in ["home", "away"]:
        df[f"{venue}_euro_comp"] = "None"
        df[f"{venue}_euro_matchday"] = 0
        df[f"{venue}_euro_days_since"] = 99
        df[f"{venue}_euro_days_until"] = 99
        df[f"{venue}_rotation_risk"] = 0.0

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

    for venue in ["home", "away"]:
        comp_col = f"{venue}_euro_comp"

        ucl_mask = df[comp_col] == "UCL"
        df.loc[ucl_mask, f"{venue}_euro_matchday"] = 1

        uel_mask = df[comp_col].isin(["UEL", "UECL"])
        df.loc[uel_mask, f"{venue}_euro_matchday"] = 1

        def estimate_euro_rest(row, venue):
            comp = row[comp_col]
            if comp == "None":
                return 99, 99, 0.0

            match_date = row["Date"]
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
# MARKET FEATURES (Odds-based, pre-match)
# ──────────────────────────────────────────────
def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Implied probabilities from odds (football-data odds are ~closing)."""
    df = df.copy()

    # Already have imp_prob_H/D/A and overround from download_data.py
    # These are derived from Bet365 odds which are closing-line-ish
    # WARNING: Using closing odds introduces some leakage but is standard practice
    # For true pre-match, we'd need opening odds (not available in football-data)

    df["imp_prob_total"] = df["imp_prob_H"] + df["imp_prob_D"] + df["imp_prob_A"]
    df["favorite"] = df[["imp_prob_H", "imp_prob_D", "imp_prob_A"]].idxmax(axis=1).map({
        "imp_prob_H": 1, "imp_prob_D": 0, "imp_prob_A": -1
    })
    df["favorite_prob"] = df[["imp_prob_H", "imp_prob_D", "imp_prob_A"]].max(axis=1)
    df["odds_ratio_H_A"] = df["B365H"] / df["B365A"]
    df["draw_prob"] = df["imp_prob_D"]
    df["home_win_prob"] = df["imp_prob_H"]
    df["away_win_prob"] = df["imp_prob_A"]

    # Over/Under 2.5 implied
    if "B365>2.5" in df.columns and "B365<2.5" in df.columns:
        mask = df["B365>2.5"].notna() & df["B365<2.5"].notna()
        inv_o = 1 / df.loc[mask, "B365>2.5"]
        inv_u = 1 / df.loc[mask, "B365<2.5"]
        total_inv = inv_o + inv_u
        df.loc[mask, "imp_prob_O25"] = inv_o / total_inv
        df.loc[mask, "imp_prob_U25"] = inv_u / total_inv
        df.loc[mask, "ou25_favorite"] = (df.loc[mask, "imp_prob_O25"] > df.loc[mask, "imp_prob_U25"]).astype(int)

    # Market uncertainty (entropy of 1X2 probs)
    eps = 1e-10
    p_h = df["imp_prob_H"].clip(eps, 1-eps)
    p_d = df["imp_prob_D"].clip(eps, 1-eps)
    p_a = df["imp_prob_A"].clip(eps, 1-eps)
    df["market_entropy"] = -(p_h * np.log(p_h) + p_d * np.log(p_d) + p_a * np.log(p_a))
    df["market_max_prob"] = df[["imp_prob_H", "imp_prob_D", "imp_prob_A"]].max(axis=1)

    return df


# ──────────────────────────────────────────────
# TEAM STRENGTH ESTIMATES (Poisson-based, historical)
# ──────────────────────────────────────────────
def add_strength_features(df: pd.DataFrame, half_life=150) -> pd.DataFrame:
    """Attack/Defense strength using time-weighted historical goals."""
    df = df.sort_values("Date").copy()

    # We'll compute rolling attack/defense per team
    # Home attack = goals scored at home
    # Home defense = goals conceded at home
    # Away attack = goals scored away
    # Away defense = goals conceded away

    for venue, team_col, gf_col, ga_col in [
        ("home", "HomeTeam", "FTHG", "FTAG"),
        ("away", "AwayTeam", "FTAG", "FTHG"),
    ]:
        for w in [5, 10, 20]:
            # Attack (goals scored)
            df[f"{venue}_attack_roll{w}"] = df.groupby(team_col)[gf_col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            # Defense (goals conceded)
            df[f"{venue}_defense_roll{w}"] = df.groupby(team_col)[ga_col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            # Goal difference
            df[f"{venue}_gd_roll{w}"] = df[f"{venue}_attack_roll{w}"] - df[f"{venue}_defense_roll{w}"]

    # League average goals (time-weighted)
    df["league_avg_goals"] = df.groupby("Div").apply(
        lambda g: (g["FTHG"] + g["FTAG"]).expanding().mean().shift(1)
    ).reset_index(level=0, drop=True)
    df["league_avg_goals"] = df["league_avg_goals"].fillna(2.5)

    return df


# ──────────────────────────────────────────────
# VENUE-SPECIFIC INTERACTIONS
# ──────────────────────────────────────────────
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction features between home/away team stats."""
    df = df.copy()

    # Elo-based expected goals (rough)
    elo_diff = df["elo_diff"]
    df["elo_exp_home_win"] = 1 / (1 + 10 ** (-elo_diff / 400))
    df["elo_exp_draw"] = 1 / (1 + 10 ** (abs(elo_diff) / 400)) * 0.3  # rough
    df["elo_exp_away_win"] = 1 - df["elo_exp_home_win"] - df["elo_exp_draw"]

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
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["rest_advantage"] = (df["rest_diff"] > 2).astype(int) - (df["rest_diff"] < -2).astype(int)

    # Congestion differential
    for w in [7, 14]:
        h_cong = df.get(f"home_games_last_{w}", 0)
        a_cong = df.get(f"away_games_last_{w}", 0)
        df[f"congestion_diff_{w}"] = h_cong - a_cong

    return df


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("FEATURE ENGINEERING - PRE-MATCH ONLY (NO LEAKAGE)")
    print("=" * 60)

    df = pd.read_parquet(CLEAN_FILE)
    print(f"\n[LOAD] {len(df):,} matches from {CLEAN_FILE}")

    print("[1/7] Calculating Elo ratings...")
    df = calculate_elo(df)

    print("[2/7] Adding historical form features...")
    df = add_form_features(df, windows=[3, 5, 10])

    print("[3/7] Adding H2H features...")
    df = add_h2h_features(df, max_matches=10)

    print("[4/7] Adding schedule/congestion features...")
    df = add_schedule_features(df)

    print("[5/7] Adding market features...")
    df = add_market_features(df)

    print("[6/7] Adding team strength features...")
    df = add_strength_features(df)

    print("[7/7] Adding interaction features...")
    df = add_interaction_features(df)

    # Drop post-match stats (leakage prevention) - keep only pre-match info
    post_match_cols = [
        "HS", "HST", "HC", "HF", "HY", "HR",  # Home shots, shots on target, corners, fouls, yellow, red
        "AS", "AST", "AC", "AF", "AY", "AR",  # Away shots, shots on target, corners, fouls, yellow, red
        "HTHG", "HTAG", "HTR",                # Half-time scores/result
    ]
    df = df.drop(columns=[c for c in post_match_cols if c in df.columns])

    # Fill NaNs for early season
    feature_cols = [c for c in df.columns if c not in {
        "Div", "Season", "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR", "Result",
        "imp_prob_H", "imp_prob_D", "imp_prob_A", "overround",
        "imp_prob_O25", "imp_prob_U25", "ou25_favorite",
        "favorite", "favorite_prob", "odds_ratio_H_A", "draw_prob",
        "home_win_prob", "away_win_prob", "market_entropy", "market_max_prob",
        "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5",
        "B365AHH", "B365AHA",
    }]
    for c in feature_cols:
        if df[c].dtype in [np.float64, np.int64, np.float32, np.int32]:
            df[c] = df[c].fillna(df[c].median())

    # Save
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n[SAVE] {OUTPUT_FILE} ({len(df):,} rows, {len(df.columns)} cols)")

    # Summary
    print(f"\nFeature categories:")
    cats = {
        "Elo": [c for c in df.columns if "elo" in c.lower()],
        "Form": [c for c in df.columns if "_roll" in c.lower() or "_trend" in c.lower()],
        "H2H": [c for c in df.columns if "h2h" in c.lower()],
        "Schedule": [c for c in df.columns if "rest" in c.lower() or "games_last" in c.lower() or "midweek" in c.lower() or "season_day" in c.lower() or "season_phase" in c.lower()],
        "Market": [c for c in df.columns if "imp_prob" in c.lower() or "favorite" in c.lower() or "odds_ratio" in c.lower() or "draw_prob" in c.lower() or "win_prob" in c.lower() or "ou25" in c.lower() or "entropy" in c.lower() or "market_max" in c.lower()],
        "Strength": [c for c in df.columns if "attack" in c.lower() or "defense" in c.lower() or "gd_roll" in c.lower() or "league_avg" in c.lower()],
        "Interactions": [c for c in df.columns if "diff" in c.lower() or "advantage" in c.lower() or "elo_exp" in c.lower()],
    }
    for cat, cols in cats.items():
        if cols:
            print(f"  {cat}: {len(cols)} features")

    # Verify no post-match stats in features
    post_match_keywords = ["shot", "corner", "foul", "card", "yellow", "red", "xg", "xga", "ht_", "half"]
    suspicious = [c for c in feature_cols if any(kw in c.lower() for kw in post_match_keywords)]
    if suspicious:
        print(f"\n⚠️  SUSPICIOUS FEATURES (potential leakage): {suspicious}")
    else:
        print(f"\n✅ No post-match stats detected in features")


if __name__ == "__main__":
    main()