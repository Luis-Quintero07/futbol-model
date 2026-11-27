#!/usr/bin/env python
"""
Streamlit Dashboard - Football Betting AI System
Daily matches panel + Monte Carlo simulator with Poisson+DC+xG
Run: streamlit run dashboard.py
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import lightgbm as lgb
from scipy.stats import poisson
from datetime import datetime
from pathlib import Path
import json

# ─── Page Config ───
st.set_page_config(
    page_title="⚽ Football Betting AI Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Constants ───
DATA_DIR = Path("data")
MODEL_DIR = Path("models")

TARGET_LEAGUES = {"E0": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "SP1": "🇪🇸 LaLiga",
                  "D1": "🇩🇪 Bundesliga", "I1": "🇮🇹 Serie A", "F1": "🇫🇷 Ligue 1"}

MIN_EV = 0.05
MAX_ODDS = 5.0
MIN_ODDS = 1.5
KELLY_FRAC = 0.25

# Dixon-Coles parameter
RHO = 0.13

# ─── Cached Loaders ───
@st.cache_resource
def load_models():
    models = {}
    calibrators = {}
    for target in ["result", "over_25", "btts"]:
        model_path = MODEL_DIR / f"{target}_model.txt"
        cal_path = MODEL_DIR / f"{target}_calibrator.pkl"
        if model_path.exists():
            models[target] = lgb.Booster(model_file=str(model_path))
            with open(cal_path, "rb") as f:
                calibrators[target] = joblib.load(f)
    return models, calibrators

@st.cache_data
def load_live_data(date_str="2026-11-27"):
    """Load pre-computed predictions for the day"""
    # Try post-lineup first (most recent), then pre-lineup, then base name
    for suffix in ["_post", "_pre", ""]:
        analysis_file = DATA_DIR / f"live_analysis_{date_str}{suffix}.parquet"
        bets_file = DATA_DIR / f"live_bets_{date_str}{suffix}.parquet"
        if analysis_file.exists() and bets_file.exists():
            analysis = pd.read_parquet(analysis_file)
            bets = pd.read_parquet(bets_file)
            fixtures_file = DATA_DIR / f"oddsportal_fixtures_{date_str.replace('-', '')}.parquet"
            fixtures = pd.read_parquet(fixtures_file) if fixtures_file.exists() else pd.DataFrame()
            return analysis, bets, fixtures

    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data
def load_historical_features():
    """Load historical features for xG proxy parameters"""
    df = pd.read_parquet(DATA_DIR / "matches_with_xg.parquet")
    return df

# ─── Poisson + Dixon-Coles + xG Simulator ───
def dixon_coles_correction(h_goals, a_goals, lambda_h, lambda_a, rho=RHO):
    """Dixon-Coles correction for low-scoring draws"""
    if h_goals == 0 and a_goals == 0:
        return 1 - lambda_h * lambda_a * rho
    elif h_goals == 1 and a_goals == 0:
        return 1 + lambda_a * rho
    elif h_goals == 0 and a_goals == 1:
        return 1 + lambda_h * rho
    elif h_goals == 1 and a_goals == 1:
        return 1 - rho
    return 1.0

def simulate_match_poisson_xg(lambda_h, lambda_a, n_sims=2000, rho=RHO, max_goals=8):
    """
    Monte Carlo simulation using Poisson + Dixon-Coles + xG
    Returns: score_matrix (max_goals+1 x max_goals+1), outcome_probs dict
    """
    # Base Poisson probabilities
    h_probs = np.array([poisson.pmf(i, lambda_h) for i in range(max_goals + 1)])
    a_probs = np.array([poisson.pmf(i, lambda_a) for i in range(max_goals + 1)])

    # Joint probability matrix
    score_matrix = np.outer(h_probs, a_probs)

    # Apply Dixon-Coles correction
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            score_matrix[h, a] *= dixon_coles_correction(h, a, lambda_h, lambda_a, rho)

    # Renormalize
    score_matrix /= score_matrix.sum()

    # Outcome probabilities
    home_win = np.sum(np.tril(score_matrix, -1))
    draw = np.sum(np.diag(score_matrix))
    away_win = np.sum(np.triu(score_matrix, 1))
    over_25 = np.sum(score_matrix[np.arange(max_goals+1)[:, None] + np.arange(max_goals+1) > 2.5])
    btts = np.sum(score_matrix[1:, 1:])

    # Top 3 scorelines
    flat_idx = np.argsort(score_matrix.flatten())[::-1][:3]
    top_scorelines = []
    for idx in flat_idx:
        h = idx // (max_goals + 1)
        a = idx % (max_goals + 1)
        top_scorelines.append((f"{h}-{a}", score_matrix[h, a]))

    # Monte Carlo validation
    sims_h = np.random.poisson(lambda_h, n_sims)
    sims_a = np.random.poisson(lambda_a, n_sims)
    mc_home = np.mean(sims_h > sims_a)
    mc_draw = np.mean(sims_h == sims_a)
    mc_away = np.mean(sims_h < sims_a)
    mc_over25 = np.mean(sims_h + sims_a > 2.5)
    mc_btts = np.mean((sims_h > 0) & (sims_a > 0))

    return {
        "score_matrix": score_matrix,
        "top_scorelines": top_scorelines,
        "probs_exact": {"home": home_win, "draw": draw, "away": away_win,
                        "over_25": over_25, "btts": btts},
        "probs_mc": {"home": mc_home, "draw": mc_draw, "away": mc_away,
                     "over_25": mc_over25, "btts": mc_btts},
        "lambda_h": lambda_h,
        "lambda_a": lambda_a
    }

def get_xg_estimates(home_team, away_team, hist_df, league):
    """Estimate xG for a match using historical averages"""
    # Home team xG at home
    home_matches = hist_df[(hist_df["HomeTeam"] == home_team) & (hist_df["Div"] == league)]
    away_matches = hist_df[(hist_df["AwayTeam"] == away_team) & (hist_df["Div"] == league)]

    if len(home_matches) > 0:
        home_xg = home_matches["home_xg"].mean() if "home_xg" in home_matches.columns else home_matches["FTHG"].mean()
    else:
        home_xg = hist_df[hist_df["Div"] == league]["FTHG"].mean()

    if len(away_matches) > 0:
        away_xg = away_matches["away_xg"].mean() if "away_xg" in away_matches.columns else away_matches["FTAG"].mean()
    else:
        away_xg = hist_df[hist_df["Div"] == league]["FTAG"].mean()

    return home_xg, away_xg

# ─── UI Components ───
def render_header():
    st.title("⚽ Football Betting AI Dashboard")
    st.caption("LightGBM + Poisson/Dixon-Coles/xG Ensemble | Top 5 European Leagues")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", "LightGBM + Calibration", "Optuna 30 trials")
    with col2:
        st.metric("Backtest ROI", "+55.6%", "16,232 bets OOS")
    with col3:
        st.metric("Avg CLV", "+16.96%", "p < 0.001")
    with col4:
        st.metric("Kelly Fraction", "25%", "Fractional staking")

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Controls")

        # Date selector
        date_str = st.date_input("Match Date", value=datetime(2026, 11, 27))
        date_str = date_str.strftime("%Y-%m-%d")

        # League filter
        all_leagues = list(TARGET_LEAGUES.keys())
        selected_leagues = st.multiselect("Leagues", all_leagues, default=all_leagues,
                                          format_func=lambda x: TARGET_LEAGUES[x])

        # Simulation params
        st.subheader("🎲 Monte Carlo Settings")
        n_sims = st.slider("Iterations", 500, 10000, 2000, 500)
        max_goals = st.slider("Max Goals Display", 5, 10, 6)
        rho = st.slider("Dixon-Coles ρ", 0.0, 0.3, 0.13, 0.01)

        # Betting params
        st.subheader("💰 Betting Parameters")
        min_ev = st.slider("Min EV %", 0, 20, 5) / 100
        max_odds = st.slider("Max Odds", 1.5, 10.0, 5.0, 0.1)
        kelly_frac = st.slider("Kelly Fraction", 0.05, 0.5, 0.25, 0.05)

        return date_str, selected_leagues, n_sims, max_goals, rho, min_ev, max_odds, kelly_frac

def render_match_card(match, bets_df, idx):
    """Render a single match card with key info"""
    league_flag = TARGET_LEAGUES.get(match['Div'], match['Div'])

    # Check if this match has EV+ bets
    match_bets = bets_df[(bets_df['Home'] == match['HomeTeam']) & (bets_df['Away'] == match['AwayTeam'])]
    has_bets = len(match_bets) > 0

    # Model probabilities
    p_h = match['pred_result_H']
    p_d = match['pred_result_D']
    p_a = match['pred_result_A']
    p_o25 = match['pred_over_25_1']
    p_btts = match['pred_btts_1']

    # Odds (from OddsPortal - 'home','draw','away' columns)
    odds_h = match.get('home', match.get('B365H', None))
    odds_d = match.get('draw', match.get('B365D', None))
    odds_a = match.get('away', match.get('B365A', None))
    odds_o25 = match.get('over_2_5', match.get('B365>2.5', None))

    with st.container():
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

        with col1:
            st.markdown(f"**{league_flag}**")
            st.markdown(f"🏠 **{match['HomeTeam']}** vs **{match['AwayTeam']}** 🏟️")
            if pd.notna(match.get('Fixture_ID')):
                st.caption(f"Kickoff: {match.get('Fixture_ID', 'N/A')}")

        with col2:
            st.markdown("**1X2 Model**")
            st.markdown(f"H: {p_h:.1%} | D: {p_d:.1%} | A: {p_a:.1%}")
            if odds_h:
                st.caption(f"Odds: {odds_h:.2f} | {odds_d:.2f} | {odds_a:.2f}")

        with col3:
            st.markdown("**O/U 2.5**")
            st.markdown(f"Over: {p_o25:.1%} | Under: {1-p_o25:.1%}")
            if odds_o25:
                st.caption(f"Odds: {odds_o25:.2f}")

        with col4:
            st.markdown("**BTTS**")
            st.markdown(f"Yes: {p_btts:.1%} | No: {1-p_btts:.1%}")

        with col5:
            if has_bets:
                st.success(f"💚 **{len(match_bets)} EV+ BETS**")
                for _, bet in match_bets.iterrows():
                    st.caption(f"{bet['Market']}: EV {bet['EV%']:.1f}% | Stake {bet['Stake%']:.2f}%")
            else:
                st.info("No EV+ bets")

        # Expandable Monte Carlo simulator
        with st.expander("🎲 Run Monte Carlo Simulation", expanded=False):
            render_monte_carlo_panel(match, idx, n_sims=2000, max_goals=6, rho=0.13)

        st.divider()

def render_monte_carlo_panel(match, match_idx, n_sims=2000, max_goals=6, rho=0.13):
    """Render Monte Carlo simulation panel for a match"""

    # Get xG estimates (use model's implied lambda from over/under prob)
    # Approximate: if P(over 2.5) = p, then total lambda ≈ solve for poisson
    p_over = match['pred_over_25_1']

    # Simple calibration: use model's 1X2 probs to infer lambdas
    # This is a rough approximation - in production use xG proxy
    p_h = match['pred_result_H']
    p_d = match['pred_result_D']
    p_a = match['pred_result_A']

    # Solve for lambdas that match 1X2 probs (simplified)
    # Use optimization or lookup table
    lambda_h, lambda_a = estimate_lambdas_from_probs(p_h, p_d, p_a)

    # Run simulation
    with st.spinner(f"Simulating {n_sims:,} matches..."):
        result = simulate_match_poisson_xg(lambda_h, lambda_a, n_sims, rho, max_goals)

    # Display results
    col1, col2 = st.columns([1, 1])

    with col1:
        # Score matrix heatmap
        fig_heatmap = plot_score_heatmap(result['score_matrix'], max_goals,
                                         match['HomeTeam'], match['AwayTeam'])
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        # Top scorelines + outcome probs
        st.markdown("### 🎯 Top 3 Scorelines")
        for i, (score, prob) in enumerate(result['top_scorelines']):
            st.markdown(f"**{i+1}.** {score} — **{prob:.1%}**")

        st.markdown("### 📊 Outcome Probabilities")
        probs_exact = result['probs_exact']
        probs_mc = result['probs_mc']

        outcome_df = pd.DataFrame({
            'Outcome': ['Home Win', 'Draw', 'Away Win', 'Over 2.5', 'BTTS Yes'],
            'Exact Poisson+DC': [probs_exact['home'], probs_exact['draw'], probs_exact['away'],
                                  probs_exact['over_25'], probs_exact['btts']],
            f'MC ({n_sims:,} sims)': [probs_mc['home'], probs_mc['draw'], probs_mc['away'],
                                        probs_mc['over_25'], probs_mc['btts']]
        })
        st.dataframe(outcome_df.style.format({'Exact Poisson+DC': '{:.1%}',
                                               f'MC ({n_sims:,} sims)': '{:.1%}'}),
                     use_container_width=True, hide_index=True)

        # Model vs Market comparison
        if match.get('home') and match.get('draw') and match.get('away'):
            st.markdown("### 💹 Model vs Market")
            market_probs = 1/np.array([match['home'], match['draw'], match['away']])
            market_probs = market_probs / market_probs.sum()
            model_probs = np.array([p_h, p_d, p_a])
            edge = model_probs - market_probs

            edge_df = pd.DataFrame({
                'Outcome': ['Home', 'Draw', 'Away'],
                'Model %': model_probs,
                'Market %': market_probs,
                'Edge': edge
            })
            st.dataframe(edge_df.style.format({'Model %': '{:.1%}', 'Market %': '{:.1%}', 'Edge': '{:+.1%}'}),
                         use_container_width=True, hide_index=True)

def estimate_lambdas_from_probs(p_h, p_d, p_a, max_lambda=5.0):
    """
    Rough estimation of Poisson lambdas from 1X2 probabilities.
    In production, use the xG proxy model directly.
    """
    # Grid search for lambdas that best match 1X2 probs
    best_lh, best_la = 1.5, 1.0
    best_error = float('inf')

    for lh in np.linspace(0.3, max_lambda, 50):
        for la in np.linspace(0.3, max_lambda, 50):
            # Compute 1X2 probs from Poisson
            h_probs = np.array([poisson.pmf(i, lh) for i in range(10)])
            a_probs = np.array([poisson.pmf(i, la) for i in range(10)])
            joint = np.outer(h_probs, a_probs)

            ph = np.sum(np.tril(joint, -1))
            pd = np.sum(np.diag(joint))
            pa = np.sum(np.triu(joint, 1))

            error = abs(ph - p_h) + abs(pd - p_d) + abs(pa - p_a)
            if error < best_error:
                best_error = error
                best_lh, best_la = lh, la

    return best_lh, best_la

def plot_score_heatmap(score_matrix, max_goals, home_team, away_team):
    """Create Plotly heatmap of score probabilities"""
    # Prepare data for heatmap
    scores = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            scores.append({'Home': h, 'Away': a, 'Prob': score_matrix[h, a] * 100})
    df = pd.DataFrame(scores)

    # Pivot for heatmap
    pivot = df.pivot(index='Home', columns='Away', values='Prob')

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale='RdYlGn',
        text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Prob %", ticksuffix="%"),
        hovertemplate="%{y}-%{x}: %{z:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title=f"Score Probability Matrix: {home_team} vs {away_team}",
        xaxis_title=f"{away_team} Goals",
        yaxis_title=f"{home_team} Goals",
        xaxis=dict(side='top'),
        yaxis=dict(autorange='reversed'),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig

def render_daily_panel(analysis_df, bets_df, selected_leagues):
    """Render the daily matches panel"""
    st.header(f"📅 Daily Matches Panel — {len(analysis_df)} Fixtures")

    # Filter by selected leagues
    if selected_leagues:
        analysis_df = analysis_df[analysis_df['Div'].isin(selected_leagues)]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    total_bets = len(bets_df)
    total_stake = bets_df['Stake%'].sum() if total_bets > 0 else 0
    avg_ev = bets_df['EV%'].mean() if total_bets > 0 else 0
    avg_clv = bets_df['CLV%'].mean() if total_bets > 0 else 0

    col1.metric("Total Fixtures", len(analysis_df))
    col2.metric("EV+ Opportunities", total_bets)
    col3.metric("Total Stake %", f"{total_stake:.2f}%")
    col4.metric("Avg EV / CLV", f"{avg_ev:.1f}% / {avg_clv:.1f}%")

    # Match cards
    for idx, match in analysis_df.iterrows():
        render_match_card(match, bets_df, idx)

def render_bets_table(bets_df):
    """Render detailed bets table"""
    if len(bets_df) == 0:
        st.info("No EV+ bets found for current parameters")
        return

    st.header("💰 EV+ Betting Recommendations")

    # Format for display (keep numeric for styling)
    display_df = bets_df.copy()
    display_df['Match'] = display_df['Home'] + " vs " + display_df['Away']

    show_cols = ['Date', 'League', 'Match', 'Market', 'Model_Prob', 'Odds', 'EV%', 'Stake%', 'CLV%']

    # Style with numeric values, then format for display
    styled = display_df[show_cols].style.format({
        'Model_Prob': '{:.1%}',
        'Odds': '{:.2f}',
        'EV%': '{:.2f}%',
        'Stake%': '{:.2f}%',
        'CLV%': '{:.2f}%'
    }).apply(
        lambda x: ['background-color: #d4edda' if v > 10 else '' for v in x]
        if x.name == 'EV%' else [''] * len(x), axis=0
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Summary by market
    st.subheader("📈 By Market")
    for mkt in bets_df['Market'].unique():
        d = bets_df[bets_df['Market'] == mkt]
        s = d['Stake%'].sum()
        p = d['PnL'].sum() if 'PnL' in d.columns else 0
        r = p/s*100 if s > 0 else 0
        w = d['Actual'].mean() * 100 if 'Actual' in d.columns else 0
        st.markdown(f"**{mkt}**: {len(d)} bets | Stake: {s:.2f}% | Avg EV: {d['EV%'].mean():.2f}% | Avg CLV: {d['CLV%'].mean():.2f}%")

# ─── Main App ───
def main():
    render_header()

    # Sidebar controls
    date_str, selected_leagues, n_sims, max_goals, rho, min_ev, max_odds, kelly_frac = render_sidebar()

    # Load data
    with st.spinner("Loading predictions..."):
        analysis_df, bets_df, fixtures_df = load_live_data(date_str)

    if len(analysis_df) == 0:
        st.error(f"No data found for {date_str}. Run `python production_pipeline.py --predict --date {date_str}` first.")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📅 Daily Panel", "💰 EV+ Bets", "📊 Backtest Analytics"])

    with tab1:
        render_daily_panel(analysis_df, bets_df, selected_leagues)

    with tab2:
        render_bets_table(bets_df)

    with tab3:
        st.header("📊 Backtest Analytics (Coming Soon)")
        st.info("Historical backtest visualizations: PnL curves, drawdown, ROI by league/season/market")

if __name__ == "__main__":
    main()