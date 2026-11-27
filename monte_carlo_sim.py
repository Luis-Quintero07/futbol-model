#!/usr/bin/env python
"""
Monte Carlo simulation for match outcomes using Poisson + Dixon-Coles
Estimates lambdas from ML model probabilities, runs 10k simulations
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import poisson
import warnings
warnings.filterwarnings("ignore")

# Dixon-Coles parameter
RHO = 0.13  # Standard DC parameter for low-scoring correction

def dixon_coles_prob(h_goals, a_goals, lambda_h, lambda_a, rho=RHO):
    """Dixon-Coles adjusted probability for (h_goals, a_goals)"""
    # Base Poisson
    p = poisson.pmf(h_goals, lambda_h) * poisson.pmf(a_goals, lambda_a)

    # DC correction for low scores
    if h_goals == 0 and a_goals == 0:
        tau = 1 - lambda_h * lambda_a * rho
    elif h_goals == 1 and a_goals == 0:
        tau = 1 + lambda_a * rho
    elif h_goals == 0 and a_goals == 1:
        tau = 1 + lambda_h * rho
    elif h_goals == 1 and a_goals == 1:
        tau = 1 - rho
    else:
        tau = 1.0

    return p * tau

def match_probs_from_lambdas(lambda_h, lambda_a, max_goals=10):
    """Compute 1X2, Over 2.5, BTTS probabilities from lambdas using DC"""
    h_win = d_win = a_win = over25 = btts = 0.0

    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = dixon_coles_prob(hg, ag, lambda_h, lambda_a)
            if hg > ag:
                h_win += p
            elif hg == ag:
                d_win += p
            else:
                a_win += p

            if hg + ag > 2.5:
                over25 += p
            if hg > 0 and ag > 0:
                btts += p

    return {
        'H': h_win, 'D': d_win, 'A': a_win,
        'over_25': over25, 'under_25': 1 - over25,
        'btts_yes': btts, 'btts_no': 1 - btts
    }

def objective_lambdas(params, target_probs):
    """Objective: minimize squared error between DC probs and target ML probs"""
    lambda_h, lambda_a = params
    if lambda_h <= 0 or lambda_a <= 0:
        return 1e6
    dc_probs = match_probs_from_lambdas(lambda_h, lambda_a)
    error = 0.0
    for key in ['H', 'D', 'A', 'over_25', 'btts_yes']:
        target = target_probs.get(key, 0)
        if target > 0:
            error += (dc_probs[key] - target) ** 2
    return error

def estimate_lambdas(ml_probs, init_guess=(1.5, 1.0)):
    """Estimate Poisson lambdas that match ML model probabilities"""
    # Target probabilities from ML model
    target = {
        'H': ml_probs['pred_result_H'],
        'D': ml_probs['pred_result_D'],
        'A': ml_probs['pred_result_A'],
        'over_25': ml_probs['pred_over_25_1'],
        'btts_yes': ml_probs['pred_btts_1']
    }

    # Grid search for good starting point
    best_error = float('inf')
    best_params = init_guess

    for lh in np.linspace(0.5, 3.5, 13):
        for la in np.linspace(0.5, 3.0, 11):
            error = objective_lambdas((lh, la), target)
            if error < best_error:
                best_error = error
                best_params = (lh, la)

    # Refine with optimization
    result = minimize_scalar(
        lambda lh: objective_lambdas((lh, best_params[1]), target),
        bounds=(0.3, 4.0), method='bounded'
    )
    best_lh = result.x

    result = minimize_scalar(
        lambda la: objective_lambdas((best_lh, la), target),
        bounds=(0.3, 4.0), method='bounded'
    )
    best_la = result.x

    return best_lh, best_la, best_error

def simulate_match(lambda_h, lambda_a, n_sims=10000, max_goals=8):
    """Run Monte Carlo simulation using DC-adjusted Poisson"""
    # Pre-compute DC probabilities for all score combinations
    score_probs = {}
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            score_probs[(hg, ag)] = dixon_coles_prob(hg, ag, lambda_h, lambda_a)

    # Normalize
    total = sum(score_probs.values())
    score_probs = {k: v/total for k, v in score_probs.items()}

    scores = list(score_probs.keys())
    probs = list(score_probs.values())

    # Sample
    samples = np.random.choice(len(scores), size=n_sims, p=probs)
    home_goals = np.array([scores[s][0] for s in samples])
    away_goals = np.array([scores[s][1] for s in samples])

    # Compute derived
    total_goals = home_goals + away_goals
    results = np.where(home_goals > away_goals, 'H', np.where(home_goals == away_goals, 'D', 'A'))
    over25 = (total_goals > 2.5).astype(int)
    btts = ((home_goals > 0) & (away_goals > 0)).astype(int)

    return {
        'home_goals': home_goals,
        'away_goals': away_goals,
        'total_goals': total_goals,
        'results': results,
        'over25': over25,
        'btts': btts,
        'score_probs': score_probs
    }

def estimate_odds_from_probs(prob, margin=0.05):
    """Estimate fair odds from probability with margin"""
    return (1 / prob) * (1 + margin)

def run_full_simulation(fixture_data, n_sims=10000):
    """Run complete Monte Carlo analysis for a fixture"""
    # Extract ML probabilities
    ml_probs = {
        'pred_result_H': fixture_data['pred_result_H'],
        'pred_result_D': fixture_data['pred_result_D'],
        'pred_result_A': fixture_data['pred_result_A'],
        'pred_over_25_1': fixture_data['pred_over_25_1'],
        'pred_btts_1': fixture_data['pred_btts_1']
    }

    # Estimate lambdas
    lambda_h, lambda_a, fit_error = estimate_lambdas(ml_probs)

    # DC probabilities at estimated lambdas
    dc_probs = match_probs_from_lambdas(lambda_h, lambda_a)

    # Run simulation
    sim = simulate_match(lambda_h, lambda_a, n_sims)

    # Empirical probabilities from simulation
    emp_H = (sim['results'] == 'H').mean()
    emp_D = (sim['results'] == 'D').mean()
    emp_A = (sim['results'] == 'A').mean()
    emp_O25 = sim['over25'].mean()
    emp_U25 = 1 - emp_O25
    emp_BTTS_Y = sim['btts'].mean()
    emp_BTTS_N = 1 - emp_BTTS_Y

    # Top scorelines
    score_counts = pd.Series(list(zip(sim['home_goals'], sim['away_goals']))).value_counts()
    top_scorelines = score_counts.head(10)

    # Market odds (from fixture data)
    odds = {
        'home': fixture_data.get('home'),
        'draw': fixture_data.get('draw'),
        'away': fixture_data.get('away'),
        'over_2_5': fixture_data.get('over_2_5'),
        'under_2_5': fixture_data.get('under_2_5'),
        'btts_yes': fixture_data.get('btts_yes'),
        'btts_no': fixture_data.get('btts_no')
    }

    # If Over/Under odds missing, estimate from model probs
    if odds['over_2_5'] is None or pd.isna(odds['over_2_5']):
        fair_o25 = 1 / emp_O25
        fair_u25 = 1 / emp_U25
        # Add typical margin
        margin = 0.05
        odds['over_2_5'] = fair_o25 * (1 + margin)
        odds['under_2_5'] = fair_u25 * (1 + margin)

    if odds['btts_yes'] is None or pd.isna(odds['btts_yes']):
        fair_btts_y = 1 / emp_BTTS_Y
        fair_btts_n = 1 / emp_BTTS_N
        margin = 0.05
        odds['btts_yes'] = fair_btts_y * (1 + margin)
        odds['btts_no'] = fair_btts_n * (1 + margin)

    # Compute EVs for all markets
    ev_results = []
    markets = {
        '1X2_H': ('H', emp_H, odds.get('home')),
        '1X2_D': ('D', emp_D, odds.get('draw')),
        '1X2_A': ('A', emp_A, odds.get('away')),
        'OVER_2.5': ('Over 2.5', emp_O25, odds.get('over_2_5')),
        'UNDER_2.5': ('Under 2.5', emp_U25, odds.get('under_2_5')),
        'BTTS_YES': ('BTTS Yes', emp_BTTS_Y, odds.get('btts_yes')),
        'BTTS_NO': ('BTTS No', emp_BTTS_N, odds.get('btts_no'))
    }

    for mkt_name, (label, p_model, odd) in markets.items():
        if odd is None or pd.isna(odd) or odd < 1.1 or odd > 10:
            continue
        ev = p_model * odd - 1
        kelly = (p_model * odd - 1) / (odd - 1) if odd > 1 else 0
        stake = max(0, kelly * 0.25)  # 25% Kelly
        clv = (p_model - 1/odd) * 100
        ev_results.append({
            'Market': mkt_name,
            'Label': label,
            'Sim_Prob': round(p_model, 4),
            'Odds': round(odd, 2),
            'EV%': round(ev * 100, 2),
            'Kelly%': round(kelly * 100, 2),
            'Stake%': round(stake * 100, 2),
            'CLV%': round(clv, 2)
        })

    return {
        'lambda_home': round(lambda_h, 3),
        'lambda_away': round(lambda_a, 3),
        'fit_error': round(fit_error, 6),
        'dc_probs': {k: round(v, 4) for k, v in dc_probs.items()},
        'emp_probs': {
            'H': round(emp_H, 4), 'D': round(emp_D, 4), 'A': round(emp_A, 4),
            'Over2.5': round(emp_O25, 4), 'Under2.5': round(emp_U25, 4),
            'BTTS_Y': round(emp_BTTS_Y, 4), 'BTTS_N': round(emp_BTTS_N, 4)
        },
        'top_scorelines': top_scorelines,
        'ev_results': pd.DataFrame(ev_results),
        'odds_used': odds
    }

if __name__ == "__main__":
    # Load live analysis data
    df = pd.read_parquet('data/live_analysis_2026-11-23.parquet')
    # Filter to known Top 5 fixtures only (scraper misclassifies other leagues)
    top5_teams = {
        'Brentford', 'Everton',      # E0 Premier League
        'Levante', 'Elche',          # SP1 La Liga
        'Monza', 'Fiorentina'        # I1 Serie A
    }
    mask = df['HomeTeam'].isin(top5_teams) | df['AwayTeam'].isin(top5_teams)
    top5 = df[mask].copy()

    print("=" * 80)
    print("MONTE CARLO SIMULATION - 2026-11-23 TOP 5 FIXTURES")
    print("=" * 80)

    all_ev_bets = []

    for _, row in top5.iterrows():
        fixture = f"{row['HomeTeam']} vs {row['AwayTeam']} ({row['Div']})"
        print(f"\n{'='*80}")
        print(f"FIXTURE: {fixture}")
        print(f"Date: {row['Date']}")
        print(f"{'='*80}")

        result = run_full_simulation(row, n_sims=10000)

        print(f"\nEstimated Lambdas: lambda_home={result['lambda_home']}, lambda_away={result['lambda_away']}")
        print(f"Lambda Fit Error: {result['fit_error']:.6f}")

        print(f"\n--- DC Model Probabilities ---")
        for k, v in result['dc_probs'].items():
            print(f"  {k}: {v:.2%}")

        print(f"\n--- Empirical Probabilities (10k sims) ---")
        for k, v in result['emp_probs'].items():
            print(f"  {k}: {v:.2%}")

        print(f"\n--- Top 10 Scorelines ---")
        for (hg, ag), count in result['top_scorelines'].items():
            pct = count / 10000 * 100
            print(f"  {hg}-{ag}: {pct:.2f}%")

        print(f"\n--- EV+ Betting Opportunities ---")
        ev_df = result['ev_results']
        ev_plus = ev_df[ev_df['EV%'] > 5].sort_values('EV%', ascending=False)
        if len(ev_plus) > 0:
            print(ev_df[['Market', 'Sim_Prob', 'Odds', 'EV%', 'Stake%', 'CLV%']].to_string(index=False))
            # Add fixture info
            ev_plus['Fixture'] = fixture
            ev_plus['Date'] = row['Date']
            all_ev_bets.append(ev_plus)
        else:
            print("  No EV+ bets found (threshold > 5%)")

    # Summary: Top 3 singles + 1 combo (Top 5 leagues only - already filtered)
    if all_ev_bets:
        all_bets = pd.concat(all_ev_bets, ignore_index=True)
        all_bets = all_bets[all_bets['EV%'] > 5].sort_values('EV%', ascending=False)

        print(f"\n{'='*80}")
        print("FINAL RECOMMENDATIONS - 3 SINGLES + 1 COMBO")
        print(f"{'='*80}")

        # Top 3 singles by EV
        top3 = all_bets.head(3)
        print("\n[TARGET] TOP 3 SINGLE BETS:")
        for i, (_, bet) in enumerate(top3.iterrows(), 1):
            print(f"  {i}. {bet['Fixture']} | {bet['Label']} @ {bet['Odds']} | "
                  f"Prob: {bet['Sim_Prob']:.1%} | EV: {bet['EV%']:.1f}% | "
                  f"Stake: {bet['Stake%']:.2f}% | CLV: {bet['CLV%']:.2f}%")

        # Best combo: 2 highest EV singles from different matches
        print("\n[TARGET] RECOMMENDED COMBO (2-leg):")
        combo_legs = []
        for _, bet in top3.iterrows():
            if len(combo_legs) < 2:
                combo_legs.append(bet)

        if len(combo_legs) == 2:
            combo_odds = combo_legs[0]['Odds'] * combo_legs[1]['Odds']
            combo_prob = combo_legs[0]['Sim_Prob'] * combo_legs[1]['Sim_Prob']
            combo_ev = combo_prob * combo_odds - 1
            # Kelly for combo (more conservative)
            combo_kelly = (combo_prob * combo_odds - 1) / (combo_odds - 1) if combo_odds > 1 else 0
            combo_stake = max(0, combo_kelly * 0.15)  # 15% Kelly for combo
            combo_clv = (combo_prob - 1/combo_odds) * 100

            print(f"  Leg 1: {combo_legs[0]['Fixture']} | {combo_legs[0]['Label']} @ {combo_legs[0]['Odds']}")
            print(f"  Leg 2: {combo_legs[1]['Fixture']} | {combo_legs[1]['Label']} @ {combo_legs[1]['Odds']}")
            print(f"  Combined Odds: {combo_odds:.2f} | Prob: {combo_prob:.2%} | EV: {combo_ev*100:.1f}%")
            print(f"  Recommended Stake: {combo_stake*100:.2f}% | CLV: {combo_clv:.2f}%")

        # Total bankroll allocation
        total_stake = top3['Stake%'].sum() + (combo_stake * 100 if 'combo_stake' in locals() else 0)
        print(f"\n[MONEY] TOTAL BANKROLL ALLOCATION: {total_stake:.2f}%")
        print(f"   (Singles: {top3['Stake%'].sum():.2f}% + Combo: {combo_stake*100:.2f}%)")