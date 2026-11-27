# Daily Betting Report - 2026-11-23

## Executive Summary
**3 Singles + 1 Combo** | **Total Bankroll Allocation: 13.42%**

| # | Fixture | League | Selection | Odds | Model Prob | EV% | Stake% | CLV% |
|---|---------|--------|-----------|------|------------|-----|--------|------|
| 1 | Levante vs Elche | SP1 | **Home (Levante)** | 2.18 | 55.9% | **21.9%** | **4.64%** | 10.04% |
| 2 | Monza vs Fiorentina | I1 | **Away (Fiorentina)** | 1.98 | 61.3% | **21.4%** | **5.47%** | 10.82% |
| 3 | Brentford vs Everton | E0 | **Away (Everton)** | 3.10 | 35.4% | **9.6%** | **1.14%** | 3.09% |
| **Combo** | Levante (H) + Fiorentina (A) | SP1+I1 | **2-leg** | 4.32 | 34.3% | **48.0%** | **2.17%** | 11.12% |

---

## Detailed Fixture Analysis

### 1. Levante vs Elche (SP1) - 15:00
**Recommendation: LEVANTE HOME WIN @ 2.18**

| Metric | Value |
|--------|-------|
| Model Probability | 55.9% |
| Market Implied Prob | 45.9% |
| **Edge** | **+10.0%** |
| Kelly Stake (25%) | 4.64% |
| Monte Carlo Sim Prob | 55.91% |
| DC Model Prob | 56.19% |

**Key Stats:**
- Levante home attack: Strong (λ_home = 1.91)
- Elche away defense: Weak (λ_away = 1.16)
- Most likely scorelines: 2-1 (9.8%), 1-0 (9.7%), 1-1 (9.3%)
- Over 2.5: 59.2% | BTTS: 57.0%

**Additional EV+ Markets:**
- Over 2.5 @ 1.77 (EV: 5.0%, Stake: 1.61%)
- BTTS Yes @ 1.84 (EV: 5.0%, Stake: 1.48%)

---

### 2. Monza vs Fiorentina (I1) - 14:45
**Recommendation: FIORENTINA AWAY WIN @ 1.98**

| Metric | Value |
|--------|-------|
| Model Probability | 61.3% |
| Market Implied Prob | 50.5% |
| **Edge** | **+10.8%** |
| Kelly Stake (25%) | 5.47% |
| Monte Carlo Sim Prob | 61.33% |
| DC Model Prob | 60.97% |

**Key Stats:**
- Fiorentina away attack: Very strong (λ_away = 2.05)
- Monza home defense: Weak (λ_home = 1.08)
- Most likely scorelines: 0-1 (10.0%), 1-2 (9.8%), 0-2 (9.4%)
- Over 2.5: 60.3% | BTTS: 55.9%

**Additional EV+ Markets:**
- Over 2.5 @ 1.74 (EV: 5.0%, Stake: 1.69%)
- BTTS Yes @ 1.88 (EV: 5.0%, Stake: 1.42%)

---

### 3. Brentford vs Everton (E0) - 15:00
**Recommendation: EVERTON AWAY WIN @ 3.10**

| Metric | Value |
|--------|-------|
| Model Probability | 35.4% |
| Market Implied Prob | 32.3% |
| **Edge** | **+3.1%** |
| Kelly Stake (25%) | 1.14% |
| Monte Carlo Sim Prob | 35.35% |
| DC Model Prob | 35.28% |

**Key Stats:**
- Balanced match: λ_home = 1.46, λ_away = 1.30
- Most likely scorelines: 1-0 (11.1%), 1-1 (10.5%), 0-1 (10.1%)
- Over 2.5: 52.3% | BTTS: 54.6%

**Additional EV+ Markets:**
- Over 2.5 @ 2.01 (EV: 5.0%, Stake: 1.24%)
- BTTS Yes @ 1.92 (EV: 5.0%, Stake: 1.35%)

---

## Combo Bet Analysis
**Levante (H) + Fiorentina (A) @ 4.32**

| Metric | Value |
|--------|-------|
| Combined Probability | 34.29% (55.9% × 61.3%) |
| Combined Odds | 4.32 (2.18 × 1.98) |
| **EV** | **48.0%** |
| Kelly Stake (25%) | 2.17% |
| CLV | 11.12% |

**Risk Assessment:** Both legs are highest-conviction singles with strong edges. Correlation risk is low (different leagues, different kickoff times).

---

## Bankroll Management

| Bet | Stake % | Cumulative |
|-----|---------|------------|
| Levante H | 4.64% | 4.64% |
| Fiorentina A | 5.47% | 10.11% |
| Everton A | 1.14% | 11.25% |
| **Combo** | **2.17%** | **13.42%** |

**Kelly Criterion:** 25% fraction used for all bets
**Max Single Stake:** 5.47% (Fiorentina)
**Total Exposure:** 13.42% of bankroll

---

## Validation Checklist (Pre-Match)

- [ ] Odds verified at Bet365 (2.18, 1.98, 3.10)
- [ ] Lineups checked 1 hour before kickoff
- [ ] No major injury/suspension news
- [ ] Closing odds recorded at kickoff for CLV tracking
- [ ] Stakes calculated based on current bankroll

---

## Post-Match Tracking (Fill After Matches)

| Fixture | Result | PL Units | Closing Odds | CLV% | Notes |
|---------|--------|----------|--------------|------|-------|
| Levante vs Elche | | | | | |
| Monza vs Fiorentina | | | | | |
| Brentford vs Everton | | | | | |
| Combo | | | | | |

---

## Model Confidence Indicators

| Fixture | ML Model | Poisson/DC | Agreement | Confidence |
|---------|----------|------------|-----------|------------|
| Levante vs Elche | 53.8% (H) | 55.9% (H) | ✅ Strong | HIGH |
| Monza vs Fiorentina | 58.1% (A) | 61.3% (A) | ✅ Strong | HIGH |
| Brentford vs Everton | 36.9% (A) | 35.4% (A) | ✅ Moderate | MEDIUM |

**Note:** ML and Poisson models agree on direction for all 3 picks. Highest confidence on Levante and Fiorentina.

---

## Files Generated
- `bet_tracking_2026-11-23.csv` - Import into Google Sheets
- `live_analysis_2026-11-23.parquet` - Full feature + prediction data
- `live_bets_2026-11-23.parquet` - EV+ bets only
- `oddsportal_fixtures_20261123.parquet` - Raw scraped fixtures