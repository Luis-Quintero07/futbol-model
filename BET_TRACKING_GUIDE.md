# Bet Tracking Template - Football Betting System

## 📋 How to Use

### 1. **Before Match (Pre-Match)**
Fill these columns immediately after placing bets:
- `Date` - Match date (YYYY-MM-DD)
- `League` - League code (E0, SP1, D1, I1, F1)
- `Fixture` - "Home vs Away"
- `Market` - Market type (1X2, Over/Under 2.5, BTTS, Combo)
- `Selection` - Your pick (Home/Draw/Away, Over/Under, Yes/No)
- `Bookmaker` - Where you placed the bet (Bet365, etc.)
- `Odds` - **Odds at time of bet placement** (critical for CLV)
- `Model_Prob` - Model's estimated probability
- `EV_Pct` - Expected Value % (Model_Prob × Odds - 1)
- `Kelly_Stake_Pct` - Recommended stake (25% Kelly fraction)
- `Stake_Units` - Actual units risked (e.g., 1 unit = 1% bankroll)

### 2. **After Match (Post-Match)**
Fill these columns after result is known:
- `Result` - W (Win), L (Loss), P (Push/Void)
- `PL_Units` - Profit/Loss in units (Odds - 1 if W, -1 if L, 0 if P)
- `Closing_Odds` - **Closing odds at kickoff** (critical for CLV validation)
- `CLV_Pct` - Closing Line Value: (Model_Prob - 1/Closing_Odds) × 100
- `Notes` - Any relevant context (injuries, lineup changes, etc.)

---

## 🎯 Today's Bets (2026-11-23)

| Date | League | Fixture | Market | Selection | Bookmaker | Odds | Model_Prob | EV% | Kelly% | Stake | Result | PL | Close_Odds | CLV% | Notes |
|------|--------|---------|--------|-----------|-----------|------|------------|-----|--------|-------|--------|-----|------------|------|-------|
| 2026-11-23 | I1 | Monza vs Fiorentina | 1X2 | Away (Fiorentina) | Bet365 | 1.98 | 65.8% | 30.4% | 7.75% | | | | | | Monte Carlo + ML |
| 2026-11-23 | SP1 | Levante vs Elche | 1X2 | Home (Levante) | Bet365 | 2.20 | 55.9% | 23.0% | 4.79% | | | | | | Monte Carlo + ML |
| 2026-11-23 | E0 | Brentford vs Everton | 1X2 | Away (Everton) | Bet365 | 3.10 | 36.9% | 14.4% | 1.72% | | | | | | Monte Carlo + ML |
| 2026-11-23 | COMBO | Monza(A) + Levante(H) | Combo | 2-leg | Bet365 | 4.36 | 36.8% | 60.3% | 2.70% | | | | | | 2-leg combo |

---

## 📊 Key Metrics to Track

### **Primary: CLV (Closing Line Value)**
- **Formula**: `(Model_Prob - 1/Closing_Odds) × 100`
- **Target**: Positive CLV on >55% of bets = proven edge
- **Why it matters**: Beating the closing line is the only sustainable proof of skill

### **Secondary: ROI by Market/League**
- Track ROI separately for: 1X2, Over/Under, BTTS, Combos
- Track by league: E0, SP1, D1, I1, F1
- Identify where edge is strongest/weakest

### **Tertiary: Calibration**
- Bin predictions by probability (e.g., 50-55%, 55-60%, etc.)
- Actual win rate should match predicted probability
- Recalibrate if systematic bias detected

---

## 📈 Weekly Review Checklist

- [ ] All bets have closing odds recorded
- [ ] CLV calculated for every bet
- [ ] P&L updated
- [ ] ROI by market/league reviewed
- [ ] Calibration plot checked (monthly)
- [ ] Model drift detected? (PSI > 0.2)
- [ ] Bankroll adjusted for Kelly stakes

---

## 🔧 Google Sheets Setup

1. **Import** `bet_tracking_template.csv` into Google Sheets
2. **Freeze** row 1 (headers)
3. **Conditional formatting**:
   - `CLV_Pct` > 0 → Green
   - `CLV_Pct` < 0 → Red
   - `Result` = "W" → Green, "L" → Red
4. **Pivot tables** for:
   - ROI by Market
   - ROI by League
   - CLV distribution
   - Monthly P&L
5. **Charts**:
   - Cumulative P&L over time
   - CLV histogram
   - Calibration curve (monthly)

---

## ⚠️ Critical Rules

1. **NEVER** change `Odds` after bet placement - that's your entry price
2. **ALWAYS** record `Closing_Odds` at kickoff (check oddsportal/flashscore)
3. **Combo bets**: Track each leg separately + combo as single row
4. **Void bets**: Mark as "P" (push), PL = 0, still track CLV
5. **Bankroll**: Recalculate Kelly stakes weekly based on current bankroll

---

## 📁 Files

- `bet_tracking_template.csv` - Raw CSV for import
- `BET_TRACKING_GUIDE.md` - This guide
- `live_bets_YYYY-MM-DD.parquet` - Daily model output (auto-generated)
- `live_analysis_YYYY-MM-DD.parquet` - Full fixture analysis (auto-generated)