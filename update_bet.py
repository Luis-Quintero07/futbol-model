#!/usr/bin/env python
"""
Bet Tracker - Update results and track performance
Usage:
  python update_bet.py --date 2026-11-27 --match "Venezia vs Bologna" --market 1X2_A --result 1
  python update_bet.py --date 2026-11-27 --match "Venezia vs Bologna" --market 1X2_A --result 0
  python update_bet.py --list  # Show all pending bets
  python update_bet.py --summary  # Show performance summary
"""
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
TRACKER_FILE = DATA_DIR / "bet_tracker.csv"

# ─── Load/Create Tracker ───
def load_tracker():
    if TRACKER_FILE.exists():
        return pd.read_csv(TRACKER_FILE)
    return pd.DataFrame(columns=[
        'Date', 'League', 'Home', 'Away', 'Market', 'Odds', 'Model_Prob',
        'EV%', 'Stake%', 'Result', 'PnL', 'CLV%', 'Notes', 'Timestamp'
    ])

def save_tracker(df):
    df.to_csv(TRACKER_FILE, index=False)
    print(f"✅ Saved {len(df)} bets to {TRACKER_FILE}")

# ─── Commands ───
def cmd_list(args):
    df = load_tracker()
    if df.empty:
        print("No bets tracked yet.")
        return

    pending = df[df['Result'].isna()]
    settled = df[df['Result'].notna()]

    print(f"\n📋 PENDING ({len(pending)}):")
    if len(pending) > 0:
        for _, r in pending.iterrows():
            print(f"  {r['Date']} | {r['League']} | {r['Home']} vs {r['Away']} | {r['Market']} @{r['Odds']:.2f} | EV {r['EV%']:.1f}% | Stake {r['Stake%']:.2f}%")

    print(f"\n✅ SETTLED ({len(settled)}):")
    if len(settled) > 0:
        for _, r in settled.iterrows():
            status = "🟢 WIN" if r['Result'] == 1 else "🔴 LOSS"
            print(f"  {r['Date']} | {r['League']} | {r['Home']} vs {r['Away']} | {r['Market']} | {status} | PnL {r['PnL']:+.2f}")

def cmd_update(args):
    df = load_tracker()

    # Find matching bet (pending)
    mask = (
        (df['Date'] == args.date) &
        (df['Home'].str.contains(args.match.split(' vs ')[0], case=False)) &
        (df['Away'].str.contains(args.match.split(' vs ')[1], case=False)) &
        (df['Market'] == args.market) &
        (df['Result'].isna())
    )

    matches = df[mask]
    if len(matches) == 0:
        print("❌ No matching pending bet found.")
        print("Available pending bets:")
        pending = df[df['Result'].isna()]
        for _, r in pending.iterrows():
            print(f"  {r['Date']} | {r['Home']} vs {r['Away']} | {r['Market']}")
        return

    if len(matches) > 1:
        print("❌ Multiple matches found. Be more specific.")
        return

    idx = matches.index[0]
    result = args.result  # 1 = win, 0 = loss
    odds = df.loc[idx, 'Odds']
    stake = df.loc[idx, 'Stake%'] / 100  # as fraction of bankroll

    # Calculate PnL
    if result == 1:
        pnl = stake * (odds - 1)
    else:
        pnl = -stake

    # Update
    df.loc[idx, 'Result'] = result
    df.loc[idx, 'PnL'] = round(pnl, 4)
    df.loc[idx, 'Timestamp'] = datetime.now().isoformat()
    if args.notes:
        df.loc[idx, 'Notes'] = args.notes

    save_tracker(df)

    status = "🟢 WIN" if result == 1 else "🔴 LOSS"
    print(f"✅ Updated: {args.match} | {args.market} | {status} | PnL: {pnl:+.4f} units")

def cmd_summary(args):
    df = load_tracker()
    settled = df[df['Result'].notna()]

    if settled.empty:
        print("No settled bets yet.")
        return

    total = len(settled)
    wins = (settled['Result'] == 1).sum()
    wr = wins / total * 100
    total_pnl = settled['PnL'].sum()
    roi = total_pnl / total * 100
    avg_ev = settled['EV%'].mean()
    avg_clv = settled['CLV%'].mean() if 'CLV%' in settled.columns else 0

    print(f"\n📊 PERFORMANCE SUMMARY")
    print(f"  Total Bets: {total}")
    print(f"  Wins: {wins} | Losses: {total - wins}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Total PnL: {total_pnl:+.2f} units")
    print(f"  ROI: {roi:.2f}%")
    print(f"  Avg EV (predicted): {avg_ev:.1f}%")
    print(f"  Avg CLV: {avg_clv:.1f}%")

    # By market
    print(f"\n📈 BY MARKET:")
    for mkt in settled['Market'].unique():
        d = settled[settled['Market'] == mkt]
        wr_m = (d['Result'] == 1).mean() * 100
        roi_m = d['PnL'].sum() / len(d) * 100
        print(f"  {mkt}: {len(d)} bets | WR {wr_m:.1f}% | ROI {roi_m:.2f}%")

    # By league
    print(f"\n🏆 BY LEAGUE:")
    for lg in settled['League'].unique():
        d = settled[settled['League'] == lg]
        wr_m = (d['Result'] == 1).mean() * 100
        roi_m = d['PnL'].sum() / len(d) * 100
        print(f"  {lg}: {len(d)} bets | WR {wr_m:.1f}% | ROI {roi_m:.2f}%")

def cmd_import(args):
    """Import bets from live_bets parquet file"""
    df = load_tracker()

    bet_files = sorted(DATA_DIR.glob(f"live_bets_{args.date}*_post.parquet"))
    if not bet_files:
        bet_files = sorted(DATA_DIR.glob(f"live_bets_{args.date}*_pre.parquet"))

    if not bet_files:
        print(f"❌ No bets file found for {args.date}")
        return

    bets_df = pd.read_parquet(bet_files[-1])
    print(f"Found {len(bets_df)} bets in {bet_files[-1].name}")

    new_count = 0
    for _, bet in bets_df.iterrows():
        # Check if already exists
        exists = (
            (df['Date'] == bet['Date']) &
            (df['Home'] == bet['Home']) &
            (df['Away'] == bet['Away']) &
            (df['Market'] == bet['Market'])
        ).any()

        if not exists:
            new_row = {
                'Date': bet['Date'],
                'League': bet['League'],
                'Home': bet['Home'],
                'Away': bet['Away'],
                'Market': bet['Market'],
                'Odds': bet['Odds'],
                'Model_Prob': bet.get('Model_Prob', bet.get('MC_Prob', 0)),
                'EV%': bet['EV%'],
                'Stake%': bet['Stake%'],
                'Result': None,
                'PnL': None,
                'CLV%': bet.get('CLV%', 0),
                'Notes': f"Auto-imported from {bet_files[-1].name}",
                'Timestamp': datetime.now().isoformat()
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            new_count += 1

    save_tracker(df)
    print(f"✅ Imported {new_count} new bets (total: {len(df)})")

# ─── Main ───
def main():
    parser = argparse.ArgumentParser(description="Bet Tracker")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # list
    subparsers.add_parser('list', help='List all bets')

    # update
    p_update = subparsers.add_parser('update', help='Update bet result')
    p_update.add_argument('--date', required=True, help='Match date (YYYY-MM-DD)')
    p_update.add_argument('--match', required=True, help='Match (e.g., "Venezia vs Bologna")')
    p_update.add_argument('--market', required=True, help='Market (1X2_H, 1X2_D, 1X2_A, OVER_2.5, UNDER_2.5, BTTS_YES, BTTS_NO)')
    p_update.add_argument('--result', required=True, type=int, choices=[0, 1], help='Result: 1=win, 0=loss')
    p_update.add_argument('--notes', help='Optional notes')

    # summary
    subparsers.add_parser('summary', help='Show performance summary')

    # import
    p_import = subparsers.add_parser('import', help='Import bets from live_bets file')
    p_import.add_argument('--date', required=True, help='Match date (YYYY-MM-DD)')

    args = parser.parse_args()

    if args.command == 'list':
        cmd_list(args)
    elif args.command == 'update':
        cmd_update(args)
    elif args.command == 'summary':
        cmd_summary(args)
    elif args.command == 'import':
        cmd_import(args)

if __name__ == "__main__":
    main()