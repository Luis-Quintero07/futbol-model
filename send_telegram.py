#!/usr/bin/env python
"""Send Telegram summary for GitHub Actions workflow"""
import os
import pandas as pd
from pathlib import Path
import sys

def send_summary(pass_type):
    DATA = Path('data')
    files = sorted(DATA.glob(f'live_bets_*_{pass_type}.parquet'))
    if not files:
        print(f"No bets file found for {pass_type}")
        return

    df = pd.read_parquet(files[-1])
    if len(df) == 0:
        print(f"No bets for {pass_type}")
        return

    if pass_type == 'pre':
        header = f'PRE-LINEUP {df.iloc[0]["Date"]}\n{len(df)} EV+ bets | Avg EV: {df["EV%"].mean():.1f}% | Stake: {df["Stake%"].sum():.1f}%'
    else:
        header = f'POST-LINEUP {df.iloc[0]["Date"]}\n{len(df)} EV+ bets | Avg EV: {df["EV%"].mean():.1f}% | Stake: {df["Stake%"].sum():.1f}% | CLV: {df["CLV%"].mean():.1f}%'

    msg = header + '\n'
    for _, r in df.iterrows():
        msg += f'{r["League"]} {r["Home"]} vs {r["Away"]} | {r["Market"]} @{r["Odds"]:.2f} EV{r["EV%"]:.1f}% Stake{r["Stake%"]:.1f}%\n'

    import requests
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    response = requests.post(
        f'https://api.telegram.org/bot{token}/sendMessage',
        json={'chat_id': chat_id, 'text': msg}
    )
    print(f"Telegram response: {response.status_code}")

if __name__ == '__main__':
    pass_type = sys.argv[1] if len(sys.argv) > 1 else 'pre'
    send_summary(pass_type)