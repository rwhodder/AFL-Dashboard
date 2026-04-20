"""
count_qualifying_legs.py
────────────────────────
Shows how many tackle legs pass all hard filters per round.
Run this to understand typical weekly leg volume.

Usage:
    python count_qualifying_legs.py
"""

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from app import _passes_bet_filters, GEE_HOME_ROUNDS, TACKLE_BAD_OPPONENTS

GOOGLE_SHEET_ID         = "10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"
GOOGLE_SHEET_TAB        = "Bet Log"


def main():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds  = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_FILE, scopes=scopes)
    client = gspread.authorize(creds)
    ws     = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_TAB)

    print("Reading sheet ...")
    records = ws.get_all_records()
    df = pd.DataFrame(records)

    # Tackle rows only
    df = df[df['Type'].astype(str).str.strip() == 'Tackle'].copy()
    df['Year']  = pd.to_numeric(df['Year'],  errors='coerce')
    df['Round'] = pd.to_numeric(df['Round'], errors='coerce')
    df['wl']    = df['W/L'].astype(str).str.strip()

    df['resolved']   = df['wl'].isin(['1','-1','0','1.0','-1.0','0.0'])
    df['qualifying'] = df.apply(_passes_bet_filters, axis=1)

    print(f"\nTotal tackle rows:      {len(df)}")
    print(f"Qualifying (all):       {df['qualifying'].sum()}")
    print(f"Qualifying + resolved:  {(df['qualifying'] & df['resolved']).sum()}")

    print("\n── Per round breakdown ──────────────────────────────────────────────")
    print(f"  {'Year':>6}  {'Round':>5}  {'All legs':>9}  {'Qualifying':>10}  {'Resolved':>8}  {'Q+Resolved':>10}")
    print(f"  {'':─>6}  {'':─>5}  {'':─>9}  {'':─>10}  {'':─>8}  {'':─>10}")

    for (yr, rnd), grp in df.groupby(['Year', 'Round']):
        total   = len(grp)
        qual    = grp['qualifying'].sum()
        res     = grp['resolved'].sum()
        qual_res = (grp['qualifying'] & grp['resolved']).sum()
        print(f"  {int(yr):>6}  R{int(rnd):>4}  {total:>9}  {qual:>10}  {res:>8}  {qual_res:>10}")

    print("\n── Summary ──────────────────────────────────────────────────────────")
    summary = df.groupby('Year').apply(lambda g: pd.Series({
        'rounds':    g['Round'].nunique(),
        'qual_total': g['qualifying'].sum(),
        'qual_res':  (g['qualifying'] & g['resolved']).sum(),
    })).reset_index()

    for _, row in summary.iterrows():
        rpr = row['qual_total'] / row['rounds'] if row['rounds'] > 0 else 0
        print(f"  {int(row['Year'])}: {int(row['rounds'])} rounds  "
              f"{int(row['qual_total'])} qualifying legs  "
              f"({rpr:.1f}/round)  "
              f"{int(row['qual_res'])} resolved")

    # Filter reason breakdown for most recent round
    most_recent = df[df['Year'] == df['Year'].max()]
    most_recent = most_recent[most_recent['Round'] == most_recent['Round'].max()]
    print(f"\n── Filter breakdown for most recent round "
          f"(R{int(most_recent['Round'].iloc[0])} {int(most_recent['Year'].iloc[0])}) ──")

    def _fail_reason(row):
        import math
        def get(k):
            try: return str(row.get(k, '')).strip()
            except: return ''
        try:    line_val = float(get('Line'))
        except: return 'no line'
        if line_val < 3:                         return 'line < 3'
        if 'Short Break' in get('Travel Fatigue'): return 'short break'
        if get('Opponent') in TACKLE_BAD_OPPONENTS: return 'bad opponent (GCS)'
        pos = get('Position')
        dvp = get('DvP')
        if pos == 'MedF':                        return 'MedF position'
        if pos == 'FwdMid' and not any(x in dvp for x in ('Moderate Unders','Strong Unders')):
            return 'FwdMid no unders DvP'
        try:
            yr_rnd = (int(get('Year')), int(get('Round')))
            opp, team = get('Opponent'), get('Team')
            if yr_rnd in GEE_HOME_ROUNDS and (opp == 'GEE' or team == 'GEE'):
                return 'GEE home game'
        except: pass
        if pos not in ('Wing','Ruck','GenD'):
            try:
                avl = float(get('Avg vs Line').replace('%','').replace('+',''))
                if avl >= 10.0: return 'AvL >= 10%'
            except: pass
            try:
                cons = float(get('Line Consistency').replace('%',''))
                if 61 <= cons <= 80 and 'Moderate Unders' in dvp:
                    return 'cons 61-80 + Mod Unders'
            except: pass
        return 'PASS'

    most_recent = most_recent.copy()
    most_recent['reason'] = most_recent.apply(_fail_reason, axis=1)
    reason_counts = most_recent['reason'].value_counts()
    for reason, count in reason_counts.items():
        marker = '✓' if reason == 'PASS' else '✗'
        print(f"  {marker}  {reason:<30} {count}")


if __name__ == "__main__":
    main()
