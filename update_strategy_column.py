"""
update_strategy_column.py
─────────────────────────
Applies the current T1/T2 flagging logic to every row in the Bet Log
and writes the result to the 'Strategy' column (col N).

Logic (mirrors calculate_bet_flag in app.py):
  Only Tackle rows can be T1 or T2.  Disposal/Mark rows are always blank.

  AVOID (blank) if any:
    - Line < 3
    - Avg vs Line >= 10%
    - Short Break in Travel Fatigue
    - Opponent in {SYD, GCS, GEE}
    - Position = FwdMid AND DvP contains any Easy

  T1 if:
    - Position in {Wing, Ruck}

  T2 if:
    - Passes all avoid filters (not Wing/Ruck)

  Note: Stadium (narrow ground) filter cannot be applied historically
  as the sheet has no Stadium column.

Usage:
    python update_strategy_column.py
"""

import csv
import gspread
from google.oauth2.service_account import Credentials

GOOGLE_SHEET_ID         = "10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"
GOOGLE_SHEET_TAB        = "Bet Log"
FIXTURE_FILE            = "afl-2026-fixture.csv"

TACKLE_BAD_OPPONENTS = {'GCS'}  # GEE handled by home-ground round lookup below; SYD removed (coincidental overlap)


def load_gee_home_rounds(fixture_file=FIXTURE_FILE):
    """
    Parse the fixture CSV and return a set of (year, round) tuples where
    Geelong are at home at GMHBA Stadium / Kardinia Park.
    The Stadium column is not in the Bet Log, so we use the fixture instead.
    """
    gee_home = set()
    try:
        with open(fixture_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rnd_str  = row.get('Round Number', '').strip()
                home     = row.get('Home Team', '').strip()
                location = row.get('Location', '').strip()
                if 'Geelong' in home and ('GMHBA' in location or 'Kardinia' in location):
                    try:
                        gee_home.add((2026, int(rnd_str)))
                    except ValueError:
                        pass  # skip OR / non-numeric rounds
    except FileNotFoundError:
        print(f"WARN: fixture file '{fixture_file}' not found — GEE home-round filter inactive")
    return gee_home


GEE_HOME_ROUNDS = load_gee_home_rounds()


def classify(row, header):
    def get(col):
        try:
            return str(row[header.index(col)]).strip()
        except (ValueError, IndexError):
            return ''

    stat_type = get('Type')
    if stat_type != 'Tackle':
        return ''

    position = get('Position')
    dvp      = get('DvP')
    travel   = get('Travel Fatigue')
    opponent = get('Opponent')
    team     = get('Team')
    line_str = get('Line')
    avl_str  = get('Avg vs Line')
    year_str = get('Year')
    rnd_str  = get('Round')

    # Need a valid line
    try:
        line_val = float(line_str)
    except (ValueError, TypeError):
        return ''

    # Avoid: line too tight
    if line_val < 3:
        return ''

    # Avoid: short break
    if 'Short Break' in travel:
        return ''

    # Avoid: bad opponent
    if opponent in TACKLE_BAD_OPPONENTS:
        return ''

    # Avoid: GEE home game (both teams affected by Kardinia congestion — 39.5% WR)
    try:
        yr_rnd = (int(year_str), int(rnd_str))
        if yr_rnd in GEE_HOME_ROUNDS and (opponent == 'GEE' or team == 'GEE'):
            return ''
    except (ValueError, TypeError):
        pass

    # T1: Wing/Ruck (no AvL filter)
    if position in ('Wing', 'Ruck'):
        return 'T1'

    # Avoid: AvL >= 10%
    try:
        if float(avl_str.replace('%', '').replace('+', '')) >= 10.0:
            return ''
    except (ValueError, TypeError):
        pass

    # Avoid: FwdMid + any Easy DvP
    has_any_easy = any(x in dvp for x in ('Slight Easy', 'Moderate Easy', 'Strong Easy'))
    if position == 'FwdMid' and has_any_easy:
        return ''

    # T2: everything else that passes
    return 'T2'


def main():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds  = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_FILE, scopes=scopes)
    client = gspread.authorize(creds)
    sheet  = client.open_by_key(GOOGLE_SHEET_ID)
    ws     = sheet.worksheet(GOOGLE_SHEET_TAB)

    print("Reading sheet ...")
    all_values = ws.get_all_values()
    if not all_values:
        print("Sheet is empty.")
        return

    header    = all_values[0]
    data_rows = all_values[1:]

    try:
        strat_idx = header.index('Strategy')
    except ValueError:
        print("ERROR: 'Strategy' column not found.")
        return

    strat_col_1 = strat_idx + 1   # 1-based for gspread

    print(f"Found 'Strategy' at column {strat_col_1}  ({len(data_rows)} data rows)")

    cell_updates = []
    counts = {'T1': 0, 'T2': 0, 'blank': 0, 'unchanged': 0}

    for i, row in enumerate(data_rows):
        sheet_row = i + 2   # header = row 1

        current = row[strat_idx].strip() if strat_idx < len(row) else ''
        new_val  = classify(row, header)

        if new_val == current:
            counts['unchanged'] += 1
            continue

        cell_ref = gspread.utils.rowcol_to_a1(sheet_row, strat_col_1)
        cell_updates.append({'range': cell_ref, 'values': [[new_val]]})

        if new_val == 'T1':
            counts['T1'] += 1
        elif new_val == 'T2':
            counts['T2'] += 1
        else:
            counts['blank'] += 1

    if not cell_updates:
        print("Nothing to change -- all rows already correct.")
        return

    print(f"\nUpdating {len(cell_updates)} rows ...")
    print(f"     -> T1:    {counts['T1']}")
    print(f"     -> T2:    {counts['T2']}")
    print(f"     -> blank: {counts['blank']}")
    print(f"     (unchanged: {counts['unchanged']})")

    chunk = 500
    for start in range(0, len(cell_updates), chunk):
        ws.batch_update(cell_updates[start:start + chunk])
        end = min(start + chunk, len(cell_updates))
        print(f"  OK  Rows {start+1}-{end} written")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
