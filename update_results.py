"""
update_results.py
─────────────────
Run this once mid-week after downloading the latest afl_player_stats.csv.
It finds every row in your Google Sheet that has an empty "Actual" column,
looks up the real stat from the CSV, fills in Actual and W/L automatically.

Usage:
    python update_results.py

Requirements (install once):
    pip install gspread google-auth pandas
"""

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ── config — must match app.py ────────────────────────────────────────────────
GOOGLE_SHEET_ID         = "10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"
GOOGLE_SHEET_TAB        = "Bet Log"   # change if your tab is named differently
STATS_CSV               = "afl_player_stats.csv"
# ─────────────────────────────────────────────────────────────────────────────

STAT_TYPE_MAP = {
    "Disposal": "disposals",
    "Mark":     "marks",
    "Tackle":   "tackles",
}

# Column indices in the sheet (0-based for gspread row lists)
# Type Year Round Player Team Opponent Position TravelFatigue Weather DvP
# Line AvgVsLine LineConsistency BetPriority BetFlag Actual W/L
# 0    1    2     3      4    5        6        7             8       9
# 10   11          12               13          14      15     16
COL_TYPE     = 0
COL_ROUND    = 2
COL_PLAYER   = 3
COL_ACTUAL   = 15   # "Actual" is the 16th column
COL_WL       = 16   # "W/L"    is the 17th column


def get_sheets_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds  = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_FILE, scopes=scopes)
    return gspread.authorize(creds)


def load_stats():
    """Load the stats CSV and compute disposals if needed."""
    try:
        df = pd.read_csv(STATS_CSV, skiprows=3).fillna(0)
    except FileNotFoundError:
        print(f"❌  {STATS_CSV} not found. Download the latest stats file first.")
        raise

    if 'disposals' not in df.columns:
        if 'kicks' in df.columns and 'handballs' in df.columns:
            df['disposals'] = df['kicks'] + df['handballs']
        else:
            df['disposals'] = 0

    # Normalise player names to lowercase for matching
    df['player_lower'] = df['player'].str.lower().str.strip()
    return df


def find_actual(stats_df, player_name: str, round_num: int, stat_col: str):
    """
    Return the actual stat value for a player in a given round, or None.
    Tries exact match first, then last-name + first-initial.
    """
    pn = player_name.lower().strip()

    # exact
    row = stats_df[(stats_df['player_lower'] == pn) & (stats_df['round'] == round_num)]
    if not row.empty:
        return float(row.iloc[0][stat_col])

    # last name + first initial
    parts = pn.split()
    if len(parts) >= 2:
        last  = parts[-1]
        first = parts[0][0]
        row   = stats_df[
            (stats_df['player_lower'].str.endswith(last)) &
            (stats_df['player_lower'].str.startswith(first)) &
            (stats_df['round'] == round_num)
        ]
        if len(row) == 1:
            return float(row.iloc[0][stat_col])

    return None


def calculate_wl(actual: float, line: str) -> str:
    """Return '1' (win), '-1' (loss), or '0' (push)."""
    try:
        line_val = float(line)
    except (ValueError, TypeError):
        return ""
    if actual < line_val:
        return "1"
    elif actual > line_val:
        return "-1"
    else:
        return "0"


def remove_duplicates():
    """
    Scan the sheet for duplicate rows (same Type + Round + Player + Line)
    and delete the extras. When a duplicate pair exists, the row that already
    has an Actual value is kept; otherwise the first occurrence is kept.
    Rows are deleted bottom-to-top so indices don't shift mid-loop.
    """
    print("🔍  Scanning for duplicates …")
    client    = get_sheets_client()
    sheet     = client.open_by_key(GOOGLE_SHEET_ID)
    worksheet = sheet.worksheet(GOOGLE_SHEET_TAB)

    all_rows = worksheet.get_all_values()
    if not all_rows:
        print("⚠️  Sheet is empty.")
        return

    header    = all_rows[0]
    data_rows = all_rows[1:]

    def col_idx(name):
        try:
            return header.index(name)
        except ValueError:
            return None

    ci_type   = col_idx("Type")   or COL_TYPE
    ci_round  = col_idx("Round")  or COL_ROUND
    ci_player = col_idx("Player") or COL_PLAYER
    ci_line   = col_idx("Line")
    ci_actual = col_idx("Actual") or COL_ACTUAL

    # first pass — find which sheet row (1-based) to keep per key
    # sheet row = data index + 2  (header is row 1, data starts row 2)
    seen   = {}   # key → sheet_row_number of the row we intend to keep
    delete = []   # sheet row numbers to delete

    for i, row in enumerate(data_rows):
        sheet_row = i + 2   # 1-based, offset for header

        # pad if needed
        max_col = max(c for c in [ci_type, ci_round, ci_player, ci_line, ci_actual] if c is not None)
        while len(row) <= max_col:
            row.append("")

        t = row[ci_type].strip()   if ci_type   is not None else ""
        r = row[ci_round].strip()  if ci_round  is not None else ""
        p = row[ci_player].strip() if ci_player is not None else ""
        l = row[ci_line].strip()   if ci_line   is not None else ""

        if not t or not r or not p:
            continue

        key = (t, r, p, l)

        if key not in seen:
            seen[key] = sheet_row
        else:
            # decide which to keep — prefer the one with Actual filled
            existing_sheet_row = seen[key]
            existing_data_row  = data_rows[existing_sheet_row - 2]
            while len(existing_data_row) <= (ci_actual or 0):
                existing_data_row.append("")
            existing_actual = existing_data_row[ci_actual].strip() if ci_actual is not None else ""
            this_actual     = row[ci_actual].strip()                if ci_actual is not None else ""

            if this_actual and not existing_actual:
                # new row has data, existing doesn't — delete existing, keep new
                delete.append(existing_sheet_row)
                seen[key] = sheet_row
            else:
                # keep existing, delete this one
                delete.append(sheet_row)

    if not delete:
        print("✅  No duplicates found.")
        return

    print(f"🗑️   Removing {len(delete)} duplicate row(s) …")

    # Build one batch request — sort descending so indices don't shift
    requests = [
        {
            "deleteDimension": {
                "range": {
                    "sheetId":    worksheet.id,
                    "dimension":  "ROWS",
                    "startIndex": row - 1,   # 0-based
                    "endIndex":   row,        # exclusive
                }
            }
        }
        for row in sorted(delete, reverse=True)
    ]

    # Send in chunks of 500 (well within API limits as a single batch call)
    chunk_size = 500
    for i in range(0, len(requests), chunk_size):
        worksheet.spreadsheet.batch_update({"requests": requests[i:i + chunk_size]})
        print(f"  ✅  Batch deleted rows {i + 1}–{min(i + chunk_size, len(requests))}")

    print(f"✅  Deduplication complete — {len(delete)} row(s) removed.\n")


def update_results():
    print("🔄  Connecting to Google Sheets …")
    client    = get_sheets_client()
    sheet     = client.open_by_key(GOOGLE_SHEET_ID)
    worksheet = sheet.worksheet(GOOGLE_SHEET_TAB)

    print("📥  Reading sheet data …")
    all_rows = worksheet.get_all_values()

    if not all_rows:
        print("⚠️  Sheet is empty.")
        return

    header    = all_rows[0]
    data_rows = all_rows[1:]           # skip header

    # find column positions from header (defensive — handles reordering)
    def col_idx(name):
        try:
            return header.index(name)
        except ValueError:
            return None

    ci_type   = col_idx("Type")   or COL_TYPE
    ci_round  = col_idx("Round")  or COL_ROUND
    ci_player = col_idx("Player") or COL_PLAYER
    ci_line   = col_idx("Line")
    ci_actual = col_idx("Actual") or COL_ACTUAL
    ci_wl     = col_idx("W/L")    or COL_WL

    print("📊  Loading stats CSV …")
    stats_df = load_stats()

    updates     = []   # list of (sheet_row_1indexed, actual_value, wl_value)
    rows_filled = 0
    rows_skipped = 0

    for i, row in enumerate(data_rows, start=2):   # start=2 because row 1 is header
        # Pad short rows
        while len(row) <= max(ci_actual, ci_wl):
            row.append("")

        actual_val = row[ci_actual].strip()
        if actual_val != "":
            rows_skipped += 1
            continue   # already filled — skip

        player_name = row[ci_player].strip()
        raw_round   = row[ci_round].strip()
        stat_type   = row[ci_type].strip()       # "Disposal", "Mark", "Tackle"
        line_str    = row[ci_line].strip() if ci_line is not None else ""

        if not player_name or not raw_round or not stat_type:
            continue

        try:
            round_num = int(raw_round)
        except ValueError:
            continue

        stat_col = STAT_TYPE_MAP.get(stat_type)
        if not stat_col:
            print(f"  ⚠️  Unknown stat type '{stat_type}' for {player_name} — skipping")
            continue

        if stat_col not in stats_df.columns:
            print(f"  ⚠️  Column '{stat_col}' not in stats CSV — skipping")
            continue

        actual = find_actual(stats_df, player_name, round_num, stat_col)

        if actual is None:
            print(f"  ⚠️  No stats found for {player_name} R{round_num} ({stat_type})")
            continue

        wl = calculate_wl(actual, line_str)

        updates.append({
            "row":    i,
            "actual": actual,
            "wl":     wl,
            "label":  f"{player_name} R{round_num} {stat_type}: {actual} vs line {line_str} → {wl}",
        })
        rows_filled += 1

    if not updates:
        print(f"✅  Nothing to update ({rows_skipped} rows already had Actual values).")
        return

    print(f"\n📝  Updating {rows_filled} rows …")

    # Batch update using gspread's batch_update for speed
    cell_updates = []
    for u in updates:
        actual_cell = gspread.utils.rowcol_to_a1(u["row"], ci_actual + 1)
        wl_cell     = gspread.utils.rowcol_to_a1(u["row"], ci_wl     + 1)
        cell_updates.append({"range": actual_cell, "values": [[u["actual"]]]})
        cell_updates.append({"range": wl_cell,     "values": [[u["wl"]]]})
        print(f"  ✅  {u['label']}")

    worksheet.batch_update(cell_updates)

    print(f"\n🎉  Done — {rows_filled} rows updated, {rows_skipped} already complete.")


PAIR_LOG_TAB = "Pair Log"


def backfill_pair_log():
    """
    Cross-reference the Pair Log tab against Bet Log results and fill in
    Leg1 Hit, Leg2 Hit, and Pair W/L for any pairs not yet resolved.

    Leg values: '1' = win, '-1' = loss, '0' = push
    Pair W/L  : 'W' = both hit, 'L' = at least one leg lost, 'P' = push (no loss, one+ push)
    """
    print("🔄  Backfilling Pair Log …")
    client = get_sheets_client()
    sheet  = client.open_by_key(GOOGLE_SHEET_ID)

    try:
        pair_ws = sheet.worksheet(PAIR_LOG_TAB)
    except Exception:
        print("⚠️  No 'Pair Log' tab found — skipping.")
        return

    # ── Load Bet Log as a lookup ──────────────────────────────────────────────
    bet_ws   = sheet.worksheet(GOOGLE_SHEET_TAB)
    bet_rows = bet_ws.get_all_records()
    bet_df   = pd.DataFrame(bet_rows)

    if bet_df.empty:
        print("⚠️  Bet Log is empty — no results to cross-reference.")
        return

    bet_df.columns      = [c.strip() for c in bet_df.columns]
    bet_df['pl_lower']  = bet_df['Player'].astype(str).str.lower().str.strip()
    bet_df['Round']     = pd.to_numeric(bet_df['Round'], errors='coerce')

    def lookup_wl(player, round_num, stat_type):
        pn  = str(player).lower().strip()
        sub = bet_df[(bet_df['pl_lower'] == pn) & (bet_df['Round'] == round_num)]
        if 'Type' in bet_df.columns and stat_type:
            typed = sub[sub['Type'].str.strip() == stat_type]
            if not typed.empty:
                sub = typed
        if sub.empty:
            return ''
        wl = str(sub.iloc[0].get('W/L', '')).strip()
        return wl if wl in ('1', '-1', '0', '1.0', '-1.0', '0.0') else ''

    # ── Load Pair Log ─────────────────────────────────────────────────────────
    all_values = pair_ws.get_all_values()
    if len(all_values) < 2:
        print("✅  Pair Log has no data rows.")
        return

    header    = all_values[0]
    data_rows = all_values[1:]

    def ci(name):
        try:    return header.index(name)
        except: return None

    ci_round = ci("Round");     ci_p1   = ci("Leg1 Player"); ci_t1  = ci("Leg1 Type")
    ci_p2    = ci("Leg2 Player"); ci_t2 = ci("Leg2 Type")
    ci_l1h   = ci("Leg1 Hit"); ci_l2h  = ci("Leg2 Hit");    ci_pwl = ci("Pair W/L")

    if any(x is None for x in [ci_round, ci_p1, ci_p2, ci_l1h, ci_l2h, ci_pwl]):
        print("❌  Pair Log is missing expected columns. Check the header row.")
        return

    updates = []
    max_col = max(ci_round, ci_p1, ci_p2, ci_l1h, ci_l2h, ci_pwl)

    for i, row in enumerate(data_rows):
        while len(row) <= max_col:
            row.append('')

        # Skip rows already fully resolved
        if row[ci_l1h].strip() and row[ci_l2h].strip():
            continue

        try:
            round_num = int(row[ci_round])
        except (ValueError, TypeError):
            continue

        p1 = row[ci_p1].strip()
        p2 = row[ci_p2].strip()
        t1 = row[ci_t1].strip() if ci_t1 is not None else ''
        t2 = row[ci_t2].strip() if ci_t2 is not None else ''

        if not p1 or not p2:
            continue

        wl1 = lookup_wl(p1, round_num, t1)
        wl2 = lookup_wl(p2, round_num, t2)

        if not wl1 or not wl2:
            print(f"  ⚠️  No result yet for {p1} or {p2} R{round_num} — skipping")
            continue

        wl1f = float(wl1)
        wl2f = float(wl2)
        if wl1f == 1 and wl2f == 1:
            pair_wl = "W"
        elif wl1f == -1 or wl2f == -1:
            pair_wl = "L"
        else:
            pair_wl = "P"   # at least one push, no loss

        sheet_row = i + 2   # 1-based (header = row 1)
        updates.append({
            'row':     sheet_row,
            'l1h_col': ci_l1h + 1,
            'l2h_col': ci_l2h + 1,
            'pwl_col': ci_pwl + 1,
            'l1h':     wl1,
            'l2h':     wl2,
            'pair_wl': pair_wl,
            'label':   f"{p1} + {p2}  R{round_num}: {wl1} + {wl2} → {pair_wl}",
        })

    if not updates:
        print("✅  Nothing to backfill — all Pair Log rows are already resolved.")
        return

    print(f"\n📝  Resolving {len(updates)} pair(s) …")
    cell_updates = []
    for u in updates:
        cell_updates.append({"range": gspread.utils.rowcol_to_a1(u['row'], u['l1h_col']), "values": [[u['l1h']]]})
        cell_updates.append({"range": gspread.utils.rowcol_to_a1(u['row'], u['l2h_col']), "values": [[u['l2h']]]})
        cell_updates.append({"range": gspread.utils.rowcol_to_a1(u['row'], u['pwl_col']), "values": [[u['pair_wl']]]})
        print(f"  ✅  {u['label']}")

    pair_ws.batch_update(cell_updates)
    print(f"\n🎉  Done — {len(updates)} pair(s) backfilled in Pair Log.")


if __name__ == "__main__":
    remove_duplicates()
    update_results()
    backfill_pair_log()
