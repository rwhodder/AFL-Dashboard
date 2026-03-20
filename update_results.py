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


if __name__ == "__main__":
    update_results()
