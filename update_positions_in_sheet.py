"""
update_positions_in_sheet.py
────────────────────────────
Reads player_positions.csv and overwrites the Position column in every
row of the Google Sheet Bet Log.

Run AFTER you have finished editing player_positions.csv:
    python update_positions_in_sheet.py
"""

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

GOOGLE_SHEET_ID         = "10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"
GOOGLE_SHEET_TAB        = "Bet Log"
PLAYER_POSITIONS_FILE   = "player_positions.csv"

COL_PLAYER   = 3   # 0-based index of "Player" column
COL_POSITION = 6   # 0-based index of "Position" column


def get_sheets_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_FILE, scopes=scopes)
    return gspread.authorize(creds)


def load_position_lookup():
    pos_df = pd.read_csv(PLAYER_POSITIONS_FILE)
    return dict(zip(pos_df["player"].str.lower().str.strip(), pos_df["position"]))


def update_positions():
    print("📥  Loading player_positions.csv …")
    pos_lookup = load_position_lookup()
    print(f"    {len(pos_lookup)} players loaded.\n")

    print("🔄  Connecting to Google Sheets …")
    client    = get_sheets_client()
    sheet     = client.open_by_key(GOOGLE_SHEET_ID)
    worksheet = sheet.worksheet(GOOGLE_SHEET_TAB)

    print("📋  Reading sheet …")
    all_rows = worksheet.get_all_values()
    if not all_rows:
        print("⚠️  Sheet is empty.")
        return

    header    = all_rows[0]
    data_rows = all_rows[1:]

    # Resolve column indices from header (defensive against reordering)
    def col_idx(name, fallback):
        try:
            return header.index(name)
        except ValueError:
            return fallback

    ci_player   = col_idx("Player",   COL_PLAYER)
    ci_position = col_idx("Position", COL_POSITION)

    print(f"    Player col={ci_player}, Position col={ci_position}\n")

    cell_updates = []
    matched      = 0
    no_match     = []

    for i, row in enumerate(data_rows, start=2):   # row 1 = header, data from row 2
        # Pad short rows
        while len(row) <= max(ci_player, ci_position):
            row.append("")

        player_name = row[ci_player].strip()
        if not player_name:
            continue

        new_pos = pos_lookup.get(player_name.lower())
        if new_pos is None:
            no_match.append(player_name)
            continue

        current_pos = row[ci_position].strip()
        if current_pos == new_pos:
            continue   # already correct — skip

        cell_ref = gspread.utils.rowcol_to_a1(i, ci_position + 1)
        cell_updates.append({"range": cell_ref, "values": [[new_pos]]})
        matched += 1

    if not cell_updates:
        print("✅  All positions already up to date — nothing to change.")
    else:
        print(f"📝  Updating {matched} cell(s) …")
        # Batch in chunks of 500 to stay within API limits
        chunk_size = 500
        for start in range(0, len(cell_updates), chunk_size):
            worksheet.batch_update(cell_updates[start:start + chunk_size])
            end = min(start + chunk_size, len(cell_updates))
            print(f"    ✅  Sent batch {start + 1}–{end}")
        print(f"\n🎉  Done — {matched} position(s) updated.")

    if no_match:
        unique_missing = sorted(set(no_match))
        print(f"\n⚠️  {len(unique_missing)} player(s) in the sheet not found in player_positions.csv:")
        for name in unique_missing:
            print(f"   • {name}")
        print("   → Add them to player_positions.csv and re-run.")


if __name__ == "__main__":
    update_positions()
