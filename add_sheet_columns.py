"""
add_sheet_columns.py
────────────────────
One-time script: adds Hist WR, WR Range, Confidence columns to the
Bet Log Google Sheet header row if they don't already exist.

Safe to re-run — skips columns that are already present.

Usage:
    python add_sheet_columns.py
"""

from google.oauth2.service_account import Credentials
import gspread

GOOGLE_SHEET_ID         = "10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"
GOOGLE_SHEET_TAB        = "Bet Log"

NEW_COLUMNS = ["Hist WR", "WR Range", "Confidence"]


def main():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds  = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_FILE, scopes=scopes)
    client = gspread.authorize(creds)
    ws     = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_TAB)

    header = ws.row_values(1)
    print(f"Current columns ({len(header)}): {header}")

    to_add = [c for c in NEW_COLUMNS if c not in header]
    if not to_add:
        print("All new columns already present — nothing to do.")
        return

    # Append each new header to the end of row 1
    next_col = len(header) + 1
    for col_name in to_add:
        ws.update_cell(1, next_col, col_name)
        print(f"  Added '{col_name}' at column {next_col}")
        next_col += 1

    print(f"\nDone. Sheet now has {len(header) + len(to_add)} columns.")


if __name__ == "__main__":
    main()
