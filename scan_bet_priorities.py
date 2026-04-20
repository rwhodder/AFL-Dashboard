"""
scan_bet_priorities.py
──────────────────────
Read the Bet Log and show what Bet Priority values exist,
broken down by Type and Position, so we can confirm the T1/T2 mapping
before touching anything.
"""

import pandas as pd
from google.oauth2.service_account import Credentials
import gspread

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
    sheet  = client.open_by_key(GOOGLE_SHEET_ID)
    ws     = sheet.worksheet(GOOGLE_SHEET_TAB)

    records = ws.get_all_records()
    df = pd.DataFrame(records)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    print(f"Total rows: {len(df)}\n")
    print("Columns:", list(df.columns))
    print()

    print("=" * 60)
    print("  BET PRIORITY VALUES  (count of each)")
    print("=" * 60)
    counts = df["Strategy"].value_counts(dropna=False)
    for val, n in counts.items():
        label = "(blank)" if val in ("", "nan") else val
        print(f"  {label:<15} {n:>4} rows")
    print()

    print("=" * 60)
    print("  BREAKDOWN BY TYPE × BET PRIORITY")
    print("=" * 60)
    cross = pd.crosstab(
        df["Type"].replace("", "(blank)").replace("nan", "(blank)"),
        df["Strategy"].replace("", "(blank)").replace("nan", "(blank)"),
    )
    print(cross.to_string())
    print()

    print("=" * 60)
    print("  BREAKDOWN BY POSITION × BET PRIORITY  (non-blank priority only)")
    print("=" * 60)
    flagged = df[~df["Strategy"].isin(["", "nan"])]
    cross2 = pd.crosstab(
        flagged["Position"].replace("", "(blank)"),
        flagged["Strategy"],
    )
    print(cross2.to_string())
    print()

    print("=" * 60)
    print("  BREAKDOWN BY DVP × BET PRIORITY  (non-blank priority only)")
    print("=" * 60)
    cross3 = pd.crosstab(
        flagged["DvP"].replace("", "(blank)"),
        flagged["Strategy"],
    )
    print(cross3.to_string())


if __name__ == "__main__":
    main()
