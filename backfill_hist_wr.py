"""
backfill_hist_wr.py
───────────────────
Retrospectively computes Hist WR, WR Range, and Confidence for every Tackle
row in the Bet Log using ONLY data available before that round.

Why at-time-of-round, not current data:
  Using current data introduces look-ahead bias — early rounds would show
  High Confidence because they now have 40+ results, but at the time there
  were 3. That measures nothing useful. At-time computation simulates what
  you would have seen had the system existed from day 1, and is the only
  valid way to measure whether following these signals was profitable.

Algorithm:
  1. Read all rows sorted by (Year, Round).
  2. For each round group, compute Hist WR using only Tackle rows from
     earlier rounds that have a resolved W/L.
  3. After processing a round, add its resolved rows to the running history
     so the next round can see them.
  4. Batch-write Hist WR, WR Range, Confidence back to the sheet.

Non-Tackle rows (Disposal, Mark) are skipped — the cascade lookup is
tackle-only and would return empty strings anyway.

Usage:
    python backfill_hist_wr.py
    python backfill_hist_wr.py --dry-run   # print stats but do not write
"""

import math
import sys
from itertools import groupby

import gspread
from google.oauth2.service_account import Credentials

GOOGLE_SHEET_ID         = "10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"
GOOGLE_SHEET_TAB        = "Bet Log"

DRY_RUN = "--dry-run" in sys.argv


# ── WR helpers (mirrors app.py exactly) ──────────────────────────────────────

def _compute_hist_wr(history: list, position: str, dvp: str, opponent: str) -> str:
    if not history:
        return ""

    def _wr(pred):
        sub = [r for r in history if pred(r)]
        n = len(sub)
        if n == 0:
            return None, 0
        return sum(r["win"] for r in sub) / n * 100, n

    wr, n = _wr(lambda r: r["pos"] == position and r["dvp"] == dvp and r["opp"] == opponent)
    if n >= 5:
        return f"{wr:.0f}% ({n})"

    wr, n = _wr(lambda r: r["pos"] == position and r["dvp"] == dvp)
    if n >= 8:
        return f"{wr:.0f}% ({n})"

    wr, n = _wr(lambda r: r["pos"] == position)
    if n >= 5:
        return f"{wr:.0f}% ({n})"

    return ""


def _parse_wr(s: str):
    try:
        pct_part, n_part = s.split("%")
        return float(pct_part.strip()), int(n_part.strip().strip("()"))
    except Exception:
        return None, 0


def _wr_range(s: str) -> str:
    pct, n = _parse_wr(s)
    if pct is None or n == 0:
        return ""
    se = math.sqrt(pct / 100 * (1 - pct / 100) / n) * 100
    return f"{max(0.0, pct - se):.0f}-{min(100.0, pct + se):.0f}%"


def _confidence(s: str) -> str:
    _, n = _parse_wr(s)
    if n >= 30: return "High"
    if n >= 15: return "Med"
    if n >= 5:  return "Low"
    return "—"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Mode: {'DRY RUN (no writes)' if DRY_RUN else 'LIVE'}")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds  = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_FILE, scopes=scopes)
    client = gspread.authorize(creds)
    ws     = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_TAB)

    print("Reading sheet ...")
    all_values = ws.get_all_values()
    if not all_values:
        print("Sheet is empty.")
        return

    header    = all_values[0]
    data_rows = all_values[1:]
    print(f"  {len(data_rows)} data rows found")

    def ci(name):
        try:
            return header.index(name)
        except ValueError:
            raise ValueError(f"Column '{name}' not found. Header: {header}")

    idx = {
        "type":       ci("Type"),
        "year":       ci("Year"),
        "round":      ci("Round"),
        "position":   ci("Position"),
        "dvp":        ci("DvP"),
        "opponent":   ci("Opponent"),
        "wl":         ci("W/L"),
        "hist_wr":    ci("Hist WR"),
        "wr_range":   ci("Range"),
        "confidence": ci("Confidence"),
    }

    # 1-based column numbers for gspread
    col1 = {k: idx[k] + 1 for k in ("hist_wr", "wr_range", "confidence")}

    def get(row, key):
        i = idx[key]
        return str(row[i]).strip() if i < len(row) else ""

    def sort_key(row):
        try:
            y = int(get(row, "year"))
        except ValueError:
            y = 0
        try:
            r = int(get(row, "round"))
        except ValueError:
            r = 0
        return (y, r)

    # Sort rows chronologically (keep original index so we know which sheet row to write)
    indexed = sorted(enumerate(data_rows), key=lambda x: sort_key(x[1]))

    # Process one round group at a time
    cumulative_history = []   # Tackle rows resolved before the current round
    cell_updates       = []
    counts = {"updated": 0, "empty": 0, "skipped_non_tackle": 0, "skipped_no_wl": 0}

    for (year, rnd), group_iter in groupby(indexed, key=lambda x: sort_key(x[1])):
        group = list(group_iter)
        snapshot = list(cumulative_history)   # history visible at round start

        for orig_i, row in group:
            sheet_row = orig_i + 2  # 1-indexed, header is row 1

            if get(row, "type") != "Tackle":
                counts["skipped_non_tackle"] += 1
                continue

            position = get(row, "position")
            dvp      = get(row, "dvp")
            opponent = get(row, "opponent")

            hist_wr    = _compute_hist_wr(snapshot, position, dvp, opponent)
            wr_range   = _wr_range(hist_wr)
            confidence = _confidence(hist_wr)

            if not hist_wr:
                counts["empty"] += 1
            else:
                counts["updated"] += 1

            for col_key, value in [("hist_wr", hist_wr), ("wr_range", wr_range), ("confidence", confidence)]:
                cell_ref = gspread.utils.rowcol_to_a1(sheet_row, col1[col_key])
                cell_updates.append({"range": cell_ref, "values": [[value]]})

        # Add this round's resolved Tackle results to cumulative history
        for orig_i, row in group:
            if get(row, "type") != "Tackle":
                continue
            wl_str = get(row, "wl")
            if wl_str not in {"1", "-1", "0", "1.0", "-1.0", "0.0"}:
                counts["skipped_no_wl"] += 1
                continue
            try:
                wl = float(wl_str)
            except ValueError:
                continue
            cumulative_history.append({
                "pos": get(row, "position"),
                "dvp": get(row, "dvp"),
                "opp": get(row, "opponent"),
                "win": 1 if wl == 1.0 else 0,
            })

    tackle_rows = counts["updated"] + counts["empty"]
    print(f"\n  Tackle rows processed:  {tackle_rows}")
    print(f"  → Hist WR populated:    {counts['updated']}")
    print(f"  → Not enough data (—):  {counts['empty']}")
    print(f"  Non-Tackle rows skipped: {counts['skipped_non_tackle']}")
    print(f"  Resolved rows added to history: {len(cumulative_history)}")
    print(f"\n  Total cell writes queued: {len(cell_updates)}")

    if DRY_RUN:
        print("\nDRY RUN — no changes written to sheet.")
        return

    if not cell_updates:
        print("Nothing to write.")
        return

    print("\nWriting to sheet ...")
    chunk = 500
    for start in range(0, len(cell_updates), chunk):
        end = min(start + chunk, len(cell_updates))
        ws.batch_update(cell_updates[start:end])
        print(f"  OK  cells {start + 1}–{end}")

    print(f"\nDone — {tackle_rows} Tackle rows backfilled with at-time Hist WR.")


if __name__ == "__main__":
    main()
