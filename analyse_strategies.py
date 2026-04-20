"""
analyse_strategies.py
─────────────────────
Reads your Google Sheet bet log and prints a full strategy performance report.
Also scans unflagged bets for hidden winning patterns.

Usage:
    python analyse_strategies.py
"""

import pandas as pd
from google.oauth2.service_account import Credentials
import gspread

GOOGLE_SHEET_ID         = "10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"
GOOGLE_SHEET_TAB        = "Bet Log"
MIN_SAMPLE              = 10   # minimum bets before reporting a pattern


def get_worksheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds     = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_FILE, scopes=scopes)
    client    = gspread.authorize(creds)
    sheet     = client.open_by_key(GOOGLE_SHEET_ID)
    return sheet.worksheet(GOOGLE_SHEET_TAB)


def load_data():
    print("📥  Reading sheet data …\n")
    ws      = get_worksheet()
    records = ws.get_all_records()
    df      = pd.DataFrame(records)

    # Normalise
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Only rows with a result
    df = df[df["W/L"].isin(["1", "-1", "0", "1.0", "-1.0", "0.0"])]
    df["W/L"]          = pd.to_numeric(df["W/L"], errors="coerce")
    df["Round"]        = pd.to_numeric(df["Round"], errors="coerce")
    df["Line"]         = pd.to_numeric(df["Line"], errors="coerce")
    df["Bet Priority"] = df["Bet Priority"].astype(str).str.strip()

    print(f"✅  Loaded {len(df)} completed bets\n")
    return df


def win_rate(sub):
    sub = sub.dropna(subset=["W/L"])
    excl_push = sub[sub["W/L"] != 0]
    if len(excl_push) == 0:
        return 0.0, 0, 0, 0
    wins   = (excl_push["W/L"] == 1).sum()
    losses = (excl_push["W/L"] == -1).sum()
    pushes = (sub["W/L"] == 0).sum()
    wr     = wins / len(excl_push) * 100
    return round(wr, 1), int(wins), int(losses), int(pushes)


def bar(wr, width=20):
    filled = int(wr / 100 * width)
    return "█" * filled + "░" * (width - filled)


def section(title):
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def report(df):

    # ── 1. Overall summary ────────────────────────────────────────────────────
    section("OVERALL SUMMARY")
    wr, w, l, p = win_rate(df)
    print(f"  Total bets  : {len(df)}  ({w}W  {l}L  {p}P)")
    print(f"  Win rate    : {wr}%  {bar(wr)}")
    ev = round((wr / 100) ** 2 * 3.2 * 100 - 100, 1)
    print(f"  2-leg EV    : {ev:+.1f}%  (at $3.20 payout)")
    rounds = sorted(df["Round"].dropna().unique())
    if rounds:
        print(f"  Rounds      : R{int(min(rounds))} → R{int(max(rounds))}  ({len(rounds)} rounds)")
    print()

    # ── 2. Current strategies ─────────────────────────────────────────────────
    section("STRATEGY BREAKDOWN")
    TARGETS = {
        "S1": 76.7, "S2": 67.5, "S3": 63.5,
        "D1": 69.9, "D2": 69.0,
        "W1": 92.3, "W2": 86.7,
    }
    LABELS  = {
        "S1": "Tackle + Strong Unders DvP          [CONFIRMED]",
        "S2": "Tackle + Slight/Moderate Easy DvP   [CONFIRMED]",
        "S3": "Tackle + InsM position               [CONFIRMED]",
        "D1": "Tackle + Wing/Ruck                  [DEVELOPING]",
        "D2": "Disposal + SmF + Slight Unders/Easy [DEVELOPING]",
        "W1": "Mark + SmF                          [WATCH]",
        "W2": "Mark + Line ≥8.0                    [WATCH]",
    }
    flagged = df[df["Bet Priority"].isin(TARGETS.keys())]
    for p in ["S1", "S2", "S3", "D1", "D2", "W1", "W2"]:
        sub = flagged[flagged["Bet Priority"] == p]
        if sub.empty:
            continue
        wr, w, l, push = win_rate(sub)
        target = TARGETS[p]
        trend  = "↑" if wr >= target else "↓"
        vs     = round(wr - target, 1)
        # rolling last 10
        recent = sub.tail(10)
        r_wr, *_ = win_rate(recent)
        status = "CONFIRMED" if p in ("S1","S2","S3") else "DEVELOPING" if p in ("D1","D2") else "WATCH"
        print(f"  {p}  {LABELS[p]}")
        print(f"      {w}W {l}L {push}P  |  WR {wr}%  {bar(wr)}  {trend} {abs(vs)}pp vs target {target}%  [{status}]")
        print(f"      Last 10 WR: {r_wr}%")
        print()

    # ── 3. By stat type ───────────────────────────────────────────────────────
    section("BY STAT TYPE")
    for t in ["Disposal", "Mark", "Tackle"]:
        sub = df[df["Type"] == t]
        if sub.empty:
            continue
        wr, w, l, push = win_rate(sub)
        print(f"  {t:<10}  {len(sub):>4} bets  |  {w}W {l}L {push}P  |  WR {wr}%  {bar(wr)}")
    print()

    # ── 4. By position ────────────────────────────────────────────────────────
    section("BY POSITION  (all bets)")
    for pos in ["KeyF", "SmF", "MedF", "FwdMid", "GenF", "Ruck", "InsM", "Wing", "GenD", "KeyD"]:
        sub = df[df["Position"] == pos]
        if len(sub) < MIN_SAMPLE:
            continue
        wr, w, l, push = win_rate(sub)
        print(f"  {pos:<6}  {len(sub):>4} bets  |  {w}W {l}L {push}P  |  WR {wr}%  {bar(wr)}")
    print()

    # ── 5. By DvP ─────────────────────────────────────────────────────────────
    section("BY DVP RATING  (all bets)")
    dvp_order = [
        "🔴 Strong Unders", "🟠 Moderate Unders", "🟡 Slight Unders",
        "✅ Neutral",
        "🔹 Slight Easy", "🔷 Moderate Easy", "🔵 Strong Easy",
    ]
    for dvp in dvp_order:
        sub = df[df["DvP"] == dvp]
        if len(sub) < MIN_SAMPLE:
            continue
        wr, w, l, push = win_rate(sub)
        print(f"  {dvp:<25}  {len(sub):>4} bets  |  {w}W {l}L  |  WR {wr}%  {bar(wr)}")
    print()

    # ── 6. By travel fatigue ──────────────────────────────────────────────────
    section("BY TRAVEL FATIGUE  (all bets)")
    for tf in df["Travel Fatigue"].unique():
        sub = df[df["Travel Fatigue"] == tf]
        if len(sub) < MIN_SAMPLE:
            continue
        wr, w, l, push = win_rate(sub)
        label = tf[:40]
        print(f"  {label:<42}  {len(sub):>4} bets  |  {w}W {l}L  |  WR {wr}%  {bar(wr)}")
    print()

    # ── 7. Hidden patterns — unflagged bets that are winning ──────────────────
    section("HIDDEN PATTERNS  (unflagged bets only — potential new strategies)")
    unflagged = df[~df["Bet Priority"].isin(TARGETS.keys())]
    print(f"  Total unflagged completed bets: {len(unflagged)}\n")

    combos = []

    # Stat × Position
    for stat in ["Disposal", "Mark", "Tackle"]:
        for pos in ["KeyF", "SmF", "MedF", "FwdMid", "GenF", "Ruck", "InsM", "Wing", "GenD", "KeyD"]:
            sub = unflagged[(unflagged["Type"] == stat) & (unflagged["Position"] == pos)]
            if len(sub) < MIN_SAMPLE:
                continue
            wr, w, l, push = win_rate(sub)
            if wr >= 65:
                combos.append((wr, len(sub), f"{stat} × {pos}", w, l, push))

    # Stat × DvP
    for stat in ["Disposal", "Mark", "Tackle"]:
        for dvp in dvp_order:
            sub = unflagged[(unflagged["Type"] == stat) & (unflagged["DvP"] == dvp)]
            if len(sub) < MIN_SAMPLE:
                continue
            wr, w, l, push = win_rate(sub)
            if wr >= 65:
                combos.append((wr, len(sub), f"{stat} × {dvp}", w, l, push))

    # Stat × Travel
    for stat in ["Disposal", "Mark", "Tackle"]:
        for tf in unflagged["Travel Fatigue"].unique():
            sub = unflagged[(unflagged["Type"] == stat) & (unflagged["Travel Fatigue"] == tf)]
            if len(sub) < MIN_SAMPLE:
                continue
            wr, w, l, push = win_rate(sub)
            if wr >= 65:
                combos.append((wr, len(sub), f"{stat} × {tf[:30]}", w, l, push))

    # Position × DvP
    for pos in ["KeyF", "SmF", "MedF", "FwdMid", "GenF", "Ruck", "InsM", "Wing", "GenD", "KeyD"]:
        for dvp in dvp_order:
            sub = unflagged[(unflagged["Position"] == pos) & (unflagged["DvP"] == dvp)]
            if len(sub) < MIN_SAMPLE:
                continue
            wr, w, l, push = win_rate(sub)
            if wr >= 65:
                combos.append((wr, len(sub), f"{pos} × {dvp}", w, l, push))

    if combos:
        combos.sort(reverse=True)
        print(f"  {'Pattern':<45}  {'n':>4}  {'W':>4}  {'L':>4}  WR")
        print(f"  {'-'*45}  {'-'*4}  {'-'*4}  {'-'*4}  ------")
        for wr, n, label, w, l, push in combos:
            print(f"  {label:<45}  {n:>4}  {w:>4}  {l:>4}  {wr}%  {'⭐' if wr >= 75 else ''}")
    else:
        print("  No unflagged patterns above 65% WR found with enough sample size.")
    print()

    # ── 8. Round-by-round summary ─────────────────────────────────────────────
    section("WIN RATE BY ROUND  (S1–S3 + D1–D2 only)")
    print(f"  {'Round':>5}  {'Bets':>5}  {'W':>4}  {'L':>4}  WR")
    print(f"  {'─'*5}  {'─'*5}  {'─'*4}  {'─'*4}  ──────")
    for rnd in sorted(flagged["Round"].dropna().unique()):
        sub = flagged[flagged["Round"] == rnd]
        wr, w, l, push = win_rate(sub)
        print(f"  R{int(rnd):>4}  {len(sub):>5}  {w:>4}  {l:>4}  {wr}%")
    print()

    print("=" * 70)
    print("  END OF REPORT")
    print("=" * 70)


if __name__ == "__main__":
    df = load_data()
    report(df)
