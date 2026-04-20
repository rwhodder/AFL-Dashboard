"""
analyse_avl_cons.py
───────────────────
Breaks down historical WR for criteria-flagged tackle bets by:
  - AvL bucket
  - Consistency bucket
  - Line bucket
  - AvL bucket × Position (to check position-specific effects)
  - Consistency bucket × Position

"Criteria-flagged" = Tackle bets with a non-blank Strategy (T1, T2, S1..S3 etc.)

Usage:
    python analyse_avl_cons.py
"""

import pandas as pd
from google.oauth2.service_account import Credentials
import gspread

GOOGLE_SHEET_ID         = "10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"
GOOGLE_SHEET_TAB        = "Bet Log"
MIN_SAMPLE              = 10


def get_data():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds  = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_FILE, scopes=scopes)
    client = gspread.authorize(creds)
    ws     = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_TAB)
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Only completed bets
    df = df[df["W/L"].isin(["1", "-1", "0", "1.0", "-1.0", "0.0"])].copy()
    df["W/L"]   = pd.to_numeric(df["W/L"], errors="coerce")
    df["Line"]  = pd.to_numeric(df["Line"], errors="coerce")
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce")

    # Parse AvL: "+12.5%" → 12.5
    df["_avl"] = pd.to_numeric(
        df["Avg vs Line"].astype(str).str.replace("%","").str.replace("+",""),
        errors="coerce"
    )

    # Parse Consistency: "62.5%" → 62.5
    df["_cons"] = pd.to_numeric(
        df["Line Consistency"].astype(str).str.replace("%",""),
        errors="coerce"
    )

    print(f"Loaded {len(df)} completed bets\n")
    return df


def win_rate(sub):
    sub = sub.dropna(subset=["W/L"])
    excl = sub[sub["W/L"] != 0]
    if len(excl) == 0:
        return None, 0
    wr = (excl["W/L"] == 1).sum() / len(excl) * 100
    return round(wr, 1), len(excl)


def section(title):
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def breakdown(df, group_col, buckets, label_col=None):
    """Print WR breakdown for labelled buckets."""
    rows = []
    for label, mask in buckets:
        sub = df[mask(df)]
        wr, n = win_rate(sub)
        if n >= MIN_SAMPLE:
            rows.append((label, n, wr))
    if not rows:
        print("  (no buckets with enough data)")
        return
    print(f"  {'Bucket':<30}  {'n':>5}  {'WR':>7}")
    print(f"  {'-'*30}  {'-'*5}  {'-'*7}")
    for label, n, wr in rows:
        flag = "  <- AVOID?" if wr is not None and wr < 54 else ""
        flag = "  <- STRONG" if wr is not None and wr >= 65 else flag
        wr_str = f"{wr}%" if wr is not None else "n/a"
        print(f"  {label:<30}  {n:>5}  {wr_str:>7}{flag}")
    print()


def main():
    df = get_data()

    # Criteria-flagged tackle bets only
    tackles = df[
        (df["Type"] == "Tackle") &
        (~df["Strategy"].isin(["", "nan"])) &
        (df["_avl"].notna()) &
        (df["_cons"].notna())
    ].copy()

    print(f"Criteria-flagged tackle bets with AvL + Consistency data: {len(tackles)}\n")

    # ── AvL buckets ──────────────────────────────────────────────────────────
    section("AVL BREAKDOWN  (criteria-flagged tackles)")
    avl_buckets = [
        ("AvL <= -20%  (Dabble oversets big)", lambda d: d["_avl"] <= -20),
        ("AvL -20% to -10%",                   lambda d: (d["_avl"] > -20) & (d["_avl"] <= -10)),
        ("AvL -10% to 0%",                     lambda d: (d["_avl"] > -10) & (d["_avl"] <= 0)),
        ("AvL 0% to +10%",                     lambda d: (d["_avl"] > 0)   & (d["_avl"] <= 10)),
        ("AvL +10% to +20%",                   lambda d: (d["_avl"] > 10)  & (d["_avl"] <= 20)),
        ("AvL > +20%  (Dabble has line right)", lambda d: d["_avl"] > 20),
    ]
    breakdown(tackles, "_avl", avl_buckets)

    # ── Consistency buckets ──────────────────────────────────────────────────
    section("CONSISTENCY BREAKDOWN  (criteria-flagged tackles)")
    cons_buckets = [
        ("Cons < 45%",    lambda d: d["_cons"] < 45),
        ("Cons 45-55%",   lambda d: (d["_cons"] >= 45) & (d["_cons"] < 55)),
        ("Cons 55-65%",   lambda d: (d["_cons"] >= 55) & (d["_cons"] < 65)),
        ("Cons 65-80%",   lambda d: (d["_cons"] >= 65) & (d["_cons"] < 80)),
        ("Cons 80%+",     lambda d: d["_cons"] >= 80),
    ]
    breakdown(tackles, "_cons", cons_buckets)

    # ── Line buckets ─────────────────────────────────────────────────────────
    section("LINE BREAKDOWN  (criteria-flagged tackles)")
    line_buckets = [
        ("Line 3.0",       lambda d: d["Line"] == 3.0),
        ("Line 3.5",       lambda d: d["Line"] == 3.5),
        ("Line 4.0",       lambda d: d["Line"] == 4.0),
        ("Line 4.5",       lambda d: d["Line"] == 4.5),
        ("Line 5.0",       lambda d: d["Line"] == 5.0),
        ("Line 5.5+",      lambda d: d["Line"] >= 5.5),
    ]
    breakdown(tackles, "Line", line_buckets)

    # ── AvL × Position ───────────────────────────────────────────────────────
    section("AVL > +10%  breakdown by Position  (criteria-flagged tackles)")
    high_avl = tackles[tackles["_avl"] > 10]
    print(f"  Total bets with AvL > +10%: {len(high_avl)}\n")
    positions = ["InsM", "Wing", "SmF", "FwdMid", "GenD", "Ruck", "KeyD", "KeyF"]
    print(f"  {'Position':<10}  {'n':>5}  {'WR':>7}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*7}")
    for pos in positions:
        sub = high_avl[high_avl["Position"] == pos]
        wr, n = win_rate(sub)
        if n >= 5:
            flag = "  <- AVOID?" if wr is not None and wr < 54 else ""
            flag = "  <- FINE"   if wr is not None and wr >= 60 else flag
            wr_str = f"{wr}%" if wr is not None else "n/a"
            print(f"  {pos:<10}  {n:>5}  {wr_str:>7}{flag}")
    print()

    # Consistency x Position
    section("CONSISTENCY < 55%  breakdown by Position  (criteria-flagged tackles)")
    low_cons = tackles[tackles["_cons"] < 55]
    print(f"  Total bets with Consistency < 55%: {len(low_cons)}\n")
    print(f"  {'Position':<10}  {'n':>5}  {'WR':>7}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*7}")
    for pos in positions:
        sub = low_cons[low_cons["Position"] == pos]
        wr, n = win_rate(sub)
        if n >= 5:
            flag = "  <- AVOID?" if wr is not None and wr < 54 else ""
            flag = "  <- FINE"   if wr is not None and wr >= 60 else flag
            wr_str = f"{wr}%" if wr is not None else "n/a"
            print(f"  {pos:<10}  {n:>5}  {wr_str:>7}{flag}")
    print()

    # AvL combined with Consistency
    section("COMBINED: AvL > +10%  AND  Consistency < 65%")
    combo = tackles[(tackles["_avl"] > 10) & (tackles["_cons"] < 65)]
    wr, n = win_rate(combo)
    print(f"  n={n}  WR={wr}%\n")

    section("COMBINED: AvL > +10%  AND  Consistency >= 65%")
    combo2 = tackles[(tackles["_avl"] > 10) & (tackles["_cons"] >= 65)]
    wr2, n2 = win_rate(combo2)
    print(f"  n={n2}  WR={wr2}%\n")


if __name__ == "__main__":
    main()
