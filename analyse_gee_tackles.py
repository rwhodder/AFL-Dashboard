"""
analyse_gee_tackles.py
──────────────────────
Tests the theory that Geelong's narrow home ground inflates tackle counts,
causing unders lines to lose more often when GEE is the opponent.

Compares tackle W/L results for GEE-opponent games vs all other opponents,
broken down by position, and shows avg Line vs avg Actual for GEE games.

Usage:
    python analyse_gee_tackles.py
"""

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

GOOGLE_SHEET_ID         = "10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"
GOOGLE_SHEET_TAB        = "Bet Log"

# Rounds where GEE was an opponent (home or away — we'll verify from the data)
GEE_ROUNDS_2025 = {1, 4, 9, 11, 13, 15, 17, 19, 21, 22}
GEE_ROUNDS_2026 = {1, 3, 6}


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
    print(f"  {len(df)} rows loaded\n")

    # Keep only Tackle rows with a resolved W/L
    df = df[df["Type"].astype(str).str.strip() == "Tackle"]
    df = df[df["W/L"].astype(str).str.strip().isin(["1", "-1", "0", "1.0", "-1.0", "0.0"])]
    df["W/L"]    = pd.to_numeric(df["W/L"], errors="coerce")
    df["win"]    = (df["W/L"] == 1).astype(int)
    df["Year"]   = pd.to_numeric(df["Year"],  errors="coerce")
    df["Round"]  = pd.to_numeric(df["Round"], errors="coerce")
    df["Line"]   = pd.to_numeric(df["Line"],  errors="coerce")
    df["Actual"] = pd.to_numeric(df["Actual"],errors="coerce")
    df["Opponent"] = df["Opponent"].astype(str).str.strip().str.upper()
    df["Team"]     = df["Team"].astype(str).str.strip().str.upper()
    df["Position"] = df["Position"].astype(str).str.strip()

    print(f"Resolved Tackle rows: {len(df)}\n")

    # ── Flag any row involving GEE (as opponent OR as team) in specified rounds ─
    round_mask = (
        ((df["Year"] == 2025) & df["Round"].isin(GEE_ROUNDS_2025)) |
        ((df["Year"] == 2026) & df["Round"].isin(GEE_ROUNDS_2026))
    )
    mask_gee_opp  = (df["Opponent"] == "GEE") & round_mask
    mask_gee_team = (df["Team"]     == "GEE") & round_mask
    df["is_gee_opp"]  = mask_gee_opp
    df["is_gee_team"] = mask_gee_team
    df["is_gee"]      = mask_gee_opp | mask_gee_team

    gee      = df[df["is_gee"]]
    gee_opp  = df[df["is_gee_opp"]]   # players facing GEE
    gee_team = df[df["is_gee_team"]]  # GEE players themselves
    rest     = df[~df["is_gee"]]

    print("=" * 60)
    print("OVERALL TACKLE WR — GEE involvement vs all other games")
    print("=" * 60)

    def wr_str(sub):
        n  = len(sub)
        wr = sub["win"].mean() * 100 if n > 0 else 0
        return f"{wr:.1f}%  (n={n})"

    print(f"  GEE any involvement: {wr_str(gee)}")
    print(f"    → Opponent = GEE:  {wr_str(gee_opp)}  (non-GEE players vs GEE)")
    print(f"    → Team = GEE:      {wr_str(gee_team)}  (GEE players themselves)")
    print(f"  All other games:     {wr_str(rest)}")
    print(f"  Overall:             {wr_str(df)}")

    # ── By year ──────────────────────────────────────────────────────────────
    print("\n── GEE by Year ──")
    for yr in sorted(gee["Year"].dropna().unique()):
        sub_all  = gee[gee["Year"] == yr]
        sub_opp  = gee_opp[gee_opp["Year"] == yr]
        sub_team = gee_team[gee_team["Year"] == yr]
        print(f"  {int(yr)}: all={wr_str(sub_all)}  opp={wr_str(sub_opp)}  team={wr_str(sub_team)}")

    # ── By round (GEE games only) ─────────────────────────────────────────────
    print("\n── GEE Tackle WR by Round ──")
    print(f"  {'':12}  {'Combined':>18}  {'Opp=GEE':>18}  {'Team=GEE':>18}")
    for (yr, rnd), sub in gee.groupby(["Year", "Round"]):
        sub_o = gee_opp[(gee_opp["Year"] == yr) & (gee_opp["Round"] == rnd)]
        sub_t = gee_team[(gee_team["Year"] == yr) & (gee_team["Round"] == rnd)]
        print(f"  {int(yr)} R{int(rnd):>2}:      {wr_str(sub):>18}  {wr_str(sub_o):>18}  {wr_str(sub_t):>18}")

    # ── By position ──────────────────────────────────────────────────────────
    print("\n── GEE Tackle WR by Position (vs all-other baseline) ──")
    print(f"  {'Position':<10}  {'GEE (all)':>18}  {'Opp=GEE':>18}  {'Team=GEE':>18}  {'Others':>18}")
    positions = sorted(df["Position"].unique())
    for pos in positions:
        g  = gee[gee["Position"] == pos]
        go = gee_opp[gee_opp["Position"] == pos]
        gt = gee_team[gee_team["Position"] == pos]
        r  = rest[rest["Position"] == pos]
        if len(g) == 0:
            continue
        print(f"  {pos:<10}  {wr_str(g):>18}  {wr_str(go):>18}  {wr_str(gt):>18}  {wr_str(r):>18}")

    # ── Line vs Actual (GEE only, where both available) ──────────────────────
    gee_with_actual = gee.dropna(subset=["Line", "Actual"])
    opp_with_actual  = gee_opp.dropna(subset=["Line", "Actual"])
    team_with_actual = gee_team.dropna(subset=["Line", "Actual"])
    rest_with_actual = rest.dropna(subset=["Line", "Actual"])

    if not gee_with_actual.empty:
        def line_summary(sub, label):
            if sub.empty:
                return
            avg_over = (sub["Actual"] - sub["Line"]).mean()
            print(f"  {label:<22} avg line={sub['Line'].mean():.2f}  avg actual={sub['Actual'].mean():.2f}  avg over={avg_over:+.2f}  ({'OVERS' if avg_over > 0 else 'unders'} bias)")

        print(f"\n── GEE Line vs Actual ──")
        line_summary(gee_with_actual,  f"GEE all      (n={len(gee_with_actual)})")
        line_summary(opp_with_actual,  f"Opp=GEE      (n={len(opp_with_actual)})")
        line_summary(team_with_actual, f"Team=GEE     (n={len(team_with_actual)})")
        line_summary(rest_with_actual, f"All others   (n={len(rest_with_actual)})")

    # ── DvP breakdown for GEE games ───────────────────────────────────────────
    print("\n── GEE Tackle WR by DvP ──")
    for dvp, sub in gee.groupby("DvP"):
        print(f"  {dvp:<30} {wr_str(sub)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
