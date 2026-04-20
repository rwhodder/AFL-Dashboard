"""
line_alerts.py
──────────────
Checks Dabble for new flagged bets and sends an email alert when found.
Run manually or schedule via Windows Task Scheduler every few hours Thu–Sun.

Setup:
  1. Fill in EMAIL_FROM, EMAIL_TO, EMAIL_APP_PASSWORD below
  2. Gmail: enable 2FA → Google Account → Security → App Passwords → create one
     (search "Google App Passwords" — paste the 16-char code as EMAIL_APP_PASSWORD)
  3. Run once manually to confirm the email arrives:
         python line_alerts.py
  4. Schedule via Task Scheduler (see SCHEDULING section at bottom of file)
"""

import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import pandas as pd

from dabble_scraper import get_pickem_data_for_dashboard, normalize_player_name
from fixture_scraper import scrape_next_round_fixture
from data_processor import load_and_prepare_data

# ── Email config ──────────────────────────────────────────────────────────────
EMAIL_FROM         = "ryanhodder4421@gmail.com"
EMAIL_TO           = "ryan_hodder1@hotmail.com"
EMAIL_APP_PASSWORD = "tkzx warb srat hzcu"   # Gmail App Password (16 chars)
# ─────────────────────────────────────────────────────────────────────────────

# ── Push notification config (ntfy.sh) ───────────────────────────────────────
# 1. Install the free "ntfy" app on your phone (iOS or Android)
# 2. Choose a unique topic name below (keep it unguessable — it's your "channel")
# 3. In the app: tap + → enter your topic name → subscribe
# 4. That's it — no account needed
NTFY_TOPIC = "afl-bets-ryan4421"   # change to something unique/private
# ─────────────────────────────────────────────────────────────────────────────

CACHE_FILE = "seen_bets_cache.json"
STATS_CSV  = "afl_player_stats.csv"

POSITION_MAP = {
    "KeyF": ["FF", "CHF"],
    "GenF": ["HFFR", "HFFL", "FPL", "FPR"],
    "Ruck": ["RK"],
    "InsM": ["C", "RR", "R"],
    "Wing": ["WL", "WR"],
    "GenD": ["HBFL", "HBFR", "BPL", "BPR"],
    "KeyD": ["CHB", "FB"],
}

TEAM_NAME_MAP = {
    "ADE": "Adelaide Crows",  "BRL": "Brisbane Lions",  "CAR": "Carlton",
    "COL": "Collingwood",     "ESS": "Essendon",        "FRE": "Fremantle",
    "GCS": "Gold Coast SUNS", "GEE": "Geelong Cats",    "GWS": "GWS GIANTS",
    "HAW": "Hawthorn",        "MEL": "Melbourne",        "NTH": "North Melbourne",
    "PTA": "Port Adelaide",   "RIC": "Richmond",         "STK": "St Kilda",
    "SYD": "Sydney Swans",    "WBD": "Western Bulldogs", "WCE": "West Coast Eagles",
}

LONG_DISTANCE_PAIRS = {
    ("WCE","MEL"),("WCE","GEE"),("WCE","COL"),("WCE","HAW"),("WCE","CAR"),
    ("WCE","ESS"),("WCE","NTH"),("WCE","RIC"),("WCE","STK"),("WCE","WBD"),
    ("FRE","MEL"),("FRE","GEE"),("FRE","COL"),("FRE","HAW"),("FRE","CAR"),
    ("FRE","ESS"),("FRE","NTH"),("FRE","RIC"),("FRE","STK"),("FRE","WBD"),
    ("WCE","SYD"),("WCE","GWS"),("WCE","BRL"),("WCE","GCS"),
    ("FRE","SYD"),("FRE","GWS"),("FRE","BRL"),("FRE","GCS"),
    ("ADE","WCE"),("ADE","FRE"),("PTA","WCE"),("PTA","FRE"),
    ("WCE","ADE"),("FRE","ADE"),("WCE","PTA"),("FRE","PTA"),
    ("BRL","MEL"),("BRL","GEE"),("BRL","COL"),("BRL","HAW"),("BRL","CAR"),
    ("BRL","ESS"),("BRL","NTH"),("BRL","RIC"),("BRL","STK"),("BRL","WBD"),
    ("GCS","MEL"),("GCS","GEE"),("GCS","COL"),("GCS","HAW"),("GCS","CAR"),
    ("GCS","ESS"),("GCS","NTH"),("GCS","RIC"),("GCS","STK"),("GCS","WBD"),
    ("BRL","ADE"),("BRL","PTA"),("GCS","ADE"),("GCS","PTA"),
    ("SYD","WCE"),("SYD","FRE"),("GWS","WCE"),("GWS","FRE"),
    ("BRL","WCE"),("BRL","FRE"),("GCS","WCE"),("GCS","FRE"),
    ("MEL","WCE"),("GEE","WCE"),("COL","WCE"),("HAW","WCE"),("CAR","WCE"),
    ("ESS","WCE"),("NTH","WCE"),("RIC","WCE"),("STK","WCE"),("WBD","WCE"),
    ("MEL","FRE"),("GEE","FRE"),("COL","FRE"),("HAW","FRE"),("CAR","FRE"),
    ("ESS","FRE"),("NTH","FRE"),("RIC","FRE"),("STK","FRE"),("WBD","FRE"),
    ("MEL","BRL"),("GEE","BRL"),("COL","BRL"),("HAW","BRL"),("CAR","BRL"),
    ("ESS","BRL"),("NTH","BRL"),("RIC","BRL"),("STK","BRL"),("WBD","BRL"),
    ("MEL","GCS"),("GEE","GCS"),("COL","GCS"),("HAW","GCS"),("CAR","GCS"),
    ("ESS","GCS"),("NTH","GCS"),("RIC","GCS"),("STK","GCS"),("WBD","GCS"),
    ("ADE","BRL"),("ADE","GCS"),("PTA","BRL"),("PTA","GCS"),
}

DVP_MAP = {
    "Strong Unders":   "🔴 Strong Unders",
    "Moderate Unders": "🟠 Moderate Unders",
    "Slight Unders":   "🟡 Slight Unders",
    "Strong Easy":     "🔵 Strong Easy",
    "Moderate Easy":   "🔷 Moderate Easy",
    "Slight Easy":     "🔹 Slight Easy",
}


# ── Cache helpers ─────────────────────────────────────────────────────────────

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return set(tuple(x) for x in json.load(f))
    return set()


def save_cache(cache: set):
    with open(CACHE_FILE, "w") as f:
        json.dump([list(x) for x in cache], f)


# ── Bet flag logic (mirrors app.py calculate_bet_flag) ───────────────────────

TACKLE_BAD_OPPONENTS = {"SYD", "GCS", "GEE"}


def calculate_bet_flag(player_row, stat_type):
    try:
        position = player_row.get("Position", player_row.get("position", ""))
        dvp      = player_row.get("DvP",      player_row.get("dvp",      ""))
        travel   = player_row.get("Travel Fatigue", player_row.get("travel_fatigue", ""))
        opponent = player_row.get("Opponent", player_row.get("opponent", ""))
        line_str = player_row.get("Line", "")

        if not line_str:
            return {"priority": "", "description": ""}
        try:
            line_value = float(line_str)
        except (ValueError, TypeError):
            return {"priority": "", "description": ""}

        has_strong_unders = "Strong Unders" in dvp
        has_slight_unders = "Slight Unders" in dvp
        has_slight_easy   = "Slight Easy"   in dvp
        has_moderate_easy = "Moderate Easy" in dvp
        has_strong_easy   = "Strong Easy"   in dvp

        # Kill switch 1: Short Break
        if "Short Break" in travel:
            return {"priority": "", "description": ""}

        # ── Tackle strategies ──────────────────────────────────────────────────
        if stat_type == "tackles":
            # Kill switch 2: bad tackle opponents
            if opponent in TACKLE_BAD_OPPONENTS:
                return {"priority": "", "description": ""}

            if has_strong_unders:
                return {"priority": "S1",
                        "description": "Tackle + Strong Unders DvP → 76.7% WR (n=46) CONFIRMED"}

            if has_slight_easy or has_moderate_easy:
                return {"priority": "S2",
                        "description": "Tackle + Slight/Moderate Easy DvP → 67.5% WR (n=163) CONFIRMED"}

            if position == "InsM" and not has_strong_easy:
                return {"priority": "S3",
                        "description": "Tackle + InsM → 63.5% WR (n=220+) CONFIRMED"}

            if position in ("Wing", "Ruck"):
                return {"priority": "D1",
                        "description": "Tackle + Wing/Ruck → 69.9% WR (n=78) DEVELOPING"}

        # ── Disposal strategies ────────────────────────────────────────────────
        elif stat_type == "disposals":
            if position == "SmF" and (has_slight_unders or has_slight_easy):
                return {"priority": "D2",
                        "description": "Disposal + SmF + Slight Unders/Easy DvP → 69.0% WR (n=90) DEVELOPING"}

        # ── Mark strategies (Watch only) ───────────────────────────────────────
        elif stat_type == "marks":
            if position == "SmF":
                return {"priority": "W1",
                        "description": "Mark + SmF → 92.3% WR (n=16, 2025 only) WATCH"}
            if line_value >= 8.0:
                return {"priority": "W2",
                        "description": "Mark + Line ≥8.0 → 86.7% WR (n=16, 2025 only) WATCH"}

        return {"priority": "", "description": ""}

    except Exception:
        return {"priority": "", "description": ""}


# ── DvP calculation ───────────────────────────────────────────────────────────

def build_simplified_dvp(stat_type):
    try:
        proc = load_and_prepare_data(STATS_CSV)
        if stat_type not in proc.columns:
            if stat_type == "disposals" and "kicks" in proc.columns:
                proc[stat_type] = proc["kicks"] + proc["handballs"]
            else:
                proc[stat_type] = 0

        if "opponentTeam" not in proc.columns and "opponent" in proc.columns:
            proc["opponentTeam"] = proc["opponent"]

        role_averages = {
            role: proc[proc["role"] == role][stat_type].mean()
            for role in proc["role"].unique()
            if not proc[proc["role"] == role].empty
        }

        simplified = {}
        for team in proc["opponentTeam"].unique():
            simplified[team] = {}
            for role in proc["role"].unique():
                subset = proc[(proc["opponentTeam"] == team) & (proc["role"] == role)]
                if subset.empty or role not in role_averages:
                    continue
                dvp = subset[stat_type].mean() - role_averages[role]
                st, mt, slt = (2.0, 1.0, 0.1) if stat_type == "disposals" else (1.0, 0.5, 0.05)
                if   dvp >=  st:  strength = "Strong Easy"
                elif dvp >=  mt:  strength = "Moderate Easy"
                elif dvp >=  slt: strength = "Slight Easy"
                elif dvp <= -st:  strength = "Strong Unders"
                elif dvp <= -mt:  strength = "Moderate Unders"
                elif dvp <= -slt: strength = "Slight Unders"
                else:             strength = "Neutral"
                if strength != "Neutral":
                    simplified[team][role] = {"dvp": dvp, "strength": strength}
        return simplified
    except Exception as e:
        print(f"DvP build error: {e}")
        return {}


# ── Build current flagged bets ────────────────────────────────────────────────

def get_current_flagged_bets():
    """Returns a list of dicts for every flagged bet found this check."""
    flagged = []

    # Fixtures → home/away lookup
    fixtures = scrape_next_round_fixture()
    home_away = {}
    team_opponents = {}
    for match in fixtures:
        try:
            home_full, away_full = match["match"].split(" vs ")
            ha = ab = None
            for abbr, name in TEAM_NAME_MAP.items():
                if name == home_full: ha = abbr
                if name == away_full: ab = abbr
            if ha and ab:
                home_away[ha] = {"opponent": ab, "is_home": True}
                home_away[ab] = {"opponent": ha, "is_home": False}
            team_opponents[home_full] = away_full
            team_opponents[away_full] = home_full
        except Exception:
            pass

    # Load player stats
    try:
        stats_df = pd.read_csv(STATS_CSV, skiprows=3).fillna(0)
        latest   = stats_df["round"].max()
        recent   = stats_df[stats_df["round"].isin([latest, latest - 1])]
        players  = (recent.sort_values("round", ascending=False)
                    .groupby(["player", "team"]).first().reset_index())
    except Exception as e:
        print(f"Stats load error: {e}")
        return []

    # Map positions from player_positions.csv (source of truth)
    PLAYER_POSITIONS_FILE = "player_positions.csv"
    if os.path.exists(PLAYER_POSITIONS_FILE):
        pos_df     = pd.read_csv(PLAYER_POSITIONS_FILE)
        pos_lookup = dict(zip(pos_df["player"].str.lower().str.strip(), pos_df["position"]))
    else:
        pos_lookup = {}

    def map_pos_csv(name):
        return pos_lookup.get(str(name).lower().strip(), "Unknown")

    players["position"] = players["player"].apply(map_pos_csv)

    # Travel fatigue
    def get_travel(team):
        info    = home_away.get(team, {})
        opp     = info.get("opponent")
        is_home = info.get("is_home", True)
        if not is_home and opp and (team, opp) in LONG_DISTANCE_PAIRS:
            return "🟠 Moderate (Long Travel)"
        return "✅ Neutral"

    players["travel_fatigue"] = players["team"].apply(get_travel)

    # Opponent
    def get_opp(team):
        full = TEAM_NAME_MAP.get(team, team)
        opp  = team_opponents.get(full, "Unknown")
        for a, n in TEAM_NAME_MAP.items():
            if n == opp:
                return a
        return opp

    players["opponent"] = players["team"].apply(get_opp)

    for stat_type in ["disposals", "marks", "tackles"]:
        # Pickem lines
        try:
            pickem = get_pickem_data_for_dashboard(stat_type)
        except Exception:
            pickem = {}

        if not pickem:
            continue

        # DvP
        sdvp = build_simplified_dvp(stat_type)

        # Avg vs line analysis from stats CSV
        try:
            raw_stats = pd.read_csv(STATS_CSV, skiprows=3).fillna(0)
            if stat_type not in raw_stats.columns and stat_type == "disposals":
                raw_stats[stat_type] = raw_stats["kicks"] + raw_stats["handballs"]
        except Exception:
            raw_stats = None

        for _, p in players.iterrows():
            player_name = p["player"]
            team        = p["team"]
            position    = p["position"]
            opponent    = p["opponent"]
            travel      = p["travel_fatigue"]

            # Match line
            line_val = pickem.get(player_name, "")
            if not line_val:
                for pp, lv in pickem.items():
                    if pp.lower() == player_name.lower():
                        line_val = lv
                        break
            if not line_val:
                continue

            # DvP — SmF/MedF/FwdMid map back to GenF for the DvP lookup
            DVP_POS_NORM = {"SmF": "GenF", "MedF": "GenF", "FwdMid": "GenF"}
            dvp_pos = DVP_POS_NORM.get(position, position)
            if opponent and opponent != "Unknown" and dvp_pos != "Unknown":
                if opponent in sdvp and dvp_pos in sdvp[opponent]:
                    dvp_str = DVP_MAP.get(sdvp[opponent][dvp_pos]["strength"], "✅ Neutral")
                else:
                    dvp_str = "✅ Neutral"
            else:
                dvp_str = "⚠️ Unknown"

            # Avg vs Line + Line Consistency
            avg_vs_line = ""
            line_consistency = ""
            if raw_stats is not None and stat_type in raw_stats.columns:
                ps = raw_stats[raw_stats["player"].str.lower() == player_name.lower()]
                if not ps.empty:
                    avg = ps[stat_type].mean()
                    if avg > 0:
                        avg_vs_line = f"{((float(line_val) / avg) - 1) * 100:+.1f}%"
                    below = sum(1 for v in ps[stat_type].values if v < float(line_val))
                    line_consistency = f"{(below / len(ps)) * 100:.1f}%"

            row = {
                "Player":           player_name,
                "Team":             team,
                "Opponent":         opponent,
                "Position":         position,
                "Travel Fatigue":   travel,
                "DvP":              dvp_str,
                "Line":             str(line_val),
                "Avg vs Line":      avg_vs_line,
                "Line Consistency": line_consistency,
            }

            result = calculate_bet_flag(row, stat_type)
            if result["priority"]:
                flagged.append({
                    **row,
                    "Stat":        stat_type.title(),
                    "Bet Priority": result["priority"],
                })

    return flagged


# ── Push notification (ntfy.sh) ──────────────────────────────────────────────

def send_push_notification(new_bets: list):
    import urllib.request
    import urllib.error

    count   = len(new_bets)
    title   = f"🏉 {count} new AFL bet{'s' if count > 1 else ''} flagged"
    lines   = []
    for b in new_bets:
        lines.append(
            f"{b['Bet Priority']} {b['Stat']} — {b['Player']} "
            f"({b['Team']} vs {b['Opponent']}) Line {b['Line']}"
        )
    body = "\n".join(lines)

    # Urgency: S1 = urgent, S2/S3 = high, D = default
    priorities = [b["Bet Priority"] for b in new_bets]
    ntfy_priority = ("urgent" if "S1" in priorities
                     else "high" if any(p in priorities for p in ("S2","S3"))
                     else "default")

    req = urllib.request.Request(
        f"https://ntfy.sh/{NTFY_TOPIC}",
        data=body.encode("utf-8"),
        headers={
            "Title":    title.encode("utf-8"),
            "Priority": ntfy_priority,
            "Tags":     "football,bell",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        print(f"✅ Push notification sent to ntfy topic '{NTFY_TOPIC}'")
    except urllib.error.URLError as e:
        print(f"❌ Push notification failed: {e}")


# ── Email sender ──────────────────────────────────────────────────────────────

def send_alert_email(new_bets: list):
    subject = f"🏉 AFL Bet Alert — {len(new_bets)} new flagged bet(s)"

    # Plain text fallback
    lines = [f"New flagged bets found at {datetime.now().strftime('%H:%M %d/%m/%Y')}\n"]
    for b in new_bets:
        lines.append(
            f"  {b['Bet Priority']} | {b['Stat']:>9} | {b['Player']} ({b['Team']} vs {b['Opponent']}) "
            f"| Line: {b['Line']} | {b['DvP']}"
        )
    text_body = "\n".join(lines)

    # HTML body
    rows_html = ""
    priority_colours = {
        "S1": "#198754", "S2": "#198754", "S3": "#198754",
        "D1": "#fd7e14", "D2": "#fd7e14",
        "W1": "#6c757d", "W2": "#6c757d",
    }
    for b in new_bets:
        colour = priority_colours.get(b["Bet Priority"], "#6c757d")
        rows_html += f"""
        <tr>
          <td style="background:{colour};color:white;
                     font-weight:bold;padding:6px 10px;text-align:center">
            {b['Bet Priority']}
          </td>
          <td style="padding:6px 10px">{b['Stat']}</td>
          <td style="padding:6px 10px"><strong>{b['Player']}</strong></td>
          <td style="padding:6px 10px">{b['Team']} vs {b['Opponent']}</td>
          <td style="padding:6px 10px">{b['Position']}</td>
          <td style="padding:6px 10px">{b['Line']}</td>
          <td style="padding:6px 10px">{b['Avg vs Line']}</td>
          <td style="padding:6px 10px">{b['DvP']}</td>
        </tr>"""

    html_body = f"""
    <html><body style="font-family:Arial,sans-serif;font-size:14px">
      <h2 style="color:#343a40">🏉 AFL Bet Alert — {len(new_bets)} new flagged bet(s)</h2>
      <p style="color:#6c757d">{datetime.now().strftime('%H:%M  %d/%m/%Y')}</p>
      <table border="0" cellspacing="0" cellpadding="0"
             style="border-collapse:collapse;width:100%;margin-top:12px">
        <thead>
          <tr style="background:#343a40;color:white">
            <th style="padding:8px 10px">Priority</th>
            <th style="padding:8px 10px">Stat</th>
            <th style="padding:8px 10px">Player</th>
            <th style="padding:8px 10px">Matchup</th>
            <th style="padding:8px 10px">Position</th>
            <th style="padding:8px 10px">Line</th>
            <th style="padding:8px 10px">Avg vs Line</th>
            <th style="padding:8px 10px">DvP</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
      <p style="color:#6c757d;margin-top:20px;font-size:12px">
        Sent by AFL Dashboard line_alerts.py
      </p>
    </body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_FROM
    msg["To"]      = EMAIL_TO
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

    print(f"✅ Alert email sent to {EMAIL_TO}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_check():
    print(f"\n🔍 Checking for new flagged bets — {datetime.now().strftime('%H:%M %d/%m/%Y')}")

    cache    = load_cache()
    current  = get_current_flagged_bets()

    new_bets = []
    for bet in current:
        key = (bet["Stat"], bet["Player"], bet["Team"], bet["Line"])
        if key not in cache:
            new_bets.append(bet)
            cache.add(key)

    if not new_bets:
        print("✅ No new flagged bets found.")
        save_cache(cache)
        return

    # Separate active bets (S/D) from watch bets (W) for alerting
    active_bets = [b for b in new_bets if b["Bet Priority"] not in ("W1", "W2")]
    watch_bets  = [b for b in new_bets if b["Bet Priority"] in ("W1", "W2")]

    print(f"🚨 {len(new_bets)} new bet(s) found:")
    for b in new_bets:
        print(f"   {b['Bet Priority']} | {b['Stat']:>9} | {b['Player']} | Line {b['Line']}")
    if watch_bets:
        print(f"   👁 {len(watch_bets)} watch-only bet(s) — logged but no alert sent")

    if active_bets:
        try:
            send_alert_email(active_bets)
        except Exception as e:
            print(f"❌ Email failed: {e}")
            print("   Check EMAIL_FROM / EMAIL_TO / EMAIL_APP_PASSWORD at top of file.")

        try:
            send_push_notification(active_bets)
        except Exception as e:
            print(f"❌ Push notification failed: {e}")
            print(f"   Check NTFY_TOPIC at top of file.")
    else:
        print("ℹ️  No active bets (S/D strategies) — no email or push sent.")

    save_cache(cache)


if __name__ == "__main__":
    run_check()


# ── SCHEDULING INSTRUCTIONS ───────────────────────────────────────────────────
#
# Windows Task Scheduler — run every 2 hours Thu–Sun during AFL season:
#
# 1. Open Task Scheduler → Create Basic Task
# 2. Name: "AFL Line Alerts"
# 3. Trigger: Weekly → check Thu, Fri, Sat, Sun
# 4. Action: Start a program
#    Program/script:  C:\Users\ryan_\Documents\Python\Betting Dabble\AFL Dashboard\venv\Scripts\python.exe
#    Arguments:       line_alerts.py
#    Start in:        C:\Users\ryan_\Documents\Python\Betting Dabble\AFL Dashboard
# 5. After creating, right-click the task → Properties → Triggers → Edit
#    → Repeat task every: 2 hours  for a duration of: 1 day
# 6. Click OK
#
# To clear the cache (force re-check all bets):
#    del seen_bets_cache.json
#
# ─────────────────────────────────────────────────────────────────────────────
