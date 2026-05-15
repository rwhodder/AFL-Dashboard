import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from scipy import stats as scipy_stats
import dash
from dash import html, dcc, dash_table, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json

# Import from existing modules
from fixture_scraper import scrape_next_round_fixture
from travel_fatigue import build_travel_log
from stadium_locations import STADIUM_COORDS
from data_processor import load_and_prepare_data
from dabble_scraper import get_pickem_data_for_dashboard, normalize_player_name

# ===== CONSTANTS =====
import os as _os_env
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()
OPENWEATHER_API_KEY = _os_env.environ.get("OPENWEATHER_API_KEY", "")

# ── Google Sheets config ───────────────────────────────────────────────────────
# 1. Go to https://console.cloud.google.com
# 2. Create a project → enable "Google Sheets API" + "Google Drive API"
# 3. Create credentials → Service Account → download JSON → save as
#    "google_credentials.json" in the same folder as app.py
# 4. Open your Google Sheet → Share → add the service account email as Editor
GOOGLE_SHEET_ID          = "10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w"
GOOGLE_CREDENTIALS_FILE  = "google_credentials.json"
GOOGLE_SHEET_TAB         = "Bet Log"   # change if your tab has a different name
ROUND_CACHE_FILE         = "round_cache.json"   # local cache of scraped lines per round
# ──────────────────────────────────────────────────────────────────────────────

POSITION_MAP = {
    "KeyF":   ["FF", "CHF"],
    "GenF":   ["HFFR", "HFFL", "FPL", "FPR"],
    "Ruck":   ["RK"],
    "InsM":   ["C", "RR", "R"],
    "Wing":   ["WL", "WR"],
    "GenD":   ["HBFL", "HBFR", "BPL", "BPR"],
    "KeyD":   ["CHB", "FB"],
    # New sub-groups (only via player_positions.csv — no teamsheet tag maps to these)
    "SmF":    [],
    "MedF":   [],
    "FwdMid": [],
}

# Inverted: teamsheet tag → position group (used when auto-adding new players)
POS_TAG_LOOKUP = {tag: group for group, tags in POSITION_MAP.items() for tag in tags}

PLAYER_POSITIONS_FILE = "player_positions.csv"

# Grounds where congestion/dimensions inflate tackle counts — kill switch for tackle unders
# Add any new narrow/congested grounds here.
NARROW_GROUNDS_NO_TACKLE = {"GMHBA Stadium", "Kardinia Park"}


def _load_gee_home_rounds(fixture_file="afl-2026-fixture.csv"):
    """Rounds where GEE plays at home at GMHBA — used to filter history and live rows."""
    import csv
    rounds = set()
    try:
        with open(fixture_file, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                home     = row.get('Home Team', '')
                location = row.get('Location', '')
                rnd      = row.get('Round Number', '')
                if 'Geelong' in home and ('GMHBA' in location or 'Kardinia' in location):
                    try:
                        rounds.add((2026, int(rnd)))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return rounds

GEE_HOME_ROUNDS = _load_gee_home_rounds()


def _passes_bet_filters(row) -> bool:
    """
    Hard-filter logic — applied to both bet history (WR computation) and live _wr_qualifies.
    Works with both dicts and pandas Series.

    Hard avoids (confirmed by LC-conditional cross-tab analysis, Apr 2026):
      - Line < 3                    (29% at LC 50-60%, independent of LC band)
      - Line >= 6                   (54% post-filters; 53% 6.0-6.9, 56% 7.0+; bad across all LC bands)
      - MedF position               (24% at LC<40% where most MedF bets sit)
      - GCS as opponent             (33% total, 13% at LC<50%)
      - GEE home round              (10% at LC 50-60% as opponent, GMHBA congestion)
      - AvL > +10%                  (46.6% WR, n=266)
      - Rain weather                (50.0% WR, n=34 — wet ball suppresses tackle counts)
      - LC < 50%                    (sub-T2 threshold — no qualifying tier, hard avoid)

    NOT here (handled elsewhere):
      - Neutral DvP                 → T2-only skip in calculate_bet_flag() (fine at T0/T1)
      - ADE team block              → T2-only skip in calculate_bet_flag()

    Removed (cross-tab showed no independent signal above LC):
      - Last 5 > Line               (identical or worse within each LC band)
      - FwdMid position             (57% overall, no consistent within-band weakness)
    """
    def get(key, fallback=''):
        try:
            v = row[key] if not hasattr(row, 'get') else row.get(key, fallback)
            return '' if v is None else str(v).strip()
        except (KeyError, TypeError):
            return fallback

    position = get('Position')
    opponent = get('Opponent')
    team     = get('Team')

    try:
        line_val = float(get('Line'))
    except (ValueError, TypeError):
        return False

    if line_val < 3:                           return False
    if line_val >= 6:                          return False
    if opponent in TACKLE_BAD_OPPONENTS:       return False
    if position == 'MedF':                     return False

    # GEE home round — block both teams in that game
    try:
        yr_rnd = (int(get('Year')), int(get('Round')))
        if yr_rnd in GEE_HOME_ROUNDS and (opponent == 'GEE' or team == 'GEE'):
            return False
    except (ValueError, TypeError):
        pass

    # AvL > +10%: player's career avg is 10%+ above the line → hard avoid
    # Values from Sheets are "2000%" format (number in %-cell × 100); live app "+5.2%" text.
    # Detect scale: if |parsed| > 100, it's the Sheets format → divide by 100.
    avl_raw = get('Avg vs Line')
    if avl_raw:
        try:
            avl_num = float(avl_raw.replace('%', '').replace('+', ''))
            if abs(avl_num) > 100:
                avl_num /= 100.0
            if avl_num > 10.0:
                return False
        except (ValueError, TypeError):
            pass

    # Rain: wet ball suppresses tackle counts → hard avoid
    weather = get('Weather')
    if weather and 'rain' in weather.lower():
        return False

    # LC < 50%: below minimum qualifying tier (T2 floor) → hard avoid
    lc_raw = get('Line Consistency')
    if lc_raw:
        try:
            lc_num = float(lc_raw.replace('%', '').strip())
            if lc_num <= 1.0:
                lc_num *= 100
            if lc_num < 50:
                return False
        except (ValueError, TypeError):
            pass

    return True


def load_player_positions(stats_df=None):
    """
    Load the master player position file.
    If stats_df is provided, any player in the CSV not yet in the file
    is auto-added with their teamsheet position + an AUTO review flag.
    Returns dict: lowercase_player_name → position
    """
    import os

    if os.path.exists(PLAYER_POSITIONS_FILE):
        pos_df = pd.read_csv(PLAYER_POSITIONS_FILE)
    else:
        pos_df = pd.DataFrame(columns=["player", "team", "position", "most_common_pos", "notes"])

    if stats_df is not None:
        existing = set(pos_df["player"].str.lower().str.strip())

        non_int = stats_df[stats_df["namedPosition"] != "INT"]
        most_common = (
            non_int.groupby("player")
            .agg(team=("team", "last"),
                 most_common_pos=("namedPosition", lambda x: x.value_counts().index[0]))
            .reset_index()
        )

        new_rows = []
        for _, row in most_common.iterrows():
            if row["player"].lower().strip() not in existing:
                grp = POS_TAG_LOOKUP.get(row["most_common_pos"], "GenF")
                new_rows.append({
                    "player":          row["player"],
                    "team":            row["team"],
                    "position":        grp,
                    "most_common_pos": row["most_common_pos"],
                    "notes":           "AUTO: review position",
                })

        if new_rows:
            pos_df = pd.concat([pos_df, pd.DataFrame(new_rows)], ignore_index=True)
            pos_df.to_csv(PLAYER_POSITIONS_FILE, index=False)
            print(f"Auto-added {len(new_rows)} new player(s) to {PLAYER_POSITIONS_FILE}:")
            for r in new_rows:
                print(f"   + {r['player']} ({r['team']}) → {r['position']}  [AUTO — review needed]")

    return dict(zip(pos_df["player"].str.lower().str.strip(), pos_df["position"]))

TEAM_NAME_MAP = {
    "ADE": "Adelaide Crows",
    "BRL": "Brisbane Lions",
    "CAR": "Carlton",
    "COL": "Collingwood",
    "ESS": "Essendon",
    "FRE": "Fremantle",
    "GCS": "Gold Coast SUNS",
    "GEE": "Geelong Cats",
    "GWS": "GWS GIANTS",
    "HAW": "Hawthorn",
    "MEL": "Melbourne",
    "NTH": "North Melbourne",
    "PTA": "Port Adelaide",
    "RIC": "Richmond",
    "STK": "St Kilda",
    "SYD": "Sydney Swans",
    "WBD": "Western Bulldogs",
    "WCE": "West Coast Eagles",
}

# ===== GOOGLE SHEETS HELPERS =====
def get_sheets_client():
    """Return an authorised gspread client, or None if credentials missing."""
    try:
        import gspread
        import os, json
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        # Cloud: credentials stored as env var GOOGLE_CREDENTIALS_JSON
        creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if creds_json:
            creds = Credentials.from_service_account_info(json.loads(creds_json), scopes=scopes)
        elif os.path.exists(GOOGLE_CREDENTIALS_FILE):
            creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_FILE, scopes=scopes)
        else:
            print(f"WARN: No Google credentials found — Sheets sync disabled.")
            return None

        client = gspread.authorize(creds)
        return client
    except Exception as e:
        print(f"WARN: Sheets auth error: {e}")
        return None


def get_existing_sheet_keys(worksheet):
    """
    Return a set of (Type, Round, Player, Line) tuples already in the sheet.

    Keying on Line (not just Player) means:
    - A player pushed Thursday with 5 markets won't block new markets on Saturday
      for the same player if the line is different.
    - If a bookmaker updates a line between sessions the new line pushes as a
      fresh row (useful for tracking line movement).
    - Same player + same line in the same round is still treated as a duplicate
      and skipped, so no double-ups.
    """
    try:
        records = worksheet.get_all_records()
        keys = set()
        for row in records:
            t = str(row.get("Type",   "")).strip()
            r = str(row.get("Round",  "")).strip()
            p = str(row.get("Player", "")).strip()
            l = str(row.get("Line",   "")).strip()
            if t and r and p:
                keys.add((t, r, p, l))
        return keys
    except Exception as e:
        print(f"WARN: Could not read existing sheet keys: {e}")
        return set()


def push_to_google_sheets(disposals_df, marks_df, tackles_df, current_round):
    """
    Append new bet rows from all three stat-type dataframes to the master sheet.
    All rows with a Line are pushed (not just flagged strategies).
    Rows already present (same Type + Round + Player + Line) are skipped automatically.

    Returns (rows_added, message).
    """
    client = get_sheets_client()
    if client is None:
        return 0, "❌ Google Sheets not configured — see GOOGLE_CREDENTIALS_FILE in app.py"

    try:
        sheet     = client.open_by_key(GOOGLE_SHEET_ID)
        worksheet = sheet.worksheet(GOOGLE_SHEET_TAB)
    except Exception as e:
        return 0, f"❌ Could not open sheet: {e}"

    # Master column order — matches your sheet exactly
    COLUMNS = [
        "Type", "Year", "Round", "Player", "Team", "Opponent", "Position",
        "Travel Fatigue", "Weather", "DvP", "Line", "Avg vs Line",
        "Line Consistency", "Strategy", "Actual", "W/L",
        "Hist WR", "Range", "Confidence",
    ]

    # If sheet is empty write the header row first
    existing_data = worksheet.get_all_values()
    if not existing_data:
        worksheet.append_row(COLUMNS)

    existing_keys = get_existing_sheet_keys(worksheet)

    import datetime
    current_year = datetime.datetime.now().year

    rows_to_add = []

    for stat_label, df in [("Disposal", disposals_df),
                            ("Mark",     marks_df),
                            ("Tackle",   tackles_df)]:
        if df is None or df.empty:
            continue

        for _, row in df.iterrows():
            player   = str(row.get("Player", "")).strip()
            line_val = str(row.get("Line",   "")).strip()
            key      = (stat_label, str(current_round), player, line_val)

            if key in existing_keys:
                continue   # same player + same line already in sheet — skip

            position = str(row.get("Position", ""))
            dvp      = str(row.get("DvP",      ""))
            opponent = str(row.get("Opponent", ""))
            hist_wr  = get_leg_historical_wr(position, dvp, opponent)

            new_row = [
                stat_label,                              # Type
                current_year,                            # Year
                current_round,                           # Round
                player,                                  # Player
                str(row.get("Team",             "")),
                opponent,
                position,
                str(row.get("Travel Fatigue",   "")),
                str(row.get("Weather",          "")),
                dvp,
                line_val,                                # Line
                str(row.get("Avg vs Line",      "")),
                str(row.get("Line Consistency", "")),
                str(row.get("Bet Priority",     "")),    # Strategy: T1 / T2 / blank
                "",                                      # Actual — filled by update_results.py
                "",                                      # W/L    — filled by update_results.py
                hist_wr,                                 # Hist WR  e.g. "67% (23)"
                _wr_range_str(hist_wr),                  # WR Range e.g. "61-73%"
                _confidence_str(hist_wr),                # Confidence: High/Med/Low/—
            ]
            rows_to_add.append(new_row)
            existing_keys.add(key)   # prevent same-session dupes

    if not rows_to_add:
        return 0, "✅ Nothing new to push — all flagged bets already in sheet."

    # Batch append for speed
    worksheet.append_rows(rows_to_add, value_input_option="USER_ENTERED")
    return len(rows_to_add), f"✅ {len(rows_to_add)} new bet row(s) pushed to Google Sheets."


def save_round_cache(round_num, disposals_df, marks_df, tackles_df):
    """
    Persist scraped lines for a round to a local JSON file.
    Called every time lines are scraped so data survives past when Dabble
    removes the round's markets.
    """
    try:
        cache = {}
        if os.path.exists(ROUND_CACHE_FILE):
            with open(ROUND_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)

        key = str(round_num)
        cache[key] = {}
        for stat_type, df in [('disposals', disposals_df),
                               ('marks',     marks_df),
                               ('tackles',   tackles_df)]:
            if df is not None and not df.empty:
                cache[key][stat_type] = df.to_dict(orient='records')

        with open(ROUND_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)

        print(f"Round cache saved — round {round_num} ({sum(len(v) for v in cache[key].values())} rows across stat types).")
    except Exception as e:
        print(f"WARN: Could not save round cache: {e}")


def push_all_cached_rounds():
    """
    Push every round in round_cache.json to Google Sheets.
    Each round goes through the normal deduplication logic so already-pushed
    rows are skipped automatically. Rounds that are fully pushed will produce
    zero new rows — that's fine.

    Returns (total_rows_added, message).
    """
    if not os.path.exists(ROUND_CACHE_FILE):
        return 0, "✅ No cached rounds found — nothing to push."

    try:
        with open(ROUND_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    except Exception as e:
        return 0, f"❌ Could not read round cache: {e}"

    if not cache:
        return 0, "✅ Round cache is empty — nothing to push."

    total_pushed = 0
    round_msgs   = []
    for round_num_str, round_data in sorted(
        ((k, v) for k, v in cache.items() if str(k).lstrip('-').isdigit()),
        key=lambda x: int(x[0])
    ):
        round_num    = int(round_num_str)
        disposals_df = pd.DataFrame(round_data.get('disposals', []))
        marks_df     = pd.DataFrame(round_data.get('marks',     []))
        tackles_df   = pd.DataFrame(round_data.get('tackles',   []))

        n, _ = push_to_google_sheets(disposals_df, marks_df, tackles_df, round_num)
        total_pushed += n
        if n > 0:
            round_msgs.append(f"R{round_num}: {n} new")

    if round_msgs:
        detail = ", ".join(round_msgs)
        return total_pushed, f"✅ {total_pushed} new row(s) pushed from cache ({detail})."
    return 0, "✅ All cached rounds already in sheet — nothing new to push."


# ===== WEATHER FUNCTIONS =====
import requests
from datetime import datetime, timedelta
import pytz

def get_forecast(lat, lon):
    url = (f"https://api.openweathermap.org/data/2.5/forecast"
           f"?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}")
    try:
        res = requests.get(url)
        res.raise_for_status()
        return res.json().get("list", [])
    except Exception as e:
        print(f"Error fetching forecast: {e}")
        return []

def extract_weather_for_datetime(forecast_list, target_datetime):
    closest  = None
    min_diff = timedelta(hours=3)
    for entry in forecast_list:
        dt   = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
        diff = abs(dt - target_datetime.replace(tzinfo=None))
        if diff < min_diff:
            min_diff = diff
            closest  = entry
    if not closest:
        return None
    return {
        "rain":     closest.get("rain",  {}).get("3h", 0.0),
        "wind":     closest.get("wind",  {}).get("speed", 0.0),
        "humidity": closest.get("main",  {}).get("humidity", 0.0),
    }

def categorize_weather_for_stat(weather, stat_type='disposals'):
    rain = float(weather.get('rain', 0)) if weather else 0
    wind = float(weather.get('wind', 0)) if weather else 0

    rain_value = 0 if rain < 1.2 else (1 if rain < 3.0 else (2 if rain <= 6.0 else 3))
    wind_value = 0 if wind < 15  else (1 if wind <= 25 else 2)
    severity   = rain_value + wind_value

    weather_severity = 2 if severity >= 3 else (1 if severity >= 1 else 0)

    if stat_type in ['disposals', 'marks']:
        if weather_severity == 2:
            flag_count, rating_text = 3.0, "Strong"
        elif weather_severity == 1:
            flag_count, rating_text = (2.0 if stat_type == 'marks' else 1.5), "Medium"
        else:
            flag_count, rating_text = 0, "Neutral"
    else:  # tackles — rain effect is already priced into lines, treat as neutral
        flag_count, rating_text = 0, "Neutral"

    factors = []
    if rain_value > 0: factors.append("Rain")
    if wind_value > 0: factors.append("Wind")
    factors_str = ', '.join(factors) if factors else "Clear conditions"

    if flag_count == 0:
        rating = "✅ Neutral"
    elif rating_text == "Medium":
        rating = f"⚠️ Medium Unders Edge ({factors_str}) (excl Tackle)"
    else:
        rating = f"🔴 Strong Unders Edge ({factors_str}) (excl Tackle)"

    return {
        "Rating":    rating,
        "FlagCount": flag_count,
        "RawValues": f"Rain: {rain:.1f}mm, Wind: {wind:.1f}km/h",
    }


def calculate_dvp_for_stat(processed_df, stat_type='disposals'):
    simplified_dvp = {}

    if stat_type not in processed_df.columns:
        if stat_type == 'disposals' and 'kicks' in processed_df.columns and 'handballs' in processed_df.columns:
            processed_df[stat_type] = processed_df['kicks'] + processed_df['handballs']
        else:
            processed_df[stat_type] = 0

    if "opponentTeam" not in processed_df.columns and "opponent" in processed_df.columns:
        processed_df["opponentTeam"] = processed_df["opponent"]

    role_averages = {
        role: processed_df[processed_df['role'] == role][stat_type].mean()
        for role in processed_df['role'].unique()
        if not processed_df[processed_df['role'] == role].empty
    }

    for team in processed_df['opponentTeam'].unique():
        simplified_dvp[team] = {}
        for role in processed_df['role'].unique():
            subset = processed_df[(processed_df['opponentTeam'] == team) & (processed_df['role'] == role)]
            if subset.empty or role not in role_averages:
                continue
            dvp = subset[stat_type].mean() - role_averages[role]

            if stat_type == 'disposals':
                st, mt, slt = 2.0, 1.0, 0.1
            else:
                st, mt, slt = 1.0, 0.5, 0.05

            if   dvp >=  st:  strength = "Strong Easy"
            elif dvp >=  mt:  strength = "Moderate Easy"
            elif dvp >=  slt: strength = "Slight Easy"
            elif dvp <= -st:  strength = "Strong Unders"
            elif dvp <= -mt:  strength = "Moderate Unders"
            elif dvp <= -slt: strength = "Slight Unders"
            else:             strength = "Neutral"

            if strength != "Neutral":
                simplified_dvp[team][role] = {"dvp": dvp, "strength": strength}

    return simplified_dvp


def calculate_score(player_row, team_weather):
    score_value  = 0
    score_factors = []

    travel_fatigue = player_row.get('travel_fatigue', '')
    travel_points  = 0
    travel_details = []

    if '(' in travel_fatigue:
        flags = [f.strip() for f in travel_fatigue.split('(')[1].split(')')[0].split(',')]
        for flag in flags:
            if "Long Travel" in flag:
                travel_points += 2.0
                travel_details.append("Long Travel: +2.0")
            elif "Short Break" in flag:
                pass  # Short Break filter removed — insufficient sample
        travel_points = min(travel_points, 3.0)
        if travel_points > 0:
            score_factors.append(f"Travel: +{travel_points:.1f} ({', '.join(travel_details)})")
            score_value += travel_points

    team          = player_row.get('team', '')
    weather_points = team_weather.get(team, {}).get('FlagCount', 0)
    if weather_points > 0:
        score_factors.append(f"Weather: +{weather_points:.1f}")
    elif weather_points < 0:
        score_factors.append(f"Weather: {weather_points:.1f}")
    score_value += weather_points

    dvp_text = player_row.get('dvp', '')
    dvp_pts  = 4 if 'Strong' in dvp_text else (2 if 'Moderate' in dvp_text else (1 if 'Slight' in dvp_text else -1))
    score_factors.append(f"DvP: {'+' if dvp_pts > 0 else ''}{dvp_pts}")
    score_value += dvp_pts

    return {"ScoreValue": score_value, "Factors": " | ".join(score_factors)}


# =============================================================================
# CORE STRATEGY LOGIC
# Rebuilt from 3,055 bets across 2025+2026 seasons.
# =============================================================================

# Opponents where tackle unders consistently lose (42–47% WR historically)
TACKLE_BAD_OPPONENTS = {'GCS'}  # GEE removed: handled by GMHBA stadium check (home-only); SYD removed: 71.4% WR — coincidental

# Player teams with structural outperformance across all LC bands (LC × Team cross-tab, Apr 2026)
TACKLE_EDGE_TEAMS = {'WCE'}        # 88% at LC 60-70% (n=8), 86% at LC≥70% (n=7), 69% at LC<50% (n=26)

# Player teams where T2 (LC 50-60%) specifically underperforms — block T2 only, T0/T1 still valid
TACKLE_AVOID_T2_TEAMS = {'ADE'}    # 12% at LC 50-60% (n=8); LC≥60%: 100% n=4, 67% n=3 — fine above


def calculate_bet_flag(player_row, stat_type='disposals'):
    """
    Tackle-unders strategy — LC-driven tiering (Apr 2026 cross-tab analysis).

    HARD AVOIDS (any true → skip):
      - Line < 3                (48.6% WR)
      - Line >= 6               (54% post-filters; bad across all LC bands)
      - MedF position           (24% at LC<40% where most MedF bets sit)
      - GCS opponent            (36.4% WR, n=22)
      - GEE home round          (GMHBA narrow ground)
      - GMHBA stadium
      - AvL > +10%              (46.6% WR, n=266)
      - Rain weather            (50.0% WR, n=34)

    T2-ONLY SKIPS (inside lc_50 branch):
      - Neutral DvP             (52% at LC 50-60%; 100% at LC≥60% — do NOT skip T0/T1)
      - ADE player team         (12% at LC 50-60%; T0/T1 still valid)

    T0 (~87% WR):   LC >= 70%
    T1 (~68% WR):   LC 60–70%
    T2 (~60% WR):   LC 50–60%
    """
    try:
        if stat_type != 'tackles':
            return {"priority": "", "description": ""}

        def _get(key, alt=None):
            v = player_row.get(key, player_row.get(alt or key, ''))
            return '' if v is None else str(v).strip()

        position = _get('Position', 'position')
        opponent = _get('Opponent', 'opponent')
        line_str = _get('Line')

        if not line_str:
            return {"priority": "", "description": ""}
        try:
            line_value = float(line_str)
        except (ValueError, TypeError):
            return {"priority": "", "description": ""}

        # ── Hard avoids ───────────────────────────────────────────────────────
        if line_value < 3:
            return {"priority": "", "description": ""}
        if line_value >= 6:
            return {"priority": "", "description": ""}
        if opponent in TACKLE_BAD_OPPONENTS:
            return {"priority": "", "description": ""}
        if position == 'MedF':
            return {"priority": "", "description": ""}

        stadium = str(player_row.get('Stadium', player_row.get('stadium', '')))
        if any(g.lower() in stadium.lower() for g in NARROW_GROUNDS_NO_TACKLE):
            return {"priority": "", "description": ""}

        # AvL > +10%: career avg is 10%+ above line
        avl_str = _get('Avg vs Line', 'avg_vs_line')
        avl_value = None
        if avl_str:
            try:
                avl_value = float(avl_str.replace('%', '').replace('+', ''))
            except (ValueError, TypeError):
                avl_value = None

        if avl_value is not None and avl_value > 10.0:
            return {"priority": "", "description": ""}

        # Rain: wet ball suppresses tackle counts
        weather = _get('Weather')
        if weather and 'rain' in weather.lower():
            return {"priority": "", "description": ""}

        # ── Parse LC ──────────────────────────────────────────────────────────
        lc_str = _get('Line Consistency')
        lc_value = None
        if lc_str:
            try:
                lc_value = float(lc_str.replace('%', ''))
            except (ValueError, TypeError):
                lc_value = None

        # ── Tiering (LC-driven, Apr 2026 cross-tab) ───────────────────────────
        team    = _get('Team', 'team')
        dvp     = _get('DvP', 'dvp')
        lc_70   = lc_value  is not None and lc_value  >= 70.0
        lc_60   = lc_value  is not None and lc_value  >= 60.0
        lc_50   = lc_value  is not None and lc_value  >= 50.0
        edge    = f"  ⭐{team}" if team in TACKLE_EDGE_TEAMS else ""

        if lc_70:
            return {"priority": "T0", "description": f"T0  LC {lc_value:.0f}%{edge}"}
        if lc_60:
            return {"priority": "T1", "description": f"T1  LC {lc_value:.0f}%{edge}"}
        if lc_50:
            if team in TACKLE_AVOID_T2_TEAMS:
                return {"priority": "", "description": ""}
            if dvp and 'Neutral' in dvp:
                return {"priority": "", "description": ""}
            return {"priority": "T2", "description": f"T2  LC {lc_value:.0f}%{edge}"}

        return {"priority": "", "description": ""}

    except Exception as e:
        print(f"ERROR in calculate_bet_flag: {e}")
        return {"priority": "", "description": "ERROR"}


# =============================================================================
# HISTORICAL WIN-RATE LOOKUP
# Used by the legs table to show contextual WR for each flagged bet.
# Cached with a 5-minute TTL so browser refreshes stay fast but data
# stays current throughout a weekend session.
# =============================================================================

BET_HISTORY_TTL = 1800  # seconds — re-fetch after 30 minutes
_BET_HISTORY: dict = {'df': None, 'cached_at': None}
_BET_HISTORY_LOCK = __import__('threading').Lock()

_RAW_SHEET: dict = {'records': None, 'cached_at': None}
_RAW_SHEET_LOCK = __import__('threading').Lock()

def _load_raw_sheet():
    """Single cached fetch of all Bet Log records. All other sheet readers call this."""
    now = datetime.now()
    if (_RAW_SHEET['records'] is not None and
            _RAW_SHEET['cached_at'] is not None and
            (now - _RAW_SHEET['cached_at']).total_seconds() < BET_HISTORY_TTL):
        return _RAW_SHEET['records']

    with _RAW_SHEET_LOCK:
        now = datetime.now()
        if (_RAW_SHEET['records'] is not None and
                _RAW_SHEET['cached_at'] is not None and
                (now - _RAW_SHEET['cached_at']).total_seconds() < BET_HISTORY_TTL):
            return _RAW_SHEET['records']

        client = get_sheets_client()
        if client is None:
            return None
        try:
            records = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_TAB).get_all_records()
            _RAW_SHEET['records'] = records
            _RAW_SHEET['cached_at'] = datetime.now()
            return records
        except Exception as e:
            print(f"WARN: _load_raw_sheet failed: {e}")
            return _RAW_SHEET['records']  # return stale data if available


def _invalidate_bet_history():
    """Force the next _load_bet_history() call to re-fetch from Sheets."""
    _BET_HISTORY['df'] = None
    _BET_HISTORY['cached_at'] = None


def _load_bet_history():
    """
    Load historical tackle bet rows from Google Sheets.
    Returns a DataFrame with Position, DvP, Opponent, W/L columns (already resolved).
    Cached for BET_HISTORY_TTL seconds; re-fetches automatically when stale.
    """
    now = datetime.now()
    if (_BET_HISTORY['df'] is not None and
            _BET_HISTORY['cached_at'] is not None and
            (now - _BET_HISTORY['cached_at']).total_seconds() < BET_HISTORY_TTL):
        return _BET_HISTORY['df']

    with _BET_HISTORY_LOCK:
        # Re-check inside lock — another thread may have fetched while we waited
        now = datetime.now()
        if (_BET_HISTORY['df'] is not None and
                _BET_HISTORY['cached_at'] is not None and
                (now - _BET_HISTORY['cached_at']).total_seconds() < BET_HISTORY_TTL):
            return _BET_HISTORY['df']

        try:
            records = _load_raw_sheet()
            if not records:
                _BET_HISTORY['df'] = pd.DataFrame()
                _BET_HISTORY['cached_at'] = datetime.now()
                return _BET_HISTORY['df']

            df = pd.DataFrame(records)

            # Keep only Tackle rows with a concrete W/L result
            if 'Type' in df.columns:
                df = df[df['Type'].astype(str).str.strip() == 'Tackle'].copy()
            df = df[df['W/L'].astype(str).str.strip().isin(['1', '-1', '0', '1.0', '-1.0', '0.0'])]
            if df.empty:
                _BET_HISTORY['df'] = df
                _BET_HISTORY['cached_at'] = datetime.now()
                return df

            df['W/L']      = pd.to_numeric(df['W/L'], errors='coerce')
            df['win']      = (df['W/L'] == 1).astype(int)
            for col in ('Position', 'DvP', 'Opponent', 'Team', 'Travel Fatigue',
                        'Line', 'Avg vs Line', 'Line Consistency',
                        'Last 5 Tackles', 'Avg Tackles', 'Year', 'Round', 'Strategy'):
                if col not in df.columns:
                    df[col] = ''
                df[col] = df[col].astype(str).str.strip()

            # Apply all hard filters so WR is computed only on clean, qualifying legs
            mask = df.apply(_passes_bet_filters, axis=1)
            df = df[mask]
            if df.empty:
                _BET_HISTORY['df'] = df
                _BET_HISTORY['cached_at'] = datetime.now()
                return df

            _BET_HISTORY['df'] = df[['Position', 'DvP', 'Opponent', 'win', 'Round', 'Year', 'Strategy']].copy()
            _BET_HISTORY['cached_at'] = datetime.now()
            return _BET_HISTORY['df']

        except Exception as e:
            print(f"WARN: _load_bet_history failed: {e}")
            _BET_HISTORY['df'] = pd.DataFrame()
            return _BET_HISTORY['df']


def get_leg_historical_wr(position: str, dvp: str, opponent: str) -> str:
    """
    Return a "WR% (n)" string for a given position × DvP × opponent combination.

    Cascade (most specific → least specific):
      1. Position + DvP + Opponent   (n ≥ 5)
      2. Position + DvP              (n ≥ 8)
      3. Position only               (n ≥ 5)
      4. ''  — not enough data

    The threshold at each level is deliberately low so common combos show up.
    """
    hist = _load_bet_history()
    if hist.empty:
        return ""

    def _wr(mask):
        sub = hist[mask]
        n   = len(sub)
        if n == 0:
            return None, 0
        wr = sub['win'].sum() / n * 100
        return wr, n

    # Level 1: all three match
    wr, n = _wr(
        (hist['Position'] == position) &
        (hist['DvP']      == dvp)      &
        (hist['Opponent'] == opponent)
    )
    if n >= 5:
        return f"{wr:.0f}% ({n})"

    # Level 2: position + DvP
    wr, n = _wr(
        (hist['Position'] == position) &
        (hist['DvP']      == dvp)
    )
    if n >= 8:
        return f"{wr:.0f}% ({n})"

    # Level 3: position only
    wr, n = _wr(hist['Position'] == position)
    if n >= 5:
        return f"{wr:.0f}% ({n})"

    return ""


def _parse_wr_str(s):
    """Parse '67% (23)' -> (67.0, 23). Returns (None, 0) on failure."""
    try:
        pct_part, n_part = str(s).split('%')
        pct = float(pct_part.strip())
        n   = int(n_part.strip().strip('()'))
        return pct, n
    except Exception:
        return None, 0


def _wr_range_str(s):
    """Return ±1 SE range string, e.g. '45-60%'."""
    import math
    pct, n = _parse_wr_str(s)
    if pct is None or n == 0:
        return ""
    se = math.sqrt(pct / 100 * (1 - pct / 100) / n) * 100
    lo = max(0.0,   pct - se)
    hi = min(100.0, pct + se)
    return f"{lo:.0f}-{hi:.0f}%"


def _confidence_str(s):
    """Return High / Med / Low / — based on sample size."""
    _, n = _parse_wr_str(s)
    if n >= 30: return "High"
    if n >= 15: return "Med"
    if n >= 5:  return "Low"
    return "—"


def add_score_to_dataframe(df, team_weather):
    df['Score']        = ""
    df['ScoreFactors'] = ""
    for idx, row in df.iterrows():
        sd    = calculate_score(row, team_weather)
        sv    = sd["ScoreValue"]
        label = ("Strong Play" if sv >= 9 else "Good Play" if sv >= 6
                 else "Consider" if sv >= 3 else "Weak" if sv >= 0 else "Avoid")
        df.at[idx, 'Score']        = f"{sv:.1f} - {label}"
        df.at[idx, 'ScoreFactors'] = sd["Factors"]
    return df


STAKING_UNITS = {
    "T1": "2 units",   # Premium — Wing/Ruck or Strong Unders DvP
    "T2": "1 unit",    # Standard — all other passing filters
}

# Watch codes — none in T1/T2 system
WATCH_CODES = set()
# Active codes — eligible for multi builder
ACTIVE_CODES = {"T1", "T2"}

def add_bet_flag_to_dataframe(df, stat_type='disposals'):
    results            = df.apply(lambda row: calculate_bet_flag(row, stat_type), axis=1)
    df['Bet_Priority'] = results.apply(lambda x: x['priority'])
    df['Units']        = df['Bet_Priority'].apply(lambda p: STAKING_UNITS.get(str(p), ""))
    if stat_type == 'tackles':
        for _, row in df.iterrows():
            r = calculate_bet_flag(row, stat_type)
            player = row.get('Player', row.get('player', '?'))
            line   = row.get('Line', '')
            pos    = row.get('Position', row.get('position', ''))
            opp    = row.get('Opponent', row.get('opponent', ''))
            dvp    = row.get('DvP', row.get('dvp', ''))
            travel = row.get('Travel Fatigue', row.get('travel_fatigue', ''))
            avl    = row.get('Avg vs Line', '')
            print(f"  [{player}] line={line} pos={pos} opp={opp} dvp={dvp} travel={travel} avl={avl} => {r['priority'] or 'SKIP'} ({r['description']})")
    return df


def add_pickem_lines_to_dataframe(df, stat_type='disposals'):
    print(f"Adding pickem lines for {stat_type}...")
    try:
        pickem_data = get_pickem_data_for_dashboard(stat_type)
        if not pickem_data:
            df['Line'] = ""
            return df

        def get_player_line(player_name):
            if not player_name or pd.isna(player_name):
                return ""
            player_name = str(player_name).strip()

            for pp, lv in pickem_data.items():
                if player_name.lower() == pp.lower():
                    return str(lv)
            try:
                nn = normalize_player_name(player_name)
                for pp, lv in pickem_data.items():
                    if nn.lower() == normalize_player_name(pp).lower():
                        return str(lv)
            except Exception:
                pass
            pns = player_name.replace(" ", "").lower()
            for pp, lv in pickem_data.items():
                if pns == pp.replace(" ", "").lower():
                    return str(lv)
            try:
                parts = player_name.split()
                if len(parts) >= 2:
                    for pp, lv in pickem_data.items():
                        pp_parts = pp.split()
                        if (len(pp_parts) >= 2
                                and parts[-1].lower() == pp_parts[-1].lower()
                                and parts[0][0].lower() == pp_parts[0][0].lower()):
                            return str(lv)
            except Exception:
                pass
            for pp, lv in pickem_data.items():
                if (player_name.lower() in pp.lower() or pp.lower() in player_name.lower()):
                    if min(len(player_name), len(pp)) >= 4:
                        return str(lv)
            return ""

        df['Line'] = df['player'].apply(get_player_line)
        matched = sum(1 for l in df['Line'] if l != "")
        print(f"Matched {matched}/{len(df)} players")
        return df
    except Exception as e:
        print(f"Error adding pickem lines: {e}")
        df['Line'] = ""
        return df


def _load_tackle_avgs():
    """Load scraped tackle averages CSV → (avg_map, last5_map, gbg_df)."""
    try:
        avgs = pd.read_csv("afl_player_tackle_avgs.csv")
        avgs['PlayerFmt'] = avgs['PlayerFmt'].str.strip()
        avg_map   = dict(zip(avgs['PlayerFmt'], avgs['career_avg']))
        last5_map = dict(zip(avgs['PlayerFmt'], avgs['last5_avg']))

        # Dabble display name → CSV PlayerFmt (names that differ beyond apostrophes)
        _dabble_aliases = {
            "Alixzander Tauru":   "Alix Tauru",
            "Archer May":         "Archie May",
            "Bailey J. Williams": "Bailey Williams",
            "Christopher Scerri": "Chris Scerri",
            "Connor O'Sullivan":  "Connor OSullivan",
            "Finn O'Sullivan":    "Finn OSullivan",
            "Harrison Petty":     "Harry Petty",
            "Harry O'Farrell":    "Harry OFarrell",
            "James O'Donnell":    "James ODonnell",
            "Jordan De Goey":     "Jordan de Goey",
            "Kieran Briggs":      "Kieren Briggs",
            "Lachie Shultz":      "Lachie Schultz",
            "Mark O'Connor":      "Mark OConnor",
            "Massimo D'Ambrosio": "Massimo DAmbrosio",
            "Mitchell Hinge":     "Mitch Hinge",
            "Reilly O'Brien":     "Reilly OBrien",
            "Thomas Edwards":     "Tom Edwards",
            "Thomas Sims":        "Tom Sims",
            "Xavier O'Halloran":  "Xavier OHalloran",
        }
        for dabble_name, csv_name in _dabble_aliases.items():
            if csv_name in avg_map:
                avg_map[dabble_name]   = avg_map[csv_name]
                last5_map[dabble_name] = last5_map.get(csv_name)

        gbg25 = pd.read_csv("afl_tackles_2025_gbg.csv")
        gbg26 = pd.read_csv("afl_tackles_2026_gbg.csv")
        gbg = pd.concat([gbg25, gbg26], ignore_index=True)
        name_map = {
            "Jaeger OMeara":    "Jaeger O'Meara",
            "Nathan ODriscoll": "Nathan O'Driscoll",
            "Leo Lombard":      "Leonardo Lombard",
            "Matt Carroll":     "Matthew Carroll",
        }
        def fmt(raw):
            s = str(raw).strip()
            if ',' not in s:
                return s
            parts = s.split(',', 1)
            n = f"{parts[1].strip()} {parts[0].strip()}"
            return name_map.get(n, n)
        gbg['Player'] = gbg['Player'].apply(fmt)
        return avg_map, last5_map, gbg
    except Exception as e:
        print(f"Tackle avg load error: {e}")
        return {}, {}, pd.DataFrame()


def add_line_analysis_columns(df, stat_type='disposals'):
    print(f"Adding line analysis for {stat_type}...")

    # For tackles, use scraped averages (2025+2026 game-by-game)
    if stat_type == 'tackles':
        try:
            avg_map, last5_map, gbg = _load_tackle_avgs()
            df['Avg vs Line']     = ""
            df['Line Consistency'] = ""
            df['Avg Tackles']     = ""
            df['Last 5 Tackles']  = ""
            for idx, row in df.iterrows():
                player = str(row.get('Player', row.get('player', ''))).strip()
                line_str = str(row.get('Line', '')).strip()
                if not player or not line_str:
                    continue
                try:
                    line_value = float(line_str)
                except (ValueError, TypeError):
                    continue
                career = avg_map.get(player)
                last5  = last5_map.get(player)
                if career is not None:
                    avl = (career - line_value) / line_value * 100
                    df.at[idx, 'Avg vs Line']    = f"{avl:+.1f}%"
                    df.at[idx, 'Avg Tackles']    = f"{career:.1f}"
                if last5 is not None:
                    df.at[idx, 'Last 5 Tackles'] = f"{float(last5):.1f}"
                if not gbg.empty:
                    grp = gbg[gbg['Player'] == player]['Tackles']
                    if len(grp) >= 3:
                        lc = (grp < line_value).sum() / len(grp) * 100
                        df.at[idx, 'Line Consistency'] = f"{lc:.1f}%"
            return df
        except Exception as e:
            print(f"Error in tackles line analysis: {e}")
            df['Avg vs Line']     = ""
            df['Line Consistency'] = ""
            df['Avg Tackles']     = ""
            df['Last 5 Tackles']  = ""
            return df

    try:
        stats_df = pd.read_csv("afl_player_stats.csv", skiprows=3).fillna(0)

        if stat_type not in stats_df.columns:
            if stat_type == 'disposals' and 'kicks' in stats_df.columns:
                stats_df[stat_type] = stats_df['kicks'] + stats_df['handballs']
            else:
                df['Avg vs Line']     = ""
                df['Line Consistency'] = ""
                return df

        df['Avg vs Line']     = ""
        df['Line Consistency'] = ""

        for idx, row in df.iterrows():
            player_name = row.get('Player', row.get('player', ''))
            line_str    = row.get('Line', '')
            if not line_str:
                continue
            try:
                line_value = float(line_str)
            except (ValueError, TypeError):
                continue

            ps = stats_df[stats_df['player'].str.lower() == player_name.lower()]
            if ps.empty:
                for sp in stats_df['player'].unique():
                    if (sp.lower().replace(" ", "") == player_name.lower().replace(" ", "")
                            or (len(player_name.split()) >= 2 and len(sp.split()) >= 2
                                and player_name.split()[-1].lower() == sp.split()[-1].lower()
                                and player_name.split()[0][0].lower() == sp.split()[0][0].lower())):
                        ps = stats_df[stats_df['player'] == sp]
                        break
            if ps.empty:
                continue

            vals = ps[stat_type].values
            avg  = vals.mean()
            if avg > 0:
                df.at[idx, 'Avg vs Line'] = f"{((line_value / avg) - 1) * 100:+.1f}%"
            below = sum(1 for v in vals if v < line_value)
            if len(vals) > 0:
                df.at[idx, 'Line Consistency'] = f"{(below / len(vals)) * 100:.1f}%"

        return df
    except Exception as e:
        print(f"Error adding line analysis: {e}")
        df['Avg vs Line']     = ""
        df['Line Consistency'] = ""
        return df


def process_data_for_dashboard(stat_type='disposals'):
    global dvp_data_by_stat
    try:
        print(f"process_data_for_dashboard({stat_type})")

        # fixtures
        try:
            fixtures = scrape_next_round_fixture()
            if not fixtures:
                fixtures = [{"match": "Team A vs Team B",
                             "datetime": datetime.now(pytz.timezone('Australia/Melbourne')),
                             "stadium": "MCG"}]
        except Exception as e:
            print(f"Fixtures error: {e}")
            fixtures = [{"match": "Team A vs Team B",
                         "datetime": datetime.now(pytz.timezone('Australia/Melbourne')),
                         "stadium": "MCG"}]

        # weather
        weather_data = {}
        for match in fixtures:
            stadium = match["stadium"]
            latlon  = STADIUM_COORDS.get(stadium)
            if not latlon:
                for key in STADIUM_COORDS:
                    if key.lower() in stadium.lower():
                        latlon = STADIUM_COORDS[key]
                        break
                latlon = latlon or (-37.8199, 144.9834)
            try:
                forecast  = get_forecast(*latlon)
                weather   = extract_weather_for_datetime(forecast, match["datetime"])
                weather_data[match["match"]] = weather or {"rain": 0, "wind": 0, "humidity": 50}
            except Exception as e:
                weather_data[match["match"]] = {"rain": 0, "wind": 0, "humidity": 50}

        # travel log (used for build only — real fatigue via hardcoded pairs)
        try:
            build_travel_log()
        except Exception:
            pass

        # player stats
        try:
            df = pd.read_csv("afl_player_stats.csv", skiprows=3).fillna(0)
        except Exception as e:
            print(f"Stats load error: {e}")
            df = pd.DataFrame([
                {"player": "Test Player 1", "team": "Team A", "opponent": "Team B",
                 "round": 10, "namedPosition": "CHF", "disposals": 20, "marks": 5, "tackles": 4},
            ])

        # DvP
        try:
            proc = load_and_prepare_data("afl_player_stats.csv")
            if stat_type not in proc.columns:
                proc[stat_type] = proc.get('kicks', 0) + proc.get('handballs', 0) \
                    if stat_type == 'disposals' else 0
            simplified_dvp = calculate_dvp_for_stat(proc, stat_type)
            dvp_data_by_stat[stat_type] = simplified_dvp
        except Exception as e:
            print(f"DvP error: {e}")
            simplified_dvp = {}
            dvp_data_by_stat[stat_type] = {}

        # team weather + opponents
        team_weather   = {}
        team_opponents = {}
        for match in fixtures:
            try:
                home, away = match["match"].split(" vs ")
                wr = categorize_weather_for_stat(weather_data.get(match["match"], {}), stat_type)
                for name in [home, away]:
                    team_weather[name] = wr
                    for abbr, full in TEAM_NAME_MAP.items():
                        if full == name:
                            team_weather[abbr] = wr
                team_opponents[home] = away
                team_opponents[away] = home
            except Exception:
                pass

        # recent players
        try:
            latest = df['round'].max()
            recent = df[df['round'].isin([latest, latest - 1])].copy()
            players = recent.sort_values('round', ascending=False).groupby(['player', 'team']).first().reset_index()
            if players.empty:
                players = df.sort_values('round', ascending=False).groupby(['player', 'team']).first().reset_index()
        except Exception as e:
            players = df.sort_values('round', ascending=False).groupby('player').first().reset_index()

        # travel fatigue — hardcoded long-distance pairs
        long_distance_pairs = set([
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
        ])

        home_away    = {}
        team_stadium = {}
        for match in fixtures:
            try:
                hf, af = match["match"].split(" vs ")
                ha = ab = None
                for abbr, name in TEAM_NAME_MAP.items():
                    if name == hf: ha = abbr
                    if name == af: ab = abbr
                if ha and ab:
                    home_away[ha] = {'opponent': ab, 'is_home': True}
                    home_away[ab] = {'opponent': ha, 'is_home': False}
                    fixtures_by_team[ha] = match['datetime']
                    fixtures_by_team[ab] = match['datetime']
                    # Both teams play at the same venue this match
                    stad = match.get('stadium', '')
                    team_stadium[ha] = stad
                    team_stadium[ab] = stad
            except Exception:
                pass

        travel_dict = {}
        for team in players['team'].unique():
            info     = home_away.get(team, {})
            opp      = info.get('opponent')
            is_home  = info.get('is_home', True)
            if not is_home and opp and (team, opp) in long_distance_pairs:
                travel_dict[team] = "🟠 Moderate (Long Travel)"
            else:
                travel_dict[team] = "✅ Neutral"

        players['travel_fatigue'] = players['team'].map(lambda x: travel_dict.get(x, "✅ Neutral"))
        players['weather']        = players['team'].map(
            lambda x: team_weather.get(x, {"Rating": "✅ Neutral"}).get('Rating', "✅ Neutral"))

        def get_opp(team):
            full = TEAM_NAME_MAP.get(team, team)
            opp  = team_opponents.get(full, "Unknown")
            for a, n in TEAM_NAME_MAP.items():
                if n == opp:
                    return a
            return opp

        players['opponent'] = players['team'].apply(get_opp)
        players['stadium']  = players['team'].map(lambda x: team_stadium.get(x, ''))

        # ── Positions from master CSV (source of truth) ──────────────────────
        pos_lookup = load_player_positions(df)
        players["position"] = players["player"].apply(
            lambda name: pos_lookup.get(str(name).lower().strip(), "Unknown")
        )
        unknown_count = (players["position"] == "Unknown").sum()
        if unknown_count > 0:
            unknown_names = players.loc[players["position"] == "Unknown", "player"].tolist()
            print(f"WARN: {unknown_count} player(s) not in {PLAYER_POSITIONS_FILE}: {unknown_names}")
        # ─────────────────────────────────────────────────────────────────────

        DVP_MAP = {
            "Strong Unders":   "🔴 Strong Unders",
            "Moderate Unders": "🟠 Moderate Unders",
            "Slight Unders":   "🟡 Slight Unders",
            "Strong Easy":     "🔵 Strong Easy",
            "Moderate Easy":   "🔷 Moderate Easy",
            "Slight Easy":     "🔹 Slight Easy",
        }

        # SmF / MedF / FwdMid are sub-groups of GenF; map back for DvP lookup
        DVP_POS_NORM = {"SmF": "GenF", "MedF": "GenF", "FwdMid": "GenF"}

        def get_dvp(row):
            team     = row.get('opponent', 'Unknown')
            pos      = row.get('position', 'Unknown')
            dvp_pos  = DVP_POS_NORM.get(pos, pos)
            if team == 'Unknown' or dvp_pos == 'Unknown':
                return "⚠️ Unknown"
            if team in simplified_dvp and dvp_pos in simplified_dvp[team]:
                return DVP_MAP.get(simplified_dvp[team][dvp_pos]["strength"], "✅ Neutral")
            return "✅ Neutral"

        players['dvp'] = players.apply(get_dvp, axis=1)

        result = add_pickem_lines_to_dataframe(players, stat_type)
        result = add_line_analysis_columns(result, stat_type)

        tackle_cols = [c for c in ['Avg Tackles', 'Last 5 Tackles'] if c in result.columns]
        result = result[['player', 'team', 'opponent', 'position', 'travel_fatigue',
                          'weather', 'dvp', 'Line', 'Avg vs Line', 'Line Consistency',
                          'stadium'] + tackle_cols].copy()

        result = add_score_to_dataframe(result, team_weather)
        result = add_bet_flag_to_dataframe(result, stat_type)

        display = result[['player', 'team', 'opponent', 'position', 'travel_fatigue',
                           'weather', 'dvp', 'Line', 'Avg vs Line', 'Line Consistency',
                           'stadium', 'Bet_Priority', 'Units'] + tackle_cols].copy()

        base_cols = ['Player', 'Team', 'Opponent', 'Position', 'Travel Fatigue',
                     'Weather', 'DvP', 'Line', 'Avg vs Line', 'Line Consistency',
                     'Stadium', 'Bet Priority', 'Units']
        display.columns = base_cols + tackle_cols

        display = display[display['Line'] != ""].copy()

        def sort_priority(p):
            try:
                return int(p)
            except Exception:
                return 999

        display['_sort'] = display['Bet Priority'].apply(sort_priority)
        display = display.sort_values(['_sort', 'Team']).drop('_sort', axis=1)

        print(f"OK {stat_type}: {len(display)} rows")
        return display

    except Exception as e:
        print(f"CRITICAL ERROR ({stat_type}): {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame([{
            'Player': f'Error: {e}', 'Team': '', 'Opponent': '', 'Position': '',
            'Travel Fatigue': '', 'Weather': '', 'DvP': '', 'Line': '',
            'Avg vs Line': '', 'Line Consistency': '', 'Bet Priority': ''
        }])


# ===== MULTI BUILDER =====

STRATEGY_WR_MAP = {
    "T0": 87.0,   # LC >= 70%   (train/test avg: 91.7% / 86.4%)
    "T1": 68.0,   # LC 60-70%   (train/test avg: 75.0% / 65.4%)
    "T2": 60.0,   # LC 50-60%  (train/test avg: 60.7% / 67.6%)
}


def _strat_tier(priority):
    """Map Bet Priority (T1/T2) + strategy description to S1/S2/S3."""
    p = str(priority).strip()
    # T1 covers both S1 (Wing/Ruck) and S2 (Strong Unders DvP).
    # We use Bet Priority only here; the description field distinguishes them
    # but isn't stored in the live legs table.  We return T1/T2 as-is.
    return p


def leg_quality_score(row) -> float:
    """
    Continuous quality score within a tier (0–100).
    Ranks legs by how confidently they're expected to go under,
    using three independent signals:

      Consistency (40%) — how often this player has historically hit the unders line.
                          Most predictive of future hits.
      AvL margin  (35%) — how far below average the line sits.
                          Wider margin = more cushion = safer bet.
      DvP strength(25%) — how much the opponent suppresses tackles.
                          S2 Strong Unders > Moderate > Slight > Neutral > Easy.

    T1 legs are always ranked above T2 in pairing priority — this score
    is used only to rank *within* T2.
    """
    # ── Consistency: raw percentage, 0–100 ───────────────────────────────────
    try:
        cons_str = str(row.get('Line Consistency', '') or '').replace('%', '').strip()
        cons = float(cons_str) if cons_str else 33.3
    except (ValueError, TypeError):
        cons = 33.3

    # ── AvL: negative = going under (good). Clamp to [-30, +10] then invert ──
    try:
        avl_str = str(row.get('Avg vs Line', '') or '').replace('%', '').replace('+', '').strip()
        avl = float(avl_str) if avl_str else 0.0
    except (ValueError, TypeError):
        avl = 0.0
    # Score 0–100: -30% AvL → 100, 0% → 50, +10% → ~17
    avl_score = max(0.0, min(100.0, (10.0 - avl) / 40.0 * 100.0))

    # ── DvP: opponent suppression strength → ordinal score ───────────────────
    dvp = str(row.get('DvP', '') or '')
    if   'Strong Unders'   in dvp: dvp_score = 100.0
    elif 'Moderate Unders' in dvp: dvp_score =  75.0
    elif 'Slight Unders'   in dvp: dvp_score =  55.0
    elif 'Neutral'         in dvp: dvp_score =  40.0
    elif 'Slight Easy'     in dvp: dvp_score =  25.0
    elif 'Moderate Easy'   in dvp: dvp_score =  10.0
    else:                          dvp_score =  40.0  # unknown → neutral

    # T1 legs get a 20-point bonus so they always rank above T2 in combined scores
    tier_bonus = 20.0 if str(row.get('Bet Priority', '')).strip() == 'T1' else 0.0

    return round(cons * 0.40 + avl_score * 0.35 + dvp_score * 0.25 + tier_bonus, 2)


def build_hedge_picks(legs_by_quality):
    """
    Tiered hedge pick builder.
    legs_by_quality: list of row dicts sorted T1-first then T2 by quality score (desc).

    Returns: (format_name, payout_desc, combo_picks, combo_meta, jackpot_info)
      combo_picks:  list of lists of leg dicts  (the 6-leg combo plays)
      combo_meta:   dict with min_wins, tiers, breakeven
      jackpot_info: None for small slates; dict with fmt/desc/pick/meta for n>=7
    """
    from itertools import combinations as _combs
    n = len(legs_by_quality)

    # Jackpot payout table (format: tiers list, desc string)
    _JP = {
        6:  ([(4, 0.5), (5, 2.5),  (6, 25.0)],
             "4/6 -> 0.5x  .  5/6 -> 2.5x  .  6/6 -> 25x  (+256% EV)"),
        7:  ([(5, 1.0), (6, 4.0),  (7, 40.0)],
             "5/7 -> 1x  .  6/7 -> 4x  .  7/7 -> 40x  (+320% EV)"),
        8:  ([(6, 2.0), (7, 5.0),  (8, 75.0)],
             "6/8 -> 2x  .  7/8 -> 5x  .  8/8 -> 75x  (+431% EV)"),
        9:  ([(7, 3.0), (8, 15.0), (9, 100.0)],
             "7/9 -> 3x  .  8/9 -> 15x  .  9/9 -> 100x  (+540% EV)"),
        10: ([(7, 0.5), (8, 5.0),  (9, 25.0),  (10, 125.0)],
             "7/10 -> 0.5x  .  8/10 -> 5x  .  9/10 -> 25x  .  10/10 -> 125x  (+597% EV)"),
        11: ([(8, 2.0), (9, 5.0),  (10, 25.0), (11, 250.0)],
             "8/11 -> 2x  .  9/11 -> 5x  .  10/11 -> 25x  .  11/11 -> 250x  (+665% EV)"),
        12: ([(9, 2.5), (10, 10.0),(11, 50.0), (12, 500.0)],
             "9/12 -> 2.5x  .  10/12 -> 10x  .  11/12 -> 50x  .  12/12 -> 500x  (+994% EV)"),
    }

    def _jackpot(jp_n):
        """Build jackpot_info dict for a jp_n-leg single entry."""
        tiers, desc = _JP[jp_n]
        return {
            'fmt':  f"{jp_n}-leg Jackpot  (1 entry)",
            'desc': desc,
            'pick': legs_by_quality[:jp_n],
            'meta': {'min_wins': tiers[0][0], 'tiers': tiers, 'n': jp_n},
        }

    # ── Small slate: single entry only, no jackpot ────────────────────────────
    if n < 3:
        return ("SKIP",
                f"{n} leg{'s' if n != 1 else ''} available — minimum 3 required",
                [], {}, None)

    elif n == 3:
        return ("3-leg Hedge",
                "2/3 -> 1.2x  .  3/3 -> 3.0x  .  breakeven 55.3%",
                [legs_by_quality[:3]],
                {"min_wins": 2, "tiers": [(2, 1.2), (3, 3.0)], "breakeven": "55.3%"},
                None)

    elif n == 4:
        return ("4-leg Hedge",
                "3/4 -> 2.0x  .  4/4 -> 5.0x  .  breakeven 53.9%",
                [legs_by_quality[:4]],
                {"min_wins": 3, "tiers": [(3, 2.0), (4, 5.0)], "breakeven": "53.9%"},
                None)

    elif n == 5:
        return ("5-leg Hedge",
                "3/5 -> 0.5x  .  4/5 -> 2.0x  .  5/5 -> 10.0x  .  breakeven 53.7%",
                [legs_by_quality[:5]],
                {"min_wins": 3, "tiers": [(3, 0.5), (4, 2.0), (5, 10.0)], "breakeven": "53.7%"},
                None)

    elif n == 6:
        # Single 6-leg entry — combo and jackpot are the same pick, no separate jackpot
        return ("6-leg Hedge",
                "4/6 -> 0.5x  .  5/6 -> 2.5x  .  6/6 -> 25.0x  .  breakeven 53.2%",
                [legs_by_quality[:6]],
                {"min_wins": 4, "tiers": [(4, 0.5), (5, 2.5), (6, 25.0)], "breakeven": "53.2%"},
                None)

    else:  # n >= 7 — balanced picks
        def _balanced_picks(legs, n_picks, pick_size, team_cap=2, game_cap=3):
            """
            Generate n_picks of pick_size legs, balancing exposure across all legs.
            Each round: sort by (current_exposure ASC, quality_rank ASC), pick first
            pick_size that respect team_cap and game_cap. Relax caps if needed to fill.
            Game identity is frozenset({team, opponent}) so SYD vs NTH == NTH vs SYD.
            """
            quality_rank = {l['Player']: i for i, l in enumerate(legs)}
            exposure     = {l['Player']: 0 for l in legs}
            picks = []
            for _ in range(n_picks):
                candidates = sorted(
                    legs,
                    key=lambda l: (exposure[l['Player']], quality_rank[l['Player']])
                )
                team_counts, game_counts, pick = {}, {}, []
                for leg in candidates:
                    if len(pick) >= pick_size:
                        break
                    team = leg.get('Team', '')
                    game = frozenset({team, leg.get('Opponent', '')})
                    if (team_counts.get(team, 0) < team_cap and
                            game_counts.get(game, 0) < game_cap):
                        pick.append(leg)
                        team_counts[team] = team_counts.get(team, 0) + 1
                        game_counts[game] = game_counts.get(game, 0) + 1
                if len(pick) < pick_size:
                    for leg in candidates:
                        if leg not in pick and len(pick) < pick_size:
                            pick.append(leg)
                for leg in pick:
                    exposure[leg['Player']] += 1
                picks.append(pick)
            return picks

        jp_n = min(n, 12)

        if n >= 9:
            # 9 balanced 8-leg picks from all available legs.
            # Each leg appears in ~(9×8/n) tickets — at n=12 that's ~6/9 = 67%.
            # game_cap=3 prevents >3 legs from same game per ticket (SYD/NTH correlation guard).
            combo_picks = _balanced_picks(legs_by_quality, 9, 8, team_cap=2, game_cap=3)
            combo_meta  = {"min_wins": 6, "tiers": [(6, 2.0), (7, 5.0), (8, 75.0)], "breakeven": "53.3%"}
            payout_desc = "6/8 -> 2x  .  7/8 -> 5x  .  8/8 -> 75x  .  breakeven 53.3%"
            fmt_label   = f"8-leg Hedge  .  9 balanced picks from {n} available"
        else:  # n == 7 or 8 — leave-one-out 6-leg (top 7 legs)
            pool        = legs_by_quality[:7]
            combo_picks = [[pool[j] for j in range(len(pool)) if j != i]
                           for i in range(len(pool))]
            combo_meta  = {"min_wins": 4, "tiers": [(4, 0.5), (5, 2.5), (6, 25.0)], "breakeven": "53.2%"}
            payout_desc = "4/6 -> 0.5x  .  5/6 -> 2.5x  .  6/6 -> 25.0x  .  breakeven 53.2%"
            fmt_label   = f"6-leg Hedge  .  {len(pool)} leave-one-out picks"

        return (
            fmt_label,
            payout_desc,
            combo_picks,
            combo_meta,
            _jackpot(jp_n),
        )


def build_multi_builder_layout(checked_ids=None, rr_top_n=4, excluded_teams=None, excluded_legs=None):
    # Collect bets across all stat types:
    #   all_bets  — flagged rows only (T1/T2), used for hedge logic
    #   all_rows  — every row with a Line, used for the full legs table
    all_bets = []
    all_rows = []
    for stat_type, label in [('tackles', 'Tackles')]:
        df = processed_data_by_stat.get(stat_type)
        if df is None or df.empty:
            print(f"multi_builder: {stat_type} — no data")
            continue
        flagged = df[df['Bet Priority'].astype(str).str.strip() != ''].copy()
        flagged['Stat'] = label
        all_bets.append(flagged)
        # All tackle rows with a line value (for display table)
        if stat_type == 'tackles':
            has_line = df[df['Line'].astype(str).str.strip().replace('', float('nan')).notna()].copy()
            has_line['Stat'] = label
            all_rows.append(has_line)

    if not all_bets:
        return dbc.Alert(
            "No flagged bets found. Load data first or check that Dabble markets are open.",
            color="warning", className="mt-3"
        ), {}

    combined     = pd.concat(all_bets, ignore_index=True)
    all_combined = pd.concat(all_rows, ignore_index=True) if all_rows else combined.copy()

    # ── Split into upcoming vs already started ────────────────────────────────
    now_melb = datetime.now(pytz.timezone('Australia/Melbourne'))

    def game_started(team):
        dt = fixtures_by_team.get(team)
        if dt is None:
            return False
        # ensure timezone-aware comparison
        if dt.tzinfo is None:
            dt = pytz.timezone('Australia/Melbourne').localize(dt)
        return dt < now_melb

    combined['_started'] = combined['Team'].apply(game_started)
    upcoming = combined[~combined['_started']].drop(columns=['_started'], errors='ignore')
    played   = combined[ combined['_started']].drop(columns=['_started'], errors='ignore')

    excluded       = set(excluded_teams or [])
    excluded_leg_ids = set(excluded_legs or [])

    # ── Full table — all legs after hard avoids only ──────────────────────────
    all_combined['_started'] = all_combined['Team'].apply(game_started)
    table_upcoming = all_combined[~all_combined['_started']].drop(columns=['_started'], errors='ignore').copy()
    if not table_upcoming.empty:
        table_upcoming = table_upcoming[
            pd.to_numeric(table_upcoming['Line'], errors='coerce').fillna(0) >= 3
        ].copy()
        # Compute WR columns first so sort + dimming can both use them
        table_upcoming['Hist WR'] = table_upcoming.apply(
            lambda r: get_leg_historical_wr(
                str(r.get('Position', '')), str(r.get('DvP', '')), str(r.get('Opponent', ''))
            ), axis=1
        )

    # active_upcoming is rebuilt from WR evidence after WR columns are computed
    # (placeholder here; rebuilt below after _parse_wr_str is defined)
    active_upcoming = pd.DataFrame()

    # ── Palette ───────────────────────────────────────────────────────────────
    D_BG      = "#091d26"
    D_CARD    = "rgba(18, 77, 84, 0.45)"
    D_CARD2   = "rgba(9, 29, 38, 0.6)"
    D_BORDER  = "rgba(225, 217, 207, 0.10)"
    D_TEXT    = "#e1d9cf"
    D_MUT     = "rgba(225, 217, 207, 0.55)"
    D_FADED   = "rgba(225, 217, 207, 0.25)"
    D_ACCENT  = "#f9744b"
    FONT      = "var(--display, 'Inter', sans-serif)"
    MONO      = "var(--mono, 'JetBrains Mono', monospace)"

    # ── Heat matrix helpers ───────────────────────────────────────────────────
    def _line_heat(v):
        try: v = float(v)
        except: return D_CARD, D_MUT
        if v >= 5:   return "rgba(45,212,191,0.12)", "#2dd4bf"
        if v >= 4:   return "rgba(45,212,191,0.08)", "#7dd3fc"
        if v >= 3.5: return "rgba(249,116,75,0.10)", "#f9744b"
        return "rgba(248,113,113,0.10)", "#f87171"

    def _avl_heat(v):
        try: v = float(str(v).replace('%', '').replace('+', ''))
        except: return D_CARD, D_MUT
        if v <= -10: return "rgba(45,212,191,0.12)", "#2dd4bf"
        if v <= 0:   return "rgba(45,212,191,0.08)", "#7dd3fc"
        if v <= 5:   return "rgba(249,116,75,0.10)", "#f9744b"
        if v <= 10:  return "rgba(249,116,75,0.08)", "#d84f2a"
        return "rgba(248,113,113,0.10)", "#f87171"

    def _dvp_heat(v):
        v = str(v)
        if 'Strong Unders'   in v: return "rgba(45,212,191,0.12)", "#2dd4bf"
        if 'Moderate Unders' in v: return "rgba(45,212,191,0.08)", "#7dd3fc"
        if 'Slight Unders'   in v: return "rgba(45,212,191,0.06)", "#5eead4"
        if 'Strong Easy'     in v: return "rgba(248,113,113,0.10)", "#f87171"
        if 'Moderate Easy'   in v: return "rgba(249,116,75,0.10)", "#d84f2a"
        if 'Slight Easy'     in v: return "#2e2200", "#fbbf24"
        return D_CARD, D_MUT

    def _weather_heat(v):
        v = str(v)
        if 'Strong' in v: return "rgba(45, 212, 191, 0.12)", "#2dd4bf"
        if 'Medium' in v: return "rgba(45, 212, 191, 0.07)", "#7dd3fc"
        return D_CARD, D_MUT

    def _travel_heat(v):
        v = str(v)
        if 'Long Travel' in v: return "rgba(249, 116, 75, 0.12)", "#f9744b"
        return D_CARD, D_MUT

    def _wr_heat(v):
        """Colour 'WR% (n)' strings by win-rate level."""
        if not v:
            return D_CARD, D_MUT
        try:
            pct = float(str(v).split('%')[0])
        except (ValueError, IndexError):
            return D_CARD, D_MUT
        if pct >= 65: return "rgba(45, 212, 191, 0.12)", "#2dd4bf"
        if pct >= 58: return "rgba(45, 212, 191, 0.07)", "#7dd3fc"
        if pct >= 54: return "rgba(249, 116, 75, 0.10)", "#f9744b"
        return "rgba(248, 113, 113, 0.10)", "#f87171"

    def _conf_heat(v):
        v = str(v)
        if v == "High": return "rgba(45, 212, 191, 0.12)", "#2dd4bf"
        if v == "Med":  return "rgba(45, 212, 191, 0.07)", "#7dd3fc"
        if v == "Low":  return "rgba(249, 116, 75, 0.10)", "#f9744b"
        return D_CARD, D_MUT

    # ── Compute WR Range, Confidence; sort; rebuild active pool ─────────────
    if not table_upcoming.empty:
        table_upcoming['WR Range']   = table_upcoming['Hist WR'].apply(_wr_range_str)
        table_upcoming['Confidence'] = table_upcoming['Hist WR'].apply(_confidence_str)

        def _n_sort(s):
            try: return -int(str(s).split('(')[1].rstrip(')'))
            except: return 0

        def _wr_sort(s):
            try: return -float(str(s).split('%')[0])
            except: return 0

        table_upcoming['_n'] = table_upcoming['Hist WR'].apply(_n_sort)
        table_upcoming['_w'] = table_upcoming['Hist WR'].apply(_wr_sort)
        table_upcoming = table_upcoming.sort_values(['_n', '_w']).drop(columns=['_n', '_w'])

        # ── Rebuild active_upcoming from WR evidence (not T1/T2 tiers) ──────
        # Include a leg if: passes all hard filters (no WR/confidence gate)
        def _wr_qualifies(row):
            stadium = str(row.get('Stadium', row.get('stadium', '')))
            if any(g.lower() in stadium.lower() for g in NARROW_GROUNDS_NO_TACKLE):
                return False
            return _passes_bet_filters(row)

        # Stable row id used by the DataTable for selection (persists across sorts)
        table_upcoming['id'] = (
            table_upcoming['Player'].astype(str) + '_' + table_upcoming['Team'].astype(str)
        )

        active_upcoming = table_upcoming[
            table_upcoming.apply(_wr_qualifies, axis=1) &
            table_upcoming['Bet Priority'].isin(['T0', 'T1', 'T2'])
        ].copy()
        if excluded and not active_upcoming.empty:
            active_upcoming = active_upcoming[~active_upcoming['Team'].isin(excluded)].copy()
        if excluded_leg_ids and not active_upcoming.empty:
            active_upcoming = active_upcoming[~active_upcoming['id'].isin(excluded_leg_ids)].copy()

        # Sort active legs by AvL ascending (most negative = strongest value first)
        if not active_upcoming.empty:
            active_upcoming['_avl_sort'] = pd.to_numeric(
                active_upcoming['Avg vs Line'].astype(str)
                    .str.replace('%', '').str.replace('+', ''),
                errors='coerce'
            ).fillna(0)
            active_upcoming = active_upcoming.sort_values('_avl_sort', ascending=True).drop(columns=['_avl_sort'])

    all_teams = sorted(table_upcoming['Team'].unique().tolist()) if not table_upcoming.empty else []

    # ── Prepare hidden columns for sort + conditional styling ────────────────
    if not table_upcoming.empty:
        table_upcoming['_excl'] = table_upcoming['id'].apply(
            lambda x: '×' if x in excluded_leg_ids else ''
        )
        table_upcoming['_active'] = table_upcoming.apply(
            lambda r: 0 if r.get('id', '') in excluded_leg_ids else int(
                _wr_qualifies(r) and r.get('Bet Priority', '') in ('T0', 'T1', 'T2')
            ),
            axis=1
        )
        table_upcoming['_wr_pct'] = table_upcoming['Hist WR'].apply(
            lambda s: _parse_wr_str(s)[0] if _parse_wr_str(s)[0] is not None else -1
        )
        table_upcoming['_avl_num'] = pd.to_numeric(
            table_upcoming['Avg vs Line'].astype(str)
                .str.replace('%', '').str.replace('+', ''),
            errors='coerce'
        ).fillna(0)
        table_upcoming['_cons_num'] = pd.to_numeric(
            table_upcoming['Line Consistency'].astype(str).str.replace('%', ''),
            errors='coerce'
        ).fillna(0)
        # Pre-sort: active (1) first, then WR desc
        table_upcoming = table_upcoming.sort_values(
            ['_active', '_wr_pct'], ascending=[False, False]
        )

    if table_upcoming.empty:
        legs_table = html.Div("No upcoming legs found. Load data first.",
                               style={"color": D_MUT, "fontFamily": FONT,
                                      "fontSize": "12px", "padding": "12px"})
    else:
        dt_cols = [
            {'name': '',        'id': '_excl',            'type': 'text'},
            {'name': 'Stat',    'id': 'Stat',             'type': 'text'},
            {'name': 'Player',  'id': 'Player',           'type': 'text'},
            {'name': 'Team',    'id': 'Team',             'type': 'text'},
            {'name': 'Opp',     'id': 'Opponent',         'type': 'text'},
            {'name': 'Pos',     'id': 'Position',         'type': 'text'},
            {'name': 'Tier',    'id': 'Bet Priority',     'type': 'text'},
            {'name': 'Line',    'id': 'Line',             'type': 'numeric'},
            {'name': 'AvL',     'id': 'Avg vs Line',      'type': 'text'},
            {'name': 'Cons',    'id': 'Line Consistency', 'type': 'text'},
            {'name': 'Avg',     'id': 'Avg Tackles',      'type': 'text'},
            {'name': 'L5',      'id': 'Last 5 Tackles',   'type': 'text'},
            {'name': 'DvP',     'id': 'DvP',              'type': 'text'},
            {'name': 'Weather', 'id': 'Weather',          'type': 'text'},
            {'name': 'Travel',  'id': 'Travel Fatigue',   'type': 'text'},
            {'name': 'Hist WR', 'id': 'Hist WR',          'type': 'text'},
            {'name': 'Range',   'id': 'WR Range',         'type': 'text'},
            {'name': 'Conf',    'id': 'Confidence',       'type': 'text'},
            # Hidden utility columns — must be declared for filter queries to work
            {'name': '', 'id': '_active',   'type': 'numeric'},
            {'name': '', 'id': '_wr_pct',   'type': 'numeric'},
            {'name': '', 'id': '_avl_num',  'type': 'numeric'},
            {'name': '', 'id': '_cons_num', 'type': 'numeric'},
            {'name': '', 'id': 'id',        'type': 'text'},
        ]

        cond_styles = [
            # ── Exclude toggle column ─────────────────────────────────────
            {'if': {'column_id': '_excl'},
             'cursor': 'pointer', 'textAlign': 'center',
             'color': 'rgba(225,217,207,0.25)', 'fontSize': '13px'},
            {'if': {'filter_query': '{_excl} = "×"', 'column_id': '_excl'},
             'color': '#f87171', 'fontWeight': '700'},
            # ── Tier ──────────────────────────────────────────────────────
            {'if': {'filter_query': '{Bet Priority} = "T0"', 'column_id': 'Bet Priority'},
             'backgroundColor': 'rgba(45,212,191,0.18)', 'color': '#2dd4bf', 'fontWeight': '700'},
            {'if': {'filter_query': '{Bet Priority} = "T1"', 'column_id': 'Bet Priority'},
             'backgroundColor': 'rgba(249,116,75,0.14)', 'color': '#f9744b', 'fontWeight': '700'},
            {'if': {'filter_query': '{Bet Priority} = "T2"', 'column_id': 'Bet Priority'},
             'backgroundColor': 'rgba(248,113,113,0.12)', 'color': '#f87171', 'fontWeight': '700'},
            # ── Hist WR ───────────────────────────────────────────────────
            {'if': {'filter_query': '{_wr_pct} >= 65', 'column_id': 'Hist WR'},
             'backgroundColor': 'rgba(45,212,191,0.12)', 'color': '#2dd4bf'},
            {'if': {'filter_query': '{_wr_pct} >= 56 && {_wr_pct} < 65', 'column_id': 'Hist WR'},
             'backgroundColor': 'rgba(45,212,191,0.07)', 'color': '#7dd3fc'},
            {'if': {'filter_query': '{_wr_pct} >= 53 && {_wr_pct} < 56', 'column_id': 'Hist WR'},
             'backgroundColor': 'rgba(249,116,75,0.10)', 'color': '#f9744b'},
            {'if': {'filter_query': '{_wr_pct} >= 0 && {_wr_pct} < 53', 'column_id': 'Hist WR'},
             'backgroundColor': 'rgba(248,113,113,0.10)', 'color': '#f87171'},
            # ── Confidence ────────────────────────────────────────────────
            {'if': {'filter_query': '{Confidence} = "High"', 'column_id': 'Confidence'},
             'backgroundColor': 'rgba(45,212,191,0.12)', 'color': '#2dd4bf'},
            {'if': {'filter_query': '{Confidence} = "Med"', 'column_id': 'Confidence'},
             'backgroundColor': 'rgba(45,212,191,0.07)', 'color': '#7dd3fc'},
            {'if': {'filter_query': '{Confidence} = "Low"', 'column_id': 'Confidence'},
             'backgroundColor': 'rgba(249,116,75,0.10)', 'color': '#f9744b'},
            # ── DvP ───────────────────────────────────────────────────────
            {'if': {'filter_query': '{DvP} contains "Strong Unders"', 'column_id': 'DvP'},
             'backgroundColor': 'rgba(45,212,191,0.12)', 'color': '#2dd4bf'},
            {'if': {'filter_query': '{DvP} contains "Moderate Unders"', 'column_id': 'DvP'},
             'backgroundColor': 'rgba(45,212,191,0.08)', 'color': '#7dd3fc'},
            {'if': {'filter_query': '{DvP} contains "Slight Unders"', 'column_id': 'DvP'},
             'backgroundColor': 'rgba(45,212,191,0.06)', 'color': '#5eead4'},
            {'if': {'filter_query': '{DvP} contains "Strong Easy"', 'column_id': 'DvP'},
             'backgroundColor': 'rgba(248,113,113,0.10)', 'color': '#f87171'},
            {'if': {'filter_query': '{DvP} contains "Moderate Easy"', 'column_id': 'DvP'},
             'backgroundColor': 'rgba(249,116,75,0.10)', 'color': '#d84f2a'},
            # ── Line ──────────────────────────────────────────────────────
            {'if': {'filter_query': '{Line} >= 5', 'column_id': 'Line'},
             'backgroundColor': 'rgba(45,212,191,0.12)', 'color': '#2dd4bf'},
            {'if': {'filter_query': '{Line} >= 4 && {Line} < 5', 'column_id': 'Line'},
             'backgroundColor': 'rgba(45,212,191,0.08)', 'color': '#7dd3fc'},
            {'if': {'filter_query': '{Line} >= 3.5 && {Line} < 4', 'column_id': 'Line'},
             'backgroundColor': 'rgba(249,116,75,0.10)', 'color': '#f9744b'},
            {'if': {'filter_query': '{Line} < 3.5', 'column_id': 'Line'},
             'backgroundColor': 'rgba(248,113,113,0.10)', 'color': '#f87171'},
            # ── AvL: negative = line above avg = good (new formula convention) ─
            # < -10%: 73-87% WR  |  -10 to 0: 71%  |  0-10: caution  |  >+10: hard avoid
            {'if': {'filter_query': '{_avl_num} <= -10', 'column_id': 'Avg vs Line'},
             'backgroundColor': 'rgba(45,212,191,0.14)', 'color': '#2dd4bf'},
            {'if': {'filter_query': '{_avl_num} > -10 && {_avl_num} <= 0', 'column_id': 'Avg vs Line'},
             'backgroundColor': 'rgba(45,212,191,0.08)', 'color': '#7dd3fc'},
            {'if': {'filter_query': '{_avl_num} > 0 && {_avl_num} <= 10', 'column_id': 'Avg vs Line'},
             'backgroundColor': 'rgba(249,116,75,0.10)', 'color': '#f9744b'},
            {'if': {'filter_query': '{_avl_num} > 10', 'column_id': 'Avg vs Line'},
             'backgroundColor': 'rgba(248,113,113,0.12)', 'color': '#f87171'},
            # ── LC: perfectly monotonic signal ────────────────────────────
            # >= 60%: 78-86% WR  |  50-60%: 65%  |  40-50%: 57%  |  <40%: 48%
            {'if': {'filter_query': '{_cons_num} >= 60', 'column_id': 'Line Consistency'},
             'backgroundColor': 'rgba(45,212,191,0.14)', 'color': '#2dd4bf'},
            {'if': {'filter_query': '{_cons_num} >= 50 && {_cons_num} < 60', 'column_id': 'Line Consistency'},
             'backgroundColor': 'rgba(45,212,191,0.08)', 'color': '#7dd3fc'},
            {'if': {'filter_query': '{_cons_num} >= 40 && {_cons_num} < 50', 'column_id': 'Line Consistency'},
             'backgroundColor': 'rgba(249,116,75,0.10)', 'color': '#f9744b'},
            {'if': {'filter_query': '{_cons_num} < 40', 'column_id': 'Line Consistency'},
             'backgroundColor': 'rgba(248,113,113,0.12)', 'color': '#f87171'},
            # ── Weather ───────────────────────────────────────────────────
            {'if': {'filter_query': '{Weather} contains "Strong"', 'column_id': 'Weather'},
             'backgroundColor': 'rgba(45,212,191,0.12)', 'color': '#2dd4bf'},
            {'if': {'filter_query': '{Weather} contains "Medium"', 'column_id': 'Weather'},
             'backgroundColor': 'rgba(45,212,191,0.07)', 'color': '#7dd3fc'},
            # ── Travel ────────────────────────────────────────────────────
            {'if': {'filter_query': '{Travel Fatigue} contains "Long"', 'column_id': 'Travel Fatigue'},
             'backgroundColor': 'rgba(249,116,75,0.12)', 'color': '#f9744b'},
            # ── Dim inactive rows (LAST — overrides all column-specific colours) ─
            {'if': {'filter_query': '{_active} = 0'},
             'color': 'rgba(225, 217, 207, 0.2)',
             'backgroundColor': 'rgba(9, 29, 38, 0.3)'},
        ]

        # Pre-compute selected_row_ids for excluded legs (stable across sorts)
        excluded_row_ids = [
            row['id'] for row in table_upcoming.to_dict('records')
            if row.get('id', '') in excluded_leg_ids
        ]

        legs_table = dash_table.DataTable(
            id='legs-table',
            data=table_upcoming.to_dict('records'),
            columns=dt_cols,
            hidden_columns=['_active', '_wr_pct', '_avl_num', '_cons_num', 'id'],
            sort_action='native',
            sort_by=[],
            page_action='none',
            style_table={
                'overflowX': 'auto',
                'borderRadius': '6px',
                'border': f'1px solid {D_BORDER}',
                'marginBottom': '8px',
            },
            style_header={
                'backgroundColor': D_CARD2,
                'color': D_MUT,
                'fontWeight': '700',
                'fontSize': '11px',
                'textTransform': 'uppercase',
                'letterSpacing': '0.08em',
                'borderBottom': f'1px solid {D_BORDER}',
                'fontFamily': FONT,
                'whiteSpace': 'nowrap',
                'cursor': 'pointer',
            },
            style_cell={
                'backgroundColor': D_CARD,
                'color': D_TEXT,
                'fontSize': '12px',
                'fontFamily': FONT,
                'padding': '7px 10px',
                'border': f'1px solid {D_BORDER}22',
                'whiteSpace': 'nowrap',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_cell_conditional=[
                {'if': {'column_id': '_excl'},           'minWidth': '24px',  'maxWidth': '24px', 'width': '24px'},
                {'if': {'column_id': 'Stat'},            'minWidth': '65px',  'maxWidth': '65px'},
                {'if': {'column_id': 'Player'},          'minWidth': '130px', 'maxWidth': '180px'},
                {'if': {'column_id': 'Team'},            'minWidth': '48px',  'maxWidth': '48px'},
                {'if': {'column_id': 'Opponent'},        'minWidth': '48px',  'maxWidth': '48px'},
                {'if': {'column_id': 'Position'},        'minWidth': '50px',  'maxWidth': '50px'},
                {'if': {'column_id': 'Bet Priority'},   'minWidth': '42px',  'maxWidth': '42px'},
                {'if': {'column_id': 'Line'},            'minWidth': '48px',  'maxWidth': '48px'},
                {'if': {'column_id': 'Avg vs Line'},     'minWidth': '52px',  'maxWidth': '52px'},
                {'if': {'column_id': 'Line Consistency'},'minWidth': '52px',  'maxWidth': '60px'},
                {'if': {'column_id': 'DvP'},             'minWidth': '130px', 'maxWidth': '160px'},
                {'if': {'column_id': 'Weather'},         'minWidth': '80px',  'maxWidth': '110px'},
                {'if': {'column_id': 'Travel Fatigue'},  'minWidth': '68px',  'maxWidth': '80px'},
                {'if': {'column_id': 'Hist WR'},         'minWidth': '68px',  'maxWidth': '75px'},
                {'if': {'column_id': 'WR Range'},        'minWidth': '65px',  'maxWidth': '75px'},
                {'if': {'column_id': 'Confidence'},      'minWidth': '52px',  'maxWidth': '52px'},
            ],
            style_data_conditional=cond_styles,
        )

    # ── Build hedge picks ──────────────────────────────────────────────────────
    bets_list = active_upcoming.to_dict('records') if not active_upcoming.empty else []

    # Build pre_placed counts from checked_ids store (for placed summary display)
    pre_placed: dict = {}
    for pair_id in (checked_ids or []):
        parts = pair_id.split('|')
        if len(parts) == 2:
            pre_placed[parts[0]] = pre_placed.get(parts[0], 0) + 1
            pre_placed[parts[1]] = pre_placed.get(parts[1], 0) + 1
    # Sort by hierarchy: Tier (T0 > T1 > T2) → LC% descending → AvL ascending
    def _quality_key(row):
        tier = str(row.get('Strategy', ''))
        tier_order = {'T0': 0, 'T1': 1, 'T2': 2}.get(tier, 3)
        lc_s = str(row.get('Line Consistency', ''))
        try:
            lc = float(lc_s.replace('%', '').strip())
            if lc <= 1.0: lc *= 100
        except:
            lc = 0.0
        avl_s = str(row.get('Avg vs Line', ''))
        try:    avl = float(avl_s.replace('%', '').replace('+', '').strip())
        except: avl = 0.0
        return (tier_order, -lc, avl)

    sorted_legs = sorted(bets_list, key=_quality_key)

    # Enforce max-2-per-team when selecting the best-7 for C(7,6)
    # Greedy: walk down WR rank, skip a team once it hits the cap.
    # If we still can't fill 7 after exhausting uncapped legs, relax and fill from overflow.
    def _select_team_capped(legs, target_n, cap=2):
        counts, selected, overflow = {}, [], []
        for leg in legs:
            team = leg.get('Team', '')
            if counts.get(team, 0) < cap:
                selected.append(leg)
                counts[team] = counts.get(team, 0) + 1
            else:
                overflow.append(leg)
            if len(selected) == target_n:
                break
        # Relax cap if short
        for leg in overflow:
            if len(selected) >= target_n:
                break
            selected.append(leg)
        return selected

    if len(sorted_legs) >= 7:
        best7 = _select_team_capped(sorted_legs, 7, cap=2)
        # Jackpot gets all legs: best7 first (team-balanced), then the rest in WR order
        best7_players = {l['Player'] for l in best7}
        rest  = [l for l in sorted_legs if l['Player'] not in best7_players]
        combined_legs = best7 + rest
    else:
        combined_legs = sorted_legs

    fmt_name, payout_desc, hedge_picks, hedge_meta, jackpot_info = build_hedge_picks(combined_legs)

    # Track how many picks each player appears in (combos + jackpot)
    leg_reuse = {}
    for pick in hedge_picks:
        for leg in pick:
            leg_reuse[leg['Player']] = leg_reuse.get(leg['Player'], 0) + 1
    if jackpot_info:
        for leg in jackpot_info['pick']:
            leg_reuse[leg['Player']] = leg_reuse.get(leg['Player'], 0) + 1

    # ── Hedge picks display ────────────────────────────────────────────────────
    SKIP = (fmt_name == "SKIP")

    # Format banner
    format_banner = html.Div([
        html.Span(fmt_name, style={
            "color": "#2dd4bf", "fontWeight": "700", "fontSize": "12px",
            "fontFamily": MONO, "marginRight": "16px",
        }),
        html.Span(payout_desc, style={
            "color": D_MUT, "fontSize": "10px", "fontFamily": MONO,
        }),
    ], style={
        "background": "rgba(45,212,191,0.07)", "border": "1px solid rgba(45,212,191,0.2)",
        "borderRadius": "6px", "padding": "8px 14px", "marginBottom": "12px",
    }) if not SKIP else html.Div(
        payout_desc,
        style={"color": "#f87171", "fontSize": "11px", "fontFamily": MONO,
               "padding": "10px", "background": "rgba(248,113,113,0.07)",
               "borderRadius": "6px", "border": "1px solid rgba(248,113,113,0.2)",
               "marginBottom": "12px"}
    )

    # Individual pick cards
    pick_cards = []
    for i, pick in enumerate(hedge_picks, 1):
        leg_rows = []
        for leg in sorted(pick, key=lambda l: (l.get('Team', ''), l.get('Player', ''))):
            leg_rows.append(html.Tr([
                html.Td(leg.get('Player',''), style={"color": D_TEXT, "fontSize": "11px",
                                                      "fontFamily": MONO, "padding": "4px 8px",
                                                      "whiteSpace": "nowrap"}),
                html.Td(leg.get('Team',''), style={"color": D_MUT, "fontSize": "10px",
                                                    "fontFamily": MONO, "padding": "4px 8px"}),
                html.Td(f"vs {leg.get('Opponent','')}", style={"color": D_FADED, "fontSize": "10px",
                                                                "fontFamily": MONO, "padding": "4px 8px"}),
                html.Td(leg.get('Position',''), style={"color": D_MUT, "fontSize": "10px",
                                                        "fontFamily": MONO, "padding": "4px 8px"}),
                html.Td(f"Line {leg.get('Line','')}", style={"color": "#7dd3fc", "fontSize": "10px",
                                                              "fontFamily": MONO, "padding": "4px 8px",
                                                              "fontWeight": "600"}),
            ]))

        # ── Concentration indicators ──────────────────────────────────────────
        from collections import Counter as _Counter

        def _game_key(leg):
            t, o = leg.get('Team', ''), leg.get('Opponent', '')
            return f"{min(t,o)} v {max(t,o)}"

        team_counts = _Counter(leg.get('Team','') for leg in pick)
        game_counts = _Counter(_game_key(leg) for leg in pick)

        max_same      = max(team_counts.values()) if team_counts else 0
        max_same_game = max(game_counts.values()) if game_counts else 0

        # Team concentration
        if max_same >= 4:
            conc_col  = "#f87171"
            conc_text = f"⚠ {max_same} same team"
        elif max_same == 3:
            conc_col  = "#fbbf24"
            conc_text = f"~ {max_same} same team"
        else:
            conc_col  = "#2dd4bf"
            conc_text = f"✓ team ≤{max_same}"

        # Game concentration
        if max_same_game >= 4:
            game_conc_col  = "#f87171"
            game_conc_text = f"⚠ {max_same_game} same game — correlated"
        elif max_same_game == 3:
            game_conc_col  = "#fbbf24"
            game_conc_text = f"~ {max_same_game} same game"
        else:
            game_conc_col  = "#2dd4bf"
            game_conc_text = f"✓ game ≤{max_same_game}"

        worst_col = "#f87171" if (max_same >= 4 or max_same_game >= 4) else \
                    "#fbbf24" if (max_same >= 3 or max_same_game >= 3) else "#2dd4bf"

        team_badges = [
            html.Span(f"{team} ×{cnt}", style={
                "background": "rgba(225,217,207,0.06)", "borderRadius": "3px",
                "padding": "1px 6px", "fontSize": "9px", "fontFamily": MONO,
                "color": D_MUT, "marginRight": "4px",
            })
            for team, cnt in sorted(team_counts.items(), key=lambda x: -x[1])
        ]

        pick_cards.append(html.Div([
            html.Div([
                html.Span(f"Pick {i}", style={
                    "color": D_ACCENT, "fontSize": "10px", "fontWeight": "700",
                    "fontFamily": MONO, "letterSpacing": "0.06em", "marginRight": "12px",
                }),
                html.Span(conc_text, style={
                    "fontSize": "9px", "fontFamily": MONO, "color": conc_col,
                    "marginRight": "10px",
                }),
                html.Span("·", style={"color": D_MUT, "marginRight": "10px", "fontSize": "9px"}),
                html.Span(game_conc_text, style={
                    "fontSize": "9px", "fontFamily": MONO, "color": game_conc_col,
                    "marginRight": "10px",
                }),
                *team_badges,
            ], style={"display": "flex", "alignItems": "center",
                      "marginBottom": "6px", "flexWrap": "wrap"}),
            html.Table(html.Tbody(leg_rows),
                       style={"borderCollapse": "collapse", "width": "100%"}),
        ], style={
            "background": D_CARD2,
            "border": f"1px solid {worst_col}55" if (max_same >= 3 or max_same_game >= 3) else f"1px solid {D_BORDER}",
            "borderRadius": "6px", "padding": "10px 12px", "marginBottom": "8px",
        }))

    picks_section = html.Div([
        format_banner,
        html.Div(pick_cards) if pick_cards else html.Div(),
    ])

    # ── Jackpot section (n >= 7) ──────────────────────────────────────────────
    if jackpot_info:
        jp_leg_rows = []
        for leg in sorted(jackpot_info['pick'], key=lambda l: (l.get('Team', ''), l.get('Player', ''))):
            jp_leg_rows.append(html.Tr([
                html.Td(leg.get('Player',''), style={"color": D_TEXT, "fontSize": "11px",
                                                      "fontFamily": MONO, "padding": "4px 8px",
                                                      "whiteSpace": "nowrap"}),
                html.Td(leg.get('Team',''), style={"color": D_MUT, "fontSize": "10px",
                                                    "fontFamily": MONO, "padding": "4px 8px"}),
                html.Td(f"vs {leg.get('Opponent','')}", style={"color": D_FADED, "fontSize": "10px",
                                                                "fontFamily": MONO, "padding": "4px 8px"}),
                html.Td(leg.get('Position',''), style={"color": D_MUT, "fontSize": "10px",
                                                        "fontFamily": MONO, "padding": "4px 8px"}),
                html.Td(f"Line {leg.get('Line','')}", style={"color": "#7dd3fc", "fontSize": "10px",
                                                              "fontFamily": MONO, "padding": "4px 8px",
                                                              "fontWeight": "600"}),
            ]))

        # Concentration for jackpot pick
        from collections import Counter as _Counter

        def _game_key_jp(leg):
            t, o = leg.get('Team', ''), leg.get('Opponent', '')
            return f"{min(t,o)} v {max(t,o)}"

        jp_team_counts = _Counter(leg.get('Team','') for leg in jackpot_info['pick'])
        jp_game_counts = _Counter(_game_key_jp(leg) for leg in jackpot_info['pick'])

        jp_max_same      = max(jp_team_counts.values()) if jp_team_counts else 0
        jp_max_same_game = max(jp_game_counts.values()) if jp_game_counts else 0

        if jp_max_same >= 4:
            jp_conc_col  = "#f87171"
            jp_conc_text = f"⚠ {jp_max_same} same team"
        elif jp_max_same == 3:
            jp_conc_col  = "#fbbf24"
            jp_conc_text = f"~ {jp_max_same} same team"
        else:
            jp_conc_col  = "#2dd4bf"
            jp_conc_text = f"✓ team ≤{jp_max_same}"

        if jp_max_same_game >= 4:
            jp_game_conc_col  = "#f87171"
            jp_game_conc_text = f"⚠ {jp_max_same_game} same game — correlated"
        elif jp_max_same_game == 3:
            jp_game_conc_col  = "#fbbf24"
            jp_game_conc_text = f"~ {jp_max_same_game} same game"
        else:
            jp_game_conc_col  = "#2dd4bf"
            jp_game_conc_text = f"✓ game ≤{jp_max_same_game}"

        jp_team_badges = [
            html.Span(f"{team} ×{cnt}", style={
                "background": "rgba(225,217,207,0.06)", "borderRadius": "3px",
                "padding": "1px 6px", "fontSize": "9px", "fontFamily": MONO,
                "color": D_MUT, "marginRight": "4px",
            })
            for team, cnt in sorted(jp_team_counts.items(), key=lambda x: -x[1])
        ]

        jackpot_section = html.Div([
            html.Div([
                html.Span(jackpot_info['fmt'], style={
                    "color": "#fbbf24", "fontWeight": "700", "fontSize": "12px",
                    "fontFamily": MONO, "marginRight": "16px",
                }),
                html.Span(jackpot_info['desc'], style={
                    "color": D_MUT, "fontSize": "10px", "fontFamily": MONO,
                }),
            ], style={
                "background": "rgba(251,191,36,0.07)", "border": "1px solid rgba(251,191,36,0.25)",
                "borderRadius": "6px", "padding": "8px 14px", "marginBottom": "10px",
            }),
            html.Div([
                html.Div([
                    html.Span("1 ENTRY", style={
                        "color": "#fbbf24", "fontSize": "10px", "fontWeight": "700",
                        "fontFamily": MONO, "letterSpacing": "0.06em", "marginRight": "12px",
                    }),
                    html.Span(jp_conc_text, style={
                        "fontSize": "9px", "fontFamily": MONO, "color": jp_conc_col,
                        "marginRight": "10px",
                    }),
                    html.Span("·", style={"color": D_MUT, "marginRight": "10px", "fontSize": "9px"}),
                    html.Span(jp_game_conc_text, style={
                        "fontSize": "9px", "fontFamily": MONO, "color": jp_game_conc_col,
                        "marginRight": "10px",
                    }),
                    *jp_team_badges,
                ], style={"display": "flex", "alignItems": "center",
                          "marginBottom": "6px", "flexWrap": "wrap"}),
                html.Table(html.Tbody(jp_leg_rows),
                           style={"borderCollapse": "collapse", "width": "100%"}),
            ], style={
                "background": D_CARD2, "border": "1px solid rgba(251,191,36,0.2)",
                "borderRadius": "6px", "padding": "10px 12px",
            }),
        ], style={"marginTop": "16px"})
    else:
        jackpot_section = html.Div()

    pairings_section = html.Div([picks_section, jackpot_section])

    # ── Leg exposure chart (picks per player) ────────────────────────────────
    freq: dict = leg_reuse  # already built above

    sorted_players = sorted(freq, key=lambda x: freq[x], reverse=True)
    bar_vals   = [freq[p] for p in sorted_players]
    bar_labels = sorted_players
    bar_colors = [D_ACCENT] * len(sorted_players)

    freq_fig = go.Figure(go.Bar(
        x=bar_vals,
        y=bar_labels,
        orientation='h',
        marker_color=bar_colors,
        text=bar_vals,
        textposition='outside',
        textfont=dict(size=11, color="#e1d9cf", family="JetBrains Mono"),
    ))
    freq_fig.update_layout(
        paper_bgcolor="rgba(18,77,84,0.45)",
        plot_bgcolor="rgba(9,29,38,0.4)",
        margin=dict(l=0, r=20, t=10, b=10),
        height=max(200, len(sorted_players) * 28),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[0, max(bar_vals or [1]) + 1.5]),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#e1d9cf", family="JetBrains Mono"),
                   autorange="reversed"),
        showlegend=False,
    )

    leg_freq_chart = html.Div([
        html.Div("LEG EXPOSURE", style={
            "fontSize": "9px", "fontWeight": "500", "color": D_MUT,
            "textTransform": "uppercase", "letterSpacing": "0.1em",
            "fontFamily": FONT, "marginBottom": "8px",
        }),
        dcc.Graph(figure=freq_fig, config={"displayModeBar": False},
                  style={"width": "100%"}),
    ], style={
        "background": D_CARD, "borderRadius": "6px", "padding": "14px 16px",
        "border": f"1px solid {D_BORDER}",
    })

    # Serialise hedge picks for the callback store
    def _serialise_legs(pick):
        return [{'Player': l.get('Player',''), 'Team': l.get('Team',''),
                 'Opponent': l.get('Opponent',''), 'Line': l.get('Line',''),
                 'Position': l.get('Position',''), 'Strategy': l.get('Bet Priority','')}
                for l in pick]

    pairings_store_data = {
        'hedge_picks':   [_serialise_legs(p) for p in hedge_picks],
        'hedge_format':  fmt_name,
        'hedge_meta':    hedge_meta,
        'jackpot_pick':  _serialise_legs(jackpot_info['pick']) if jackpot_info else [],
        'jackpot_format': jackpot_info['fmt'] if jackpot_info else '',
        'jackpot_meta':   jackpot_info['meta'] if jackpot_info else {},
    }

    def _tag(text, color, bg):
        return html.Span(text, style={
            "background": bg, "color": color, "borderRadius": "3px",
            "padding": "2px 7px", "fontSize": "10px", "fontWeight": "700",
            "fontFamily": FONT, "marginRight": "8px", "letterSpacing": "0.05em",
        })

    # ── Hard filters panel ────────────────────────────────────────────────────
    FILTER_OK   = "#2dd4bf"
    FILTER_WARN = "#f87171"
    FILTER_DIM  = "rgba(225,217,207,0.35)"

    def _filter_pill(label, note=None):
        return html.Div([
            html.Span("✓ ", style={"color": FILTER_OK, "fontWeight": "700", "fontSize": "10px"}),
            html.Span(label, style={"color": D_TEXT, "fontSize": "11px", "fontFamily": MONO,
                                    "fontWeight": "600"}),
            html.Span(f"  {note}", style={"color": FILTER_DIM, "fontSize": "10px",
                                          "fontFamily": MONO}) if note else None,
        ], style={"display": "flex", "alignItems": "center", "gap": "2px",
                  "padding": "5px 10px", "borderRadius": "5px",
                  "background": "rgba(45,212,191,0.05)",
                  "border": f"1px solid rgba(45,212,191,0.15)"})

    staking_legend = html.Div([
        html.Div("HARD FILTERS", style={
            "fontSize": "9px", "fontWeight": "600", "color": FILTER_DIM,
            "letterSpacing": "0.12em", "textTransform": "uppercase",
            "fontFamily": MONO, "marginBottom": "8px",
        }),
        html.Div([
            _filter_pill("Line ≥ 3.0"),
            _filter_pill("Not vs GCS", "33% total, 13% at LC<50%"),
            _filter_pill("Not MedF"),
            _filter_pill("Not GEE home", "GMHBA — 10% at LC 50-60%"),
            _filter_pill("No Rain"),
            _filter_pill("AvL > +10% avoid"),
            _filter_pill("T2: Neutral DvP skip", "100% at LC≥60% — T0/T1 unaffected"),
            _filter_pill("ADE T2 block", "12% at LC 50-60% — T0/T1 OK"),
            _filter_pill("⭐ WCE edge", "88% at LC 60-70%, 86% at LC≥70%"),
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "6px"}),
    ], style={
        "marginBottom": "14px", "padding": "12px 14px",
        "background": D_CARD2, "borderRadius": "6px",
        "border": f"1px solid {D_BORDER}",
    })

    # ── Strategy win rate panel ───────────────────────────────────────────────
    def _strategy_wr_panel():
        hist = _load_bet_history()
        if hist.empty:
            return html.Div()
        hist = hist.copy()
        hist['Round'] = pd.to_numeric(hist['Round'], errors='coerce')
        hist['Year']  = pd.to_numeric(hist['Year'],  errors='coerce')
        h2026 = hist[hist['Year'] == 2026].copy()
        if h2026.empty:
            return html.Div()

        def _wr_col(wr):
            return "#2dd4bf" if wr >= 65 else ("#f9744b" if wr >= 55 else "#f87171")

        # ── Tier KPI cards ────────────────────────────────────────────────────
        tier_cards = []
        tier_cfg = [
            ("T0", 87.0, "#2dd4bf"),
            ("T1", 68.0, "#f9744b"),
            ("T2", 60.0, "#f87171"),
        ]
        for tier, target, t_col in tier_cfg:
            sub = h2026[h2026['Strategy'] == tier]
            n   = len(sub)
            w   = int(sub['win'].sum()) if n else 0
            wr  = w / n * 100 if n else 0
            col = _wr_col(wr) if n else D_FADED
            delta = wr - target
            delta_str = f"{delta:+.1f}% vs {target:.0f}% target" if n else "no data"
            tier_cards.append(html.Div([
                html.Div([
                    html.Span(tier, style={"fontSize": "11px", "fontWeight": "700",
                                           "color": t_col, "fontFamily": MONO,
                                           "marginRight": "6px"}),
                    html.Span(f"{wr:.1f}%" if n else "—", style={
                        "fontSize": "20px", "fontWeight": "700",
                        "color": col, "fontFamily": MONO}),
                ], style={"display": "flex", "alignItems": "baseline", "gap": "4px"}),
                html.Div(f"{w}W / {n-w}L  ({n} legs)", style={
                    "fontSize": "10px", "color": D_FADED, "fontFamily": MONO, "marginTop": "2px"}),
                html.Div(delta_str, style={
                    "fontSize": "9px", "fontFamily": MONO, "marginTop": "2px",
                    "color": "#2dd4bf" if delta >= 0 else "#f87171"}),
            ], style={"padding": "8px 12px", "background": D_CARD2, "borderRadius": "6px",
                      "border": f"1px solid {t_col}33", "flex": "1", "minWidth": "110px"}))

        # Overall — tiered legs only (T0+T1+T2)
        tiered   = h2026[h2026['Strategy'].isin(['T0', 'T1', 'T2'])]
        total    = len(tiered)
        wins     = int(tiered['win'].sum())
        wr       = wins / total * 100 if total else 0
        tier_cards.append(html.Div([
            html.Div([
                html.Span("ALL", style={"fontSize": "11px", "fontWeight": "700",
                                        "color": D_MUT, "fontFamily": MONO, "marginRight": "6px"}),
                html.Span(f"{wr:.1f}%", style={"fontSize": "20px", "fontWeight": "700",
                                                "color": _wr_col(wr), "fontFamily": MONO}),
            ], style={"display": "flex", "alignItems": "baseline", "gap": "4px"}),
            html.Div(f"{wins}W / {total-wins}L  ({total} legs)", style={
                "fontSize": "10px", "color": D_FADED, "fontFamily": MONO, "marginTop": "2px"}),
            html.Div("T0 + T1 + T2 only", style={
                "fontSize": "9px", "color": D_FADED, "fontFamily": MONO, "marginTop": "2px"}),
        ], style={"padding": "8px 12px", "background": D_CARD2, "borderRadius": "6px",
                  "border": f"1px solid {D_BORDER}", "flex": "1", "minWidth": "110px"}))

        # ── Per-round breakdown with tier mini-bars (tiered legs only) ──────────
        round_rows = []
        for rnd in sorted(tiered['Round'].dropna().unique()):
            r   = tiered[tiered['Round'] == rnd]
            rw  = int(r['win'].sum())
            rn  = len(r)
            rwr = rw / rn * 100 if rn else 0
            rc  = _wr_col(rwr)

            # per-tier counts for this round
            tier_bits = []
            for tier, _, tc in tier_cfg:
                tn = len(r[r['Strategy'] == tier])
                if tn:
                    tier_bits.append(html.Span(f"{tier}:{tn}", style={
                        "fontSize": "8px", "fontFamily": MONO, "color": tc,
                        "background": f"{tc}15", "borderRadius": "2px",
                        "padding": "0px 4px", "marginLeft": "3px",
                    }))

            round_rows.append(html.Div([
                html.Span(f"R{int(rnd)}", style={"color": D_MUT, "fontSize": "10px",
                                                  "fontFamily": MONO, "minWidth": "22px"}),
                html.Div(style={
                    "height": "8px", "borderRadius": "3px",
                    "background": rc, "opacity": "0.75",
                    "width": f"{rwr:.0f}%", "minWidth": "4px",
                    "flex": "1", "maxWidth": "100px",
                }),
                html.Span(f"{rwr:.0f}% ({rn})", style={"color": rc, "fontSize": "10px",
                                                         "fontFamily": MONO, "minWidth": "58px",
                                                         "textAlign": "right"}),
                *tier_bits,
            ], style={"display": "flex", "alignItems": "center", "gap": "6px",
                      "marginBottom": "4px", "flexWrap": "wrap"}))

        # ── Per-tier by round table ───────────────────────────────────────────
        tier_rnd_rows = []
        rounds = sorted(tiered['Round'].dropna().unique())
        for tier, _, tc in tier_cfg:
            cols = []
            for rnd in rounds:
                sub = h2026[(h2026['Round'] == rnd) & (h2026['Strategy'] == tier)]
                n   = len(sub)
                w   = int(sub['win'].sum()) if n else 0
                wr_t = w / n * 100 if n else None
                cols.append(html.Td(
                    f"{wr_t:.0f}%" if wr_t is not None else "—",
                    style={"padding": "3px 8px", "fontSize": "10px", "fontFamily": MONO,
                           "textAlign": "center",
                           "color": _wr_col(wr_t) if wr_t is not None else D_FADED}
                ))
            tier_rnd_rows.append(html.Tr([
                html.Td(tier, style={"padding": "3px 8px", "fontSize": "10px",
                                     "fontFamily": MONO, "color": tc, "fontWeight": "700"}),
                *cols,
            ]))

        tier_rnd_table = html.Table([
            html.Thead(html.Tr([
                html.Th("", style={"padding": "3px 8px", "fontSize": "9px",
                                   "fontFamily": MONO, "color": D_FADED}),
                *[html.Th(f"R{int(r)}", style={"padding": "3px 8px", "fontSize": "9px",
                                               "fontFamily": MONO, "color": D_FADED,
                                               "textAlign": "center"})
                  for r in rounds],
            ])),
            html.Tbody(tier_rnd_rows),
        ], style={"borderCollapse": "collapse", "width": "100%", "marginTop": "8px"})

        cached_at = _BET_HISTORY.get('cached_at')
        ts_str = f"  ·  as of {cached_at.strftime('%H:%M')}" if cached_at else ""

        return html.Div([
            html.Div([
                html.Span("STRATEGY WIN RATE · 2026", style={
                    "fontSize": "9px", "fontWeight": "600", "color": D_MUT,
                    "letterSpacing": "0.12em", "textTransform": "uppercase",
                    "fontFamily": MONO,
                }),
                html.Span(ts_str, style={
                    "fontSize": "9px", "color": D_FADED, "fontFamily": MONO,
                }),
            ], style={"display": "flex", "alignItems": "center",
                      "gap": "4px", "marginBottom": "10px"}),
            html.Div(tier_cards, style={"display": "flex", "gap": "8px",
                                        "marginBottom": "12px", "flexWrap": "wrap"}),
            html.Div([
                html.Div([
                    html.Div("BY ROUND", style={"fontSize": "9px", "color": D_FADED,
                                                 "fontFamily": MONO, "letterSpacing": "0.1em",
                                                 "marginBottom": "6px"}),
                    *round_rows,
                ], style={"flex": "1", "minWidth": "200px"}),
                html.Div([
                    html.Div("BY TIER × ROUND", style={"fontSize": "9px", "color": D_FADED,
                                                         "fontFamily": MONO, "letterSpacing": "0.1em",
                                                         "marginBottom": "2px"}),
                    tier_rnd_table,
                ], style={"flex": "1", "minWidth": "200px"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        ], style={
            "marginBottom": "14px", "padding": "12px 14px",
            "background": D_CARD2, "borderRadius": "6px",
            "border": f"1px solid {D_BORDER}",
        })

    strategy_wr_panel = _strategy_wr_panel()

    # ── Placed capacity summary ───────────────────────────────────────────────
    placed_summary = html.Div()
    if pre_placed:
        n_picks = len(hedge_picks) if hedge_picks else 1

        cards = []
        for player, count in sorted(pre_placed.items(), key=lambda x: -x[1]):
            remaining = max(0, n_picks - count)
            full      = remaining == 0
            color     = "#f87171" if full else ("#f9744b" if remaining == 1 else "#2dd4bf")
            cards.append(html.Div([
                html.Div(player, style={"fontWeight": "600", "fontSize": "11px",
                                        "color": D_TEXT, "fontFamily": FONT}),
                html.Div(f"{count}/{n_picks} picks · {remaining} remaining",
                         style={"fontSize": "10px", "color": color, "fontFamily": FONT,
                                "marginTop": "2px"}),
            ], style={
                "background": D_CARD2, "border": f"1px solid {color}44",
                "borderRadius": "4px", "padding": "6px 10px",
                "borderLeft": f"3px solid {color}",
            }))

        placed_summary = html.Div([
            html.Div("PLACED THIS ROUND", style={
                "fontSize": "10px", "fontWeight": "700", "color": D_ACCENT,
                "textTransform": "uppercase", "letterSpacing": "0.1em",
                "fontFamily": FONT, "marginBottom": "8px",
            }),
            html.Div(cards, style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}),
        ], style={
            "background": D_CARD, "border": f"1px solid {D_BORDER}",
            "borderRadius": "6px", "padding": "12px 14px", "marginBottom": "14px",
        })

    # ── Played section ────────────────────────────────────────────────────────
    played_section = html.Div()
    if not played.empty:
        played_display = played[['Bet Priority', 'Stat', 'Player', 'Team', 'Opponent', 'Line']].copy()
        played_table = dash_table.DataTable(
            data=played_display.to_dict('records'),
            columns=[{"name": c, "id": c} for c in played_display.columns],
            style_table={"overflowX": "auto", "borderRadius": "6px", "border": f"1px solid {D_BORDER}"},
            style_cell={"textAlign": "left", "padding": "6px 10px",
                        "fontFamily": FONT, "fontSize": "10px",
                        "backgroundColor": "#0a0f1a", "color": D_MUT,
                        "border": f"1px solid {D_BORDER}22"},
            style_header={"backgroundColor": D_CARD2, "color": D_MUT,
                          "fontWeight": "700", "fontFamily": FONT, "fontSize": "11px",
                          "textTransform": "uppercase", "letterSpacing": "0.06em"},
        )
        played_section = html.Div([
            html.Div([
                html.Span("▸ ", style={"color": D_FADED}),
                f"ALREADY PLAYED  [{len(played)} leg{'s' if len(played) != 1 else ''}]",
            ], style={"fontSize": "11px", "fontWeight": "700", "color": D_FADED,
                      "textTransform": "uppercase", "letterSpacing": "0.08em",
                      "fontFamily": FONT, "marginBottom": "8px", "marginTop": "6px"}),
            played_table,
        ])

    n_active = len(active_upcoming) if not active_upcoming.empty else 0

    # ── Section helpers ───────────────────────────────────────────────────────
    def _sec_title(text, count=None, extra=None):
        return html.Div([
            html.Span(text, style={"color": D_MUT, "fontWeight": "500",
                                   "fontSize": "9px", "fontFamily": MONO,
                                   "letterSpacing": "0.1em", "textTransform": "uppercase"}),
            html.Span(f"  {count}", style={"color": D_FADED, "fontWeight": "400",
                                            "fontSize": "9px", "fontFamily": MONO}) if count is not None else None,
            html.Span(f"  ·  {extra}", style={"color": "#2dd4bf", "fontWeight": "500",
                                               "fontSize": "9px", "fontFamily": MONO}) if extra else None,
        ], style={"marginBottom": "12px"})

    def _section(content, mb="14px"):
        return html.Div(content, style={
            "background": D_CARD, "borderRadius": "10px",
            "border": f"1px solid {D_BORDER}",
            "padding": "16px 18px", "marginBottom": mb,
        })

    # ── Controls ──────────────────────────────────────────────────────────────
    input_style = {
        "background": D_CARD2, "color": D_TEXT, "border": f"1px solid {D_BORDER}",
        "borderRadius": "4px", "padding": "5px 8px",
        "fontFamily": FONT, "fontSize": "12px",
    }
    label_style = {"fontSize": "11px", "color": D_MUT, "fontFamily": FONT,
                   "textTransform": "uppercase", "letterSpacing": "0.06em", "marginBottom": "4px"}


    # ── Team filter chips ─────────────────────────────────────────────────────
    team_chips = html.Div([
        html.Div("Teams in pool", style=label_style),
        html.Div([
            html.Button(
                team,
                id={'type': 'team-chip', 'index': team},
                n_clicks=0,
                style={
                    "padding": "3px 10px", "fontSize": "11px", "fontFamily": FONT,
                    "borderRadius": "12px", "cursor": "pointer", "fontWeight": "600",
                    "border": f"1px solid {'rgba(248,113,113,0.4)' if team in excluded else 'rgba(45,212,191,0.4)'}",
                    "background": "rgba(248,113,113,0.08)" if team in excluded else "rgba(45,212,191,0.08)",
                    "color": "#f87171" if team in excluded else "#2dd4bf",
                    "textDecoration": "line-through" if team in excluded else "none",
                    "marginRight": "4px", "marginBottom": "4px",
                }
            ) for team in all_teams
        ], style={"display": "flex", "flexWrap": "wrap"}),
    ], style={"marginRight": "20px", "marginBottom": "14px"})

    html.Div(id='rr-top-n-input', style={"display": "none"})

    # ── Header ────────────────────────────────────────────────────────────────
    mb_header = html.Div([
        html.Div([
            html.Span("Multi Builder", style={"color": D_TEXT, "fontWeight": "700",
                                              "fontSize": "13px", "fontFamily": MONO,
                                              "letterSpacing": "0.04em"}),
            html.Span("  ·  Round Builder", style={"color": D_MUT, "fontSize": "11px",
                                                    "fontFamily": MONO}),
        ], style={"display": "flex", "alignItems": "baseline"}),
        html.Div([
            html.Span(f"{n_active} active", style={"color": D_MUT, "fontSize": "11px",
                                                    "fontFamily": MONO}),
            html.Span("  ·  ", style={"color": D_FADED, "fontSize": "11px"}),
            html.Span(f"{len(hedge_picks)} picks", style={"color": D_ACCENT,
                                                           "fontSize": "11px",
                                                           "fontFamily": MONO}),
            html.Span("  ·  ", style={"color": D_FADED, "fontSize": "11px"}),
            html.Span(fmt_name, style={"color": "#2dd4bf", "fontSize": "11px",
                                       "fontFamily": MONO}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "paddingBottom": "14px", "marginBottom": "16px",
        "borderBottom": f"1px solid {D_BORDER}",
    })

    layout = html.Div([
        mb_header,
        team_chips,
        staking_legend,
        strategy_wr_panel,
        _section([
            _sec_title("Active legs", n_active),
            html.Div(
                "Click × column to exclude/include a leg",
                style={"fontSize": "11px", "color": "rgba(225,217,207,0.4)",
                       "fontFamily": FONT, "marginBottom": "6px", "fontStyle": "italic"}
            ),
            legs_table,
        ]),
        _section([
            _sec_title("Hedge Picks", len(hedge_picks), extra=fmt_name if not SKIP else "skipped"),
            html.Div([
                html.Div(pairings_section, style={"flex": "3", "minWidth": "0"}),
                html.Div(leg_freq_chart,   style={"flex": "1", "minWidth": "180px"}),
            ], style={"display": "flex", "gap": "14px", "alignItems": "flex-start"}),
        ], mb="8px"),
        played_section,
    ], style={"background": D_BG, "padding": "18px", "borderRadius": "8px", "minHeight": "100vh"})

    return layout, pairings_store_data


# ===== DASH APP =====
app    = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                   suppress_callback_exceptions=True)
server = app.server

# Basic password protection — set DASHBOARD_PASSWORD env var to enable
import os as _os
from flask import request as _request, Response as _Response

@server.before_request
def _require_auth():
    pwd = _os.environ.get("DASHBOARD_PASSWORD", "")
    if not pwd:
        return None  # no password set — open access (local dev)
    auth = _request.authorization
    if auth and auth.username == "ryan" and auth.password == pwd:
        return None  # authenticated
    return _Response("Login required", 401,
                     {"WWW-Authenticate": 'Basic realm="AFL Dashboard"'})

processed_data_by_stat = {'disposals': None, 'marks': None, 'tackles': None}

# Stores simplified_dvp per stat type so edits can recalculate DvP
dvp_data_by_stat = {'disposals': {}, 'marks': {}, 'tackles': {}}

# Maps team abbr → game kickoff datetime (populated on data load)
fixtures_by_team = {}

# Tracks when each stat type was last pulled from Dabble
last_pulled_at = {'disposals': None, 'marks': None, 'tackles': None}
current_round_number   = None   # set when data loads, used by Sheets push

app.layout = dbc.Container([
    html.Div([
        html.Span("Tackle Unders", style={
            "fontFamily": "var(--mono, 'JetBrains Mono', monospace)",
            "fontSize": "13px", "fontWeight": "700", "color": "#e1d9cf",
            "letterSpacing": "0.04em",
        }),
        html.Span("  ·  AFL Dashboard", style={
            "fontFamily": "var(--mono, 'JetBrains Mono', monospace)",
            "fontSize": "11px", "color": "rgba(225, 217, 207, 0.45)",
        }),
    ], style={
        "padding": "14px 0 10px",
        "borderBottom": "1px solid rgba(225, 217, 207, 0.10)",
        "marginBottom": "14px",
    }),

    html.Div(id='loaded-data',        style={'display': 'none'}),
    html.Div(id='sheets-push-status', style={'display': 'none'}),
    dcc.Download(id="download-csv"),
    dcc.Store(id='pairings-store', data={}),
    dcc.Store(id='rr-top-n-store', storage_type='local', data=4),
    dcc.Store(id='placed-bets-store', storage_type='local', data=[]),
    dcc.Store(id='excluded-teams-store', storage_type='local', data=[]),
    dcc.Store(id='excluded-legs-store',  storage_type='local', data=[]),

    html.Div([
        dbc.Tabs([
            dbc.Tab(label="🎯 Multi Builder",     tab_id="tab-multi",        labelClassName="fw-bold text-success"),
            dbc.Tab(label="🧠 Analysis",          tab_id="tab-analysis",     labelClassName="fw-bold text-warning"),
        ], id="stat-tabs", active_tab="tab-multi"),
    ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "12px"}),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Button(
                        "🔄 Check for New Markets",
                        id="refresh-markets-button",
                        title="Re-scrapes Dabble for updated lines across all 3 stat types. Use this Thu–Sun when new markets open without restarting the dashboard.",
                        style={
                            "width": "100%", "padding": "7px 14px",
                            "background": "rgba(16, 41, 55, 0.8)", "color": "#f9744b",
                            "border": "1px solid rgba(249, 116, 75, 0.3)", "borderRadius": "6px",
                            "fontFamily": "var(--mono, 'JetBrains Mono', monospace)",
                            "fontSize": "11px", "fontWeight": "500", "cursor": "pointer",
                            "transition": "border-color 0.15s",
                        },
                    ),
                ], width=6),
                dbc.Col([
                    html.Button(
                        "⬇ Export CSV",
                        id="export-button",
                        style={
                            "width": "100%", "padding": "7px 14px",
                            "background": "rgba(16, 41, 55, 0.8)", "color": "rgba(225, 217, 207, 0.55)",
                            "border": "1px solid rgba(225, 217, 207, 0.10)", "borderRadius": "6px",
                            "fontFamily": "var(--mono, 'JetBrains Mono', monospace)",
                            "fontSize": "11px", "fontWeight": "500", "cursor": "pointer",
                            "transition": "border-color 0.15s",
                        },
                    ),
                ], width=6),
            ]),
            # status messages
            html.Div(id="refresh-markets-message", className="text-center mt-2 small"),
            html.Div(id="sheets-push-message",     className="text-center mt-1 small"),
        ], width=6),
    ], className="mb-3"),

    # ── Multi builder (shown for Multi Builder tab) ───────────────────────
    html.Div(id="multi-builder-content"),

    # ── Analysis (shown for Analysis tab) ─────────────────────────────────
    html.Div(id="analysis-content"),

    # ── Calibration (shown for Calibration tab) ────────────────────────────
    html.Div(id="calibration-content"),

], fluid=True, style={"background": "transparent", "minHeight": "100vh", "padding": "0 18px"})


# ── load data on startup ──────────────────────────────────────────────────────
@app.callback(
    Output('loaded-data',         'children'),
    Output('sheets-push-message', 'children'),
    Input('loaded-data',          'children'),
)
def load_data(data):
    global current_round_number
    if data is None:
        try:
            for stat_type in ['disposals', 'marks', 'tackles']:
                df = process_data_for_dashboard(stat_type)
                if df.empty:
                    df = pd.DataFrame([{
                        'Player': 'Test Player', 'Team': 'Test', 'Opponent': 'Test',
                        'Position': 'KeyF', 'Travel Fatigue': '✅ Neutral',
                        'Weather': '✅ Neutral', 'DvP': '✅ Neutral', 'Line': '25.5',
                        'Avg vs Line': '', 'Line Consistency': '',
                        'Bet Priority': '1',
                    }])
                processed_data_by_stat[stat_type] = df

            # infer round from the stats CSV
            try:
                raw = pd.read_csv("afl_player_stats.csv", skiprows=3).fillna(0)
                current_round_number = int(raw['round'].max()) + 1
            except Exception:
                current_round_number = 0

            # Record initial pull timestamps
            pulled_time = datetime.now().strftime('%H:%M:%S')
            for st in ['disposals', 'marks', 'tackles']:
                last_pulled_at[st] = pulled_time

            # Cache this round's lines so they survive past Dabble removing them
            save_round_cache(
                current_round_number,
                processed_data_by_stat.get('disposals'),
                processed_data_by_stat.get('marks'),
                processed_data_by_stat.get('tackles'),
            )

            # Push all cached rounds (current + any previously missed)
            _, push_msg = push_all_cached_rounds()
            return "Data loaded", push_msg
        except Exception as e:
            print(f"Load error: {e}")
            return "Error loading data", ""
    return data, ""


# ── Market refresh callback ───────────────────────────────────────────────────
# Tracks when the last successful Dabble scrape ran — used for cooldown
_last_refresh_time = None
REFRESH_COOLDOWN_SECONDS = 300   # 5 minutes — safe buffer for Dabble


@app.callback(
    Output('refresh-markets-message', 'children'),
    Output('loaded-data',             'children',  allow_duplicate=True),
    Output('sheets-push-message',     'children',  allow_duplicate=True),
    Input('refresh-markets-button',   'n_clicks'),
    prevent_initial_call=True,
)
def refresh_markets(n_clicks):
    """
    Re-scrapes Dabble for updated lines on all three stat types and
    rebuilds the processed dataframes in-place.

    Cooldown: minimum 5 minutes between scrapes to avoid rate limiting.
    If called within 5 minutes of the last pull, returns cached data
    with a message showing how long until the next refresh is allowed.
    """
    global _last_refresh_time

    if not n_clicks:
        return "", "Data loaded", ""

    # ── Cooldown check ────────────────────────────────────────────────────────
    if _last_refresh_time is not None:
        elapsed = (datetime.now() - _last_refresh_time).total_seconds()
        if elapsed < REFRESH_COOLDOWN_SECONDS:
            remaining = int(REFRESH_COOLDOWN_SECONDS - elapsed)
            mins = remaining // 60
            secs = remaining % 60
            wait_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            return (
                f"⏳ Lines are fresh — last pulled at "
                f"{_last_refresh_time.strftime('%H:%M:%S')}. "
                f"Next refresh available in {wait_str}.",
                "Data loaded",
                "",
            )

    results = []
    pulled_time = datetime.now().strftime('%H:%M:%S')

    for stat_type in ['disposals', 'marks', 'tackles']:
        try:
            existing = processed_data_by_stat.get(stat_type)
            if existing is None:
                results.append(f"{stat_type}: no base data")
                continue

            # Re-run pickem scrape on existing player list
            # We work from the existing df (already has travel/weather/dvp)
            # and just refresh lines + derived columns + bet flags
            df = existing.copy()

            # Strip old line-dependent columns so they get recalculated fresh
            for col in ['Line', 'Avg vs Line', 'Line Consistency', 'Bet Priority']:
                if col in df.columns:
                    df[col] = ''

            # Rename back to lowercase for the helper functions
            df_lower = df.rename(columns={v: k for k, v in {
                'player': 'Player', 'team': 'Team', 'opponent': 'Opponent',
                'position': 'Position', 'travel_fatigue': 'Travel Fatigue',
                'weather': 'Weather', 'dvp': 'DvP',
            }.items()})

            # Re-add pickem lines
            df_lower = add_pickem_lines_to_dataframe(df_lower, stat_type)
            df_lower = add_line_analysis_columns(df_lower, stat_type)
            df_lower = add_bet_flag_to_dataframe(df_lower, stat_type)

            # Rename back to display columns
            display = df_lower.rename(columns={
                'player': 'Player', 'team': 'Team', 'opponent': 'Opponent',
                'position': 'Position', 'travel_fatigue': 'Travel Fatigue',
                'weather': 'Weather', 'dvp': 'DvP',
                'Bet_Priority': 'Bet Priority',
            })

            # Keep only rows with a line
            display = display[display['Line'] != ''].copy()

            # Sort by bet priority
            def _sort_p(p):
                try: return int(p)
                except: return 999
            display['_s'] = display['Bet Priority'].apply(_sort_p)
            display = display.sort_values(['_s', 'Team']).drop('_s', axis=1)

            # Ensure column order matches original
            keep_cols = ['Player', 'Team', 'Opponent', 'Position',
                         'Travel Fatigue', 'Weather', 'DvP', 'Line',
                         'Avg vs Line', 'Line Consistency', 'Bet Priority']
            display = display[[c for c in keep_cols if c in display.columns]]

            processed_data_by_stat[stat_type] = display
            last_pulled_at[stat_type] = pulled_time

            new_markets = len(display)
            flagged     = (display['Bet Priority'].astype(str).str.strip() != '').sum()
            results.append(f"{stat_type}: {new_markets} players, {flagged} flagged")

        except Exception as e:
            print(f"Market refresh error ({stat_type}): {e}")
            results.append(f"{stat_type}: error — {e}")

    # Record successful refresh time for cooldown tracking
    _last_refresh_time = datetime.now()

    # Cache this round's freshly-scraped lines
    save_round_cache(
        current_round_number,
        processed_data_by_stat.get('disposals'),
        processed_data_by_stat.get('marks'),
        processed_data_by_stat.get('tackles'),
    )

    # Push all cached rounds (current + any previously missed)
    _, push_msg = push_all_cached_rounds()

    summary = " · ".join(results)
    return (
        f"✅ Markets refreshed at {pulled_time} — {summary}",
        "Data loaded",
        push_msg,
    )





def build_combined_layout():
    import math
    from math import comb

    BG     = "#0a0a0a"
    CARD   = "#111111"
    BORDER = "rgba(255,255,255,0.06)"
    TEXT   = "#f0f0f0"
    MUTED  = "rgba(240,240,240,0.5)"
    FADED  = "rgba(240,240,240,0.25)"
    WIN    = "#00d4aa"
    LOSS   = "#ff4d6d"
    AMBER  = "#f59e0b"
    MONO   = "var(--mono, 'JetBrains Mono', monospace)"

    # ── Load data ─────────────────────────────────────────────────────────────
    records = _load_raw_sheet()
    if records is None:
        return dbc.Alert("⚠️ Google Sheets not configured.", color="warning", className="mt-3")
    if not records:
        return dbc.Alert("No data in sheet yet.", color="info", className="mt-3")

    raw = pd.DataFrame(records)
    raw.columns = [c.strip() for c in raw.columns]
    if 'Type' in raw.columns:
        raw = raw[raw['Type'].astype(str).str.strip() == 'Tackle'].copy()
    else:
        raw = raw.copy()  # guarantee ownership so column additions always persist
    for col in ('Year', 'Round', 'Player', 'Team', 'Opponent', 'Position', 'W/L', 'Strategy',
                'Line', 'Avg vs Line', 'Line Consistency', 'DvP', 'Weather', 'Travel Fatigue',
                'Hist WR', 'Confidence', 'Subjective Avoid Reason', 'Actual'):
        if col not in raw.columns:
            raw[col] = ''
        raw[col] = raw[col].astype(str).str.strip()
    raw['year_num'] = pd.to_numeric(raw['Year'],  errors='coerce')
    raw['rnd_num']  = pd.to_numeric(raw['Round'], errors='coerce')
    raw['wl']       = pd.to_numeric(raw['W/L'],   errors='coerce')

    # Resolved rows only
    resolved = raw[raw['W/L'].isin(['1', '-1', '0', '1.0', '-1.0', '0.0'])].copy()
    resolved['win'] = (resolved['wl'] == 1).astype(int)

    def _p_avl(s):
        try:    return float(str(s).replace('%', '').replace('+', ''))
        except: return None

    def _p_line(s):
        try:    return float(str(s))
        except: return None

    def _p_dvp(s):
        for kw in ('Strong Unders', 'Moderate Unders', 'Slight Unders', 'Neutral',
                   'Slight Easy', 'Moderate Easy', 'Strong Easy'):
            if kw in str(s):
                return kw
        return 'Other'

    def _p_lc(s):
        try:
            v = float(str(s).replace('%', '').strip())
            return v * 100 if v <= 1.0 else v  # normalise 0-1 → 0-100 if needed
        except:
            return None

    resolved['_avl']  = resolved['Avg vs Line'].apply(_p_avl)
    resolved['_line'] = resolved['Line'].apply(_p_line)
    resolved['_dvp']  = resolved['DvP'].apply(_p_dvp)
    resolved['_pos']  = resolved['Position'].astype(str).str.strip()
    resolved['_lc']   = resolved['Line Consistency'].apply(_p_lc)

    # 4 analytical subsets: year × filter level
    df25_raw = resolved[resolved['year_num'] == 2025].copy()
    df26_raw = resolved[resolved['year_num'] == 2026].copy()
    df25_hf  = df25_raw[df25_raw.apply(_passes_bet_filters, axis=1)].copy()
    df26_hf  = df26_raw[df26_raw.apply(_passes_bet_filters, axis=1)].copy()

    # 2026 strategy performance (T0/T1/T2 + hard filters — the full qualifying gate)
    df26_strat   = df26_hf[df26_hf['Strategy'].isin(['T0', 'T1', 'T2'])].copy()
    rounds_avail = sorted(df26_strat['rnd_num'].dropna().astype(int).unique()) if not df26_strat.empty else []
    max_round    = rounds_avail[-1] if rounds_avail else "?"
    _w = (df26_strat['wl'] == 1).sum()
    _l = (df26_strat['wl'] == -1).sum()
    WR = _w / (_w + _l) if (_w + _l) > 0 else 0.689

    # Hist WR parsing for meta-calibration
    def _parse_hist(s):
        try:
            pct, n_part = str(s).split('%')
            return float(pct.strip()), int(n_part.strip().strip('()'))
        except:
            return None, 0

    for _df in [df25_hf, df26_hf]:
        _df['_hwr']   = _df['Hist WR'].astype(str).str.strip()
        _df['_hwr_p'] = _df['_hwr'].apply(lambda s: _parse_hist(s)[0])
        _df['_hwr_n'] = _df['_hwr'].apply(lambda s: _parse_hist(s)[1])

    # ── Style helpers ─────────────────────────────────────────────────────────
    TH   = {"padding": "7px 10px", "fontSize": "10px", "fontFamily": MONO,
             "color": MUTED, "fontWeight": "600", "textAlign": "right",
             "borderBottom": f"1px solid {BORDER}", "whiteSpace": "nowrap"}
    TH_L = {**TH, "textAlign": "left"}
    TD   = {"padding": "6px 10px", "fontSize": "11px", "fontFamily": MONO,
             "color": TEXT, "textAlign": "right",
             "borderBottom": "1px solid rgba(255,255,255,0.04)"}
    TD_L = {**TD, "textAlign": "left"}

    def _card(children, extra=None):
        s = {"background": CARD, "borderRadius": "10px", "padding": "18px 20px",
             "border": f"1px solid {BORDER}", "marginBottom": "16px"}
        if extra:
            s.update(extra)
        return html.Div(children, style=s)

    def _label(txt):
        return html.Div(txt, style={
            "color": MUTED, "fontWeight": "600", "fontSize": "9px",
            "fontFamily": MONO, "letterSpacing": "0.12em",
            "textTransform": "uppercase", "marginBottom": "10px",
        })

    def _se(p, n):
        if n < 2: return 0
        return math.sqrt(p / 100 * (1 - p / 100) / n) * 100

    def _wr_color(pct):
        if pct is None: return FADED
        if pct >= 65:   return WIN
        if pct >= 56:   return "#7dd3fc"
        if pct >= 50:   return AMBER
        return LOSS

    def _ev_color(ev):
        if ev >= 200: return WIN
        if ev >= 80:  return AMBER
        return TEXT

    def _bar(actual, predicted=None, width=100):
        filled = max(0, min(100, actual or 0))
        color  = _wr_color(actual)
        return html.Div(style={"display": "flex", "alignItems": "center"}, children=[
            html.Div(style={"position": "relative", "width": f"{width}px", "height": "6px",
                            "background": "rgba(255,255,255,0.08)", "borderRadius": "3px"}, children=[
                html.Div(style={"position": "absolute", "left": "0", "top": "0", "height": "6px",
                                "borderRadius": "3px", "width": f"{filled}%", "background": color}),
                *([] if predicted is None else [
                    html.Div(style={"position": "absolute", "top": "-2px",
                                    "left": f"{min(100, max(0, predicted))}%",
                                    "width": "2px", "height": "10px",
                                    "background": "rgba(255,255,255,0.5)",
                                    "transform": "translateX(-50%)"})
                ])
            ]),
        ])

    def _stats(sub):
        w = int((sub['wl'] == 1).sum())
        l = int((sub['wl'] == -1).sum())
        p = int((sub['wl'] == 0).sum())
        wr = round(w / (w + l) * 100, 1) if (w + l) > 0 else None
        return w, l, p, wr

    # ── 4-column factor breakdown helper ─────────────────────────────────────
    # Columns: 2025 Raw | 2025 HF | 2026 Raw | 2026 HF
    # HF = hard filters only (Line<3, MedF, GCS opp, GEE home, AvL>+10%, Rain). No LC.
    DSETS = [(df25_raw, "25 Raw"), (df25_hf, "25 HF"), (df26_raw, "26 Raw"), (df26_hf, "26 HF")]

    def _factor_table(title, note, group_specs):
        headers = [html.Th("Group", style=TH_L)]
        for _, col_lbl in DSETS:
            headers.append(html.Th(col_lbl, style={**TH, "fontSize": "9px"}))

        rows = []
        for row_label, fn in group_specs:
            if not any(len(fn(df)) >= 3 for df, _ in DSETS):
                continue
            cells = [html.Td(row_label, style=TD_L)]
            for df, _ in DSETS:
                sub = fn(df)
                n   = len(sub)
                if n < 3:
                    cells.append(html.Td("—", style={**TD, "color": FADED}))
                else:
                    wr = sub['win'].mean() * 100
                    cells.append(html.Td([
                        html.Span(f"{wr:.0f}%", style={"color": _wr_color(wr), "fontWeight": "600"}),
                        html.Span(f" ({n})", style={"color": FADED, "fontSize": "9px"}),
                    ], style=TD))
            rows.append(html.Tr(cells))

        return _card([
            _label(title),
            html.Div(note, style={"fontSize": "10px", "color": FADED,
                                   "fontFamily": MONO, "marginBottom": "4px"}),
            html.Div("HF = hard filters — Line<3, MedF, GCS opp, GEE home, AvL>+10%, Rain, LC<50%.",
                     style={"fontSize": "9px", "color": FADED,
                            "fontFamily": MONO, "marginBottom": "10px"}),
            html.Table([
                html.Thead(html.Tr(headers)),
                html.Tbody(rows or [html.Tr([html.Td("No data", style=TD_L)])]),
            ], style={"width": "100%", "borderCollapse": "collapse"}),
        ])

    # ── SECTION 1: STRATEGY PERFORMANCE (2026, T0/T1/T2) ─────────────────────
    TIER_DEFS = [('T0', 'T0  LC ≥70%'), ('T1', 'T1  LC 60–70%'), ('T2', 'T2  LC 50–60%')]

    tier_rows = []
    if not df26_strat.empty:
        for tier, lbl in TIER_DEFS:
            sub = df26_strat[df26_strat['Strategy'] == tier]
            if sub.empty:
                continue
            w, l, p, wr = _stats(sub)
            c = _wr_color(wr) if wr is not None else FADED
            tier_rows.append(html.Tr([
                html.Td(lbl, style={**TD_L, "color": AMBER, "fontWeight": "600"}),
                html.Td(str(w + l + p), style=TD),
                html.Td(str(w), style={**TD, "color": WIN}),
                html.Td(str(p), style={**TD, "color": FADED}),
                html.Td(str(l), style={**TD, "color": LOSS}),
                html.Td(f"{wr}%" if wr else "—", style={**TD, "color": c, "fontWeight": "700"}),
            ]))
        w2, l2, p2, wr2 = _stats(df26_strat)
        if wr2 is not None:
            sep = {"borderTop": f"1px solid {BORDER}"}
            tier_rows.append(html.Tr([
                html.Td("All Qualifying", style={**TD_L, **sep, "fontWeight": "600"}),
                html.Td(str(w2 + l2 + p2), style={**TD, **sep}),
                html.Td(str(w2), style={**TD, **sep, "color": WIN}),
                html.Td(str(p2), style={**TD, **sep, "color": FADED}),
                html.Td(str(l2), style={**TD, **sep, "color": LOSS}),
                html.Td(f"{wr2}%", style={**TD, **sep, "color": _wr_color(wr2), "fontWeight": "700"}),
            ]))

    tier_table = html.Table([
        html.Thead(html.Tr([html.Th(h, style=TH_L if i == 0 else TH)
                            for i, h in enumerate(["Tier", "n", "W", "P", "L", "WR%"])])),
        html.Tbody(tier_rows or [html.Tr([html.Td("No 2026 results yet", style=TD_L)])]),
    ], style={"width": "100%", "borderCollapse": "collapse"})

    rnd_header = [html.Th("Tier", style=TH_L)]
    for r in rounds_avail:
        rnd_header.append(html.Th(f"R{r}", style={**TH, "minWidth": "40px"}))
    rnd_header.append(html.Th("Total", style=TH))

    rnd_rows = []
    if not df26_strat.empty and rounds_avail:
        for tier, lbl in TIER_DEFS:
            sub = df26_strat[df26_strat['Strategy'] == tier]
            if sub.empty:
                continue
            cells = [html.Td(tier, style={**TD_L, "color": AMBER, "fontWeight": "600"})]
            for r in rounds_avail:
                rs = sub[sub['rnd_num'] == r]
                if rs.empty:
                    cells.append(html.Td("—", style={**TD, "color": FADED}))
                else:
                    w = int((rs['wl'] == 1).sum())
                    l = int((rs['wl'] == -1).sum())
                    n = w + l
                    c = _wr_color(w / n * 100) if n > 0 else FADED
                    pct = f" ({round(w/n*100)}%)" if n > 0 else ""
                    cells.append(html.Td(f"{w}/{n}{pct}", style={**TD, "color": c, "fontSize": "10px"}))
            w2, l2, p2, wr2 = _stats(sub)
            cells.append(html.Td(
                f"{wr2}%  ({w2 + l2})" if wr2 else "—",
                style={**TD, "fontWeight": "700", "color": _wr_color(wr2) if wr2 else FADED}
            ))
            rnd_rows.append(html.Tr(cells))

        # Combined Total row — sum across all tiers
        sep = {"borderTop": f"1px solid {BORDER}"}
        total_cells = [html.Td("Total", style={**TD_L, **sep, "fontWeight": "600"})]
        for r in rounds_avail:
            rs = df26_strat[df26_strat['rnd_num'] == r]
            if rs.empty:
                total_cells.append(html.Td("—", style={**TD, **sep, "color": FADED}))
            else:
                w = int((rs['wl'] == 1).sum())
                l = int((rs['wl'] == -1).sum())
                n = w + l
                c = _wr_color(w / n * 100) if n > 0 else FADED
                pct = f" ({round(w/n*100)}%)" if n > 0 else ""
                total_cells.append(html.Td(f"{w}/{n}{pct}", style={**TD, **sep, "color": c, "fontSize": "10px"}))
        wt, lt, pt, wrt = _stats(df26_strat)
        total_cells.append(html.Td(
            f"{wrt}%  ({wt + lt})" if wrt else "—",
            style={**TD, **sep, "fontWeight": "700", "color": _wr_color(wrt) if wrt else FADED}
        ))
        rnd_rows.append(html.Tr(total_cells))

    rnd_table = html.Table([
        html.Thead(html.Tr(rnd_header)),
        html.Tbody(rnd_rows or [html.Tr([html.Td("No 2026 results yet", style=TD_L)])]),
    ], style={"width": "100%", "borderCollapse": "collapse"})

    def _cut_rows(col, sort_asc=False, min_n=3):
        if df26_strat.empty:
            return []
        groups = []
        for val, grp in df26_strat.groupby(col):
            val = str(val).strip()
            if not val or val in ('', 'nan'):
                continue
            w, l, p, wr = _stats(grp)
            if w + l < min_n:
                continue
            groups.append((val, w + l + p, w, p, l, wr))
        groups.sort(key=lambda x: (x[5] or 0), reverse=not sort_asc)
        rows = []
        for val, n, w, p, l, wr in groups:
            c = _wr_color(wr) if wr else FADED
            rows.append(html.Tr([
                html.Td(val, style=TD_L),
                html.Td(str(n), style={**TD, "color": FADED}),
                html.Td(str(w), style={**TD, "color": WIN}),
                html.Td(str(p), style={**TD, "color": FADED}),
                html.Td(str(l), style={**TD, "color": LOSS}),
                html.Td(f"{wr}%" if wr else "—", style={**TD, "color": c, "fontWeight": "600"}),
            ]))
        return rows

    def _cut_card(title, col, sort_asc=False, min_n=3):
        rows = _cut_rows(col, sort_asc=sort_asc, min_n=min_n)
        return _card([
            _label(title),
            html.Table([
                html.Thead(html.Tr([html.Th(h, style=TH_L if i == 0 else TH)
                                    for i, h in enumerate(["", "n", "W", "P", "L", "WR%"])])),
                html.Tbody(rows or [html.Tr([html.Td("—", style=TD_L)])]),
            ], style={"width": "100%", "borderCollapse": "collapse"}),
        ], {"flex": "1"})

    pos_cut = _cut_card("BY POSITION · 2026 QUALIFYING", "Position")
    dvp_cut = _cut_card("BY DvP · 2026 QUALIFYING", "DvP")
    opp_cut = _cut_card("BY OPPONENT · 2026 QUALIFYING  (worst first)", "Opponent", sort_asc=True)

    # ── SECTION 2: SUBJECTIVE AVOIDANCE AUDIT ────────────────────────────────
    avoid_col = 'Subjective Avoid Reason'
    avoided = raw[
        raw[avoid_col].notna() &
        (raw[avoid_col] != '') &
        (raw[avoid_col].str.lower() != 'nan')
    ].copy()
    avoid_block = html.Div()
    if not avoided.empty:
        av = avoided.sort_values(['year_num', 'rnd_num', 'Player'],
                                  ascending=[False, False, True], na_position='last')
        has_result = av[av['W/L'].isin(['1', '-1', '0', '1.0', '-1.0', '0.0'])]
        good_calls = int((has_result['wl'] == -1).sum())
        missed     = int((has_result['wl'] ==  1).sum())
        pushed     = int((has_result['wl'] ==  0).sum())
        pending    = len(av) - len(has_result)
        acc        = round(good_calls / (good_calls + missed) * 100, 1) if (good_calls + missed) > 0 else None

        chips = html.Div([
            html.Span(f"Good calls: {good_calls}",
                      style={"color": WIN,  "fontFamily": MONO, "fontSize": "11px", "marginRight": "16px"}),
            html.Span(f"Missed: {missed}",
                      style={"color": LOSS, "fontFamily": MONO, "fontSize": "11px", "marginRight": "16px"}),
            html.Span(f"Push: {pushed}",
                      style={"color": FADED, "fontFamily": MONO, "fontSize": "11px", "marginRight": "16px"}),
            html.Span(f"Pending: {pending}",
                      style={"color": FADED, "fontFamily": MONO, "fontSize": "11px", "marginRight": "16px"}),
            html.Span(f"Accuracy: {acc}%" if acc else "Accuracy: —",
                      style={"color": _wr_color(acc) if acc else FADED,
                             "fontFamily": MONO, "fontSize": "11px", "fontWeight": "700"}),
        ], style={"marginBottom": "12px"})

        avoid_rows = []
        for _, r in av.iterrows():
            wl_raw = r.get('W/L', '')
            if   wl_raw in ('1',  '1.0'):  rs, rc = "W — missed",    LOSS
            elif wl_raw in ('-1', '-1.0'): rs, rc = "L — good call", WIN
            elif wl_raw in ('0',  '0.0'):  rs, rc = "Push",          FADED
            else:                          rs, rc = "pending",        FADED
            rnd = int(r['rnd_num'])  if pd.notna(r.get('rnd_num'))  else "?"
            yr  = int(r['year_num']) if pd.notna(r.get('year_num')) else ""
            avoid_rows.append(html.Tr([
                html.Td(f"{yr} R{rnd}" if yr else f"R{rnd}",
                        style={**TD_L, "color": FADED, "whiteSpace": "nowrap"}),
                html.Td(r.get('Player', ''),   style={**TD_L, "fontWeight": "500"}),
                html.Td(r.get('Strategy', ''), style={**TD, "color": AMBER}),
                html.Td(r.get('Line', ''),     style={**TD, "color": FADED}),
                html.Td(r.get('Actual', '') or "—", style={**TD, "color": FADED}),
                html.Td(rs, style={**TD_L, "color": rc, "fontWeight": "600"}),
                html.Td(r.get(avoid_col, ''),
                        style={**TD_L, "color": MUTED, "fontSize": "10px",
                               "maxWidth": "260px", "whiteSpace": "normal", "lineHeight": "1.4"}),
            ]))

        avoid_block = _card([
            _label("SUBJECTIVE AVOIDANCE AUDIT · ALL YEARS"),
            chips,
            html.Table([
                html.Thead(html.Tr([html.Th(h, style=TH_L if i not in (3, 4) else TH)
                                    for i, h in enumerate(["Rnd", "Player", "Tier", "Line",
                                                           "Actual", "Result", "Reason"])])),
                html.Tbody(avoid_rows),
            ], style={"width": "100%", "borderCollapse": "collapse"}),
        ], {"marginBottom": "24px"})

    # ── SECTION 3: PICK'EM FORMAT ANALYSIS ───────────────────────────────────
    def binom_prob(n, k, p):
        return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    def p_at_least(n, min_k, p):
        return sum(binom_prob(n, k, p) for k in range(min_k, n + 1))

    allin_data = [
        (2,  3.2,   1.789), (3,  6.5,   1.866), (4,  12,    1.861),
        (5,  25,    1.904), (6,  40,    1.849), (7,  80,    1.870),
        (8,  150,   1.871), (9,  275,   1.867), (10, 500,   1.862),
    ]
    hedge_data = [
        (3,  [(2, 1.2), (3, 3)],                        1.808),
        (4,  [(3, 2),   (4, 5)],                        1.855),
        (5,  [(3, 0.5), (4, 2),   (5, 10)],             1.861),
        (6,  [(4, 0.5), (5, 2.5), (6, 25)],             1.879),
        (7,  [(5, 1),   (6, 4),   (7, 40)],             1.871),
        (8,  [(6, 2),   (7, 5),   (8, 75)],             1.875),
        (9,  [(7, 3),   (8, 15),  (9, 100)],            1.888),
        (10, [(7, 0.5), (8, 5),   (9, 25), (10, 125)],  1.884),
    ]

    allin_rows = []
    for legs, payout, thresh in allin_data:
        be     = round(100 / thresh, 1)
        chance = round(WR ** legs * 100, 1)
        ev     = round(WR ** legs * payout * 100 - 100, 1)
        allin_rows.append(html.Tr([
            html.Td(f"{legs}-leg", style=TD_L),
            html.Td(f"{be}%",      style={**TD, "color": FADED}),
            html.Td(f"{chance}%",  style={**TD, "color": _wr_color(chance)}),
            html.Td(f"+{ev}%",     style={**TD, "color": _ev_color(ev), "fontWeight": "600"}),
            html.Td(f"{payout}x",  style={**TD, "color": FADED}),
        ]))

    hedge_rows = []
    for legs, tiers, thresh in hedge_data:
        be        = round(100 / thresh, 1)
        min_wins  = tiers[0][0]
        profit_k  = next((k for k, pay in tiers if pay > 1.0), None)
        chance    = round(p_at_least(legs, profit_k, WR) * 100, 1) if profit_k is not None else 0.0
        ev_exact  = 0
        for i, (k, pay) in enumerate(tiers):
            next_k   = tiers[i + 1][0] if i + 1 < len(tiers) else legs + 1
            p_exact  = sum(binom_prob(legs, j, WR) for j in range(k, next_k))
            ev_exact += p_exact * pay
        ev_pct  = round(ev_exact * 100 - 100, 1)
        pay_str = "/".join(str(int(p) if p == int(p) else p) for _, p in tiers)
        pl      = html.Span(f" ≥{profit_k}/{legs}",
                            style={"color": FADED, "fontSize": "9px", "marginLeft": "4px"}) if profit_k else None
        hedge_rows.append(html.Tr([
            html.Td(f"{legs}-leg (≥{min_wins})", style=TD_L),
            html.Td(f"{be}%",                   style={**TD, "color": FADED}),
            html.Td([f"{chance}%", pl],          style={**TD, "color": _wr_color(chance)}),
            html.Td(f"+{ev_pct}%",              style={**TD, "color": _ev_color(ev_pct), "fontWeight": "600"}),
            html.Td(pay_str,                     style={**TD, "color": FADED, "fontSize": "10px"}),
        ]))

    pickem_section = html.Div([
        html.Div([
            html.Span("PICK'EM FORMAT ANALYSIS", style={
                "color": TEXT, "fontWeight": "700", "fontSize": "14px", "fontFamily": MONO}),
            html.Span(f"  ·  {round(WR * 100, 1)}% actual 2026 qualifying WR",
                      style={"color": MUTED, "fontSize": "11px", "fontFamily": MONO}),
        ], style={"marginBottom": "14px"}),
        html.Div([
            _card([
                _label("ALL-IN  ·  must win every leg"),
                html.Table([
                    html.Thead(html.Tr([html.Th(h, style=TH_L if i == 0 else TH)
                                        for i, h in enumerate(["Format", "Breakeven", "Chance", "EV", "Payout"])])),
                    html.Tbody(allin_rows),
                ], style={"width": "100%", "borderCollapse": "collapse"}),
            ], {"flex": "1"}),
            _card([
                _label("HEDGE  ·  pays on partial wins"),
                html.Table([
                    html.Thead(html.Tr([html.Th(h, style=TH_L if i == 0 else TH)
                                        for i, h in enumerate(["Format", "Breakeven", "To Profit", "EV", "Payouts"])])),
                    html.Tbody(hedge_rows),
                ], style={"width": "100%", "borderCollapse": "collapse"}),
            ], {"flex": "1"}),
        ], style={"display": "flex", "gap": "14px", "alignItems": "flex-start"}),
    ], style={"marginBottom": "24px"})

    # ── SECTION 4: FACTOR BREAKDOWN (4 columns: 25 Raw / 25 HF / 26 Raw / 26 HF) ──
    factor_divider = html.Div([
        html.Div("FACTOR BREAKDOWN — 2025 TEST  vs  2026 TRAIN  ·  Raw vs Hard Filters", style={
            "fontSize": "9px", "color": MUTED, "fontFamily": MONO,
            "letterSpacing": "0.15em", "fontWeight": "600",
        }),
        html.Div(style={"flex": "1", "height": "1px", "background": BORDER, "marginLeft": "12px"}),
    ], style={"display": "flex", "alignItems": "center", "margin": "24px 0 16px"})

    pos_order = ['Wing', 'Ruck', 'GenD', 'InsM', 'FwdMid', 'SmF', 'KeyF', 'KeyD', 'MedF']
    pos_ft = _factor_table(
        "POSITION",
        "MedF is a hard filter — MedF rows vanish in HF columns. Raw shows the unfiltered rate.",
        [(p, lambda df, p=p: df[df['_pos'] == p]) for p in pos_order],
    )

    dvp_order = ['Strong Unders', 'Moderate Unders', 'Slight Unders', 'Neutral',
                 'Slight Easy', 'Moderate Easy', 'Strong Easy']
    dvp_ft = _factor_table(
        "DvP — OPPONENT TACKLE DIFFICULTY",
        "No hard DvP filter active. Neutral DvP is a T2-only soft skip in flag logic.",
        [(d, lambda df, d=d: df[df['_dvp'] == d]) for d in dvp_order],
    )

    # AvL × LC matrix — 4 datasets shown as 2×2 grid
    AVL_SPECS = [
        ("< −20%",             lambda df: df[df['_avl'] <  -20]),
        ("−20% to −10%",       lambda df: df[(df['_avl'] >= -20) & (df['_avl'] < -10)]),
        ("−10% to  0%",        lambda df: df[(df['_avl'] >= -10) & (df['_avl'] <   0)]),
        ("0% to +10%",         lambda df: df[(df['_avl'] >=   0) & (df['_avl'] <  10)]),
        ("+10%+  (HF cuts)",   lambda df: df[df['_avl'] >= 10]),
        ("All AvL",            lambda df: df),
    ]
    LC_BANDS = [
        ("<50%",         lambda df: df[df['_lc'] <  50]),
        ("50–60% (T2)",  lambda df: df[(df['_lc'] >= 50) & (df['_lc'] < 60)]),
        ("60–70% (T1)",  lambda df: df[(df['_lc'] >= 60) & (df['_lc'] < 70)]),
        ("≥70% (T0)",    lambda df: df[df['_lc'] >= 70]),
        ("All LC",       lambda df: df),
    ]
    TH_XS = {**TH, "fontSize": "8px", "padding": "5px 6px"}
    TD_XS = {**TD, "fontSize": "10px", "padding": "5px 6px"}
    TD_XS_L = {**TD_XS, "textAlign": "left"}

    def _avl_lc_mini(base_df, label):
        headers = [html.Th("AvL \\ LC", style={**TH_XS, "textAlign": "left"})]
        for lc_lbl, _ in LC_BANDS:
            headers.append(html.Th(lc_lbl, style=TH_XS))
        rows = []
        for avl_lbl, avl_fn in AVL_SPECS:
            avl_sub = avl_fn(base_df.dropna(subset=['_avl', '_lc']))
            cells = [html.Td(avl_lbl, style=TD_XS_L)]
            any_data = False
            for _, lc_fn in LC_BANDS:
                sub = lc_fn(avl_sub)
                n = len(sub)
                if n < 3:
                    cells.append(html.Td("—", style={**TD_XS, "color": FADED}))
                else:
                    wr = sub['win'].mean() * 100
                    any_data = True
                    cells.append(html.Td([
                        html.Span(f"{wr:.0f}%", style={"color": _wr_color(wr), "fontWeight": "600"}),
                        html.Span(f" ({n})", style={"color": FADED, "fontSize": "8px"}),
                    ], style=TD_XS))
            if any_data or avl_lbl == "All AvL":
                rows.append(html.Tr(cells))
        return html.Div([
            html.Div(label, style={"fontSize": "9px", "color": MUTED, "fontFamily": MONO,
                                   "letterSpacing": "0.08em", "fontWeight": "600",
                                   "marginBottom": "6px", "marginTop": "10px"}),
            html.Table([
                html.Thead(html.Tr(headers)),
                html.Tbody(rows),
            ], style={"width": "100%", "borderCollapse": "collapse"}),
        ])

    avl_ft = _card([
        _label("AVL × LC MATRIX — AVG vs LINE  ×  LINE CONSISTENCY"),
        html.Div(
            "Rows = AvL bucket. Cols = LC band (maps to T2/T1/T0). "
            "HF removes AvL>+10% — that row is sparse in HF columns.",
            style={"fontSize": "10px", "color": FADED, "fontFamily": MONO, "marginBottom": "4px"}),
        html.Div("HF = hard filters — Line<3, MedF, GCS opp, GEE home, AvL>+10%, Rain, LC<50%.",
                 style={"fontSize": "9px", "color": FADED, "fontFamily": MONO, "marginBottom": "10px"}),
        dbc.Row([
            dbc.Col(_avl_lc_mini(df25_raw, "2025 RAW"), md=6),
            dbc.Col(_avl_lc_mini(df26_raw, "2026 RAW"), md=6),
        ], style={"marginBottom": "4px"}),
        dbc.Row([
            dbc.Col(_avl_lc_mini(df25_hf,  "2025 HF — TEST"),  md=6),
            dbc.Col(_avl_lc_mini(df26_hf,  "2026 HF — TRAIN"), md=6),
        ]),
    ])

    # Line × LC  and  Line × AvL  interaction matrices
    LINE_BUCKETS = [
        ("3.0–3.9",  lambda df: df[(df['_line'] >= 3.0) & (df['_line'] < 4.0)]),
        ("4.0–4.9",  lambda df: df[(df['_line'] >= 4.0) & (df['_line'] < 5.0)]),
        ("5.0–5.9",  lambda df: df[(df['_line'] >= 5.0) & (df['_line'] < 6.0)]),
        ("6.0+",     lambda df: df[df['_line'] >= 6.0]),
        ("All",      lambda df: df),
    ]
    AVL_GROUPS = [
        ("< −10%",    lambda df: df[df['_avl'] < -10]),
        ("−10 to 0%", lambda df: df[(df['_avl'] >= -10) & (df['_avl'] <  0)]),
        ("0 to +10%", lambda df: df[(df['_avl'] >=   0) & (df['_avl'] < 10)]),
        ("All AvL",   lambda df: df),
    ]

    def _line_lc_mini(base_df, label):
        headers = [html.Th("Line \\ LC", style={**TH_XS, "textAlign": "left"})]
        for lc_lbl, _ in LC_BANDS:
            headers.append(html.Th(lc_lbl, style=TH_XS))
        rows = []
        for line_lbl, line_fn in LINE_BUCKETS:
            line_sub = line_fn(base_df.dropna(subset=['_line', '_lc']))
            cells = [html.Td(line_lbl, style=TD_XS_L)]
            any_data = False
            for _, lc_fn in LC_BANDS:
                sub = lc_fn(line_sub)
                n = len(sub)
                if n < 3:
                    cells.append(html.Td("—", style={**TD_XS, "color": FADED}))
                else:
                    wr = sub['win'].mean() * 100
                    any_data = True
                    cells.append(html.Td([
                        html.Span(f"{wr:.0f}%", style={"color": _wr_color(wr), "fontWeight": "600"}),
                        html.Span(f" ({n})",    style={"color": FADED, "fontSize": "8px"}),
                    ], style=TD_XS))
            if any_data or line_lbl == "All":
                rows.append(html.Tr(cells))
        return html.Div([
            html.Div(label, style={"fontSize": "9px", "color": MUTED, "fontFamily": MONO,
                                   "letterSpacing": "0.08em", "fontWeight": "600",
                                   "marginBottom": "6px", "marginTop": "10px"}),
            html.Table([html.Thead(html.Tr(headers)), html.Tbody(rows)],
                       style={"width": "100%", "borderCollapse": "collapse"}),
        ])

    def _line_avl_mini(base_df, label):
        headers = [html.Th("Line \\ AvL", style={**TH_XS, "textAlign": "left"})]
        for avl_lbl, _ in AVL_GROUPS:
            headers.append(html.Th(avl_lbl, style=TH_XS))
        rows = []
        for line_lbl, line_fn in LINE_BUCKETS:
            line_sub = line_fn(base_df.dropna(subset=['_line', '_avl']))
            cells = [html.Td(line_lbl, style=TD_XS_L)]
            any_data = False
            for _, avl_fn in AVL_GROUPS:
                sub = avl_fn(line_sub)
                n = len(sub)
                if n < 3:
                    cells.append(html.Td("—", style={**TD_XS, "color": FADED}))
                else:
                    wr = sub['win'].mean() * 100
                    any_data = True
                    cells.append(html.Td([
                        html.Span(f"{wr:.0f}%", style={"color": _wr_color(wr), "fontWeight": "600"}),
                        html.Span(f" ({n})",    style={"color": FADED, "fontSize": "8px"}),
                    ], style=TD_XS))
            if any_data or line_lbl == "All":
                rows.append(html.Tr(cells))
        return html.Div([
            html.Div(label, style={"fontSize": "9px", "color": MUTED, "fontFamily": MONO,
                                   "letterSpacing": "0.08em", "fontWeight": "600",
                                   "marginBottom": "6px", "marginTop": "10px"}),
            html.Table([html.Thead(html.Tr(headers)), html.Tbody(rows)],
                       style={"width": "100%", "borderCollapse": "collapse"}),
        ])

    line_lc_ft = _card([
        _label("LINE × LC — LINE SIZE  ×  LINE CONSISTENCY"),
        html.Div("Does WR within each LC tier hold flat across line sizes, or do certain line buckets underperform?",
                 style={"fontSize": "10px", "color": FADED, "fontFamily": MONO, "marginBottom": "4px"}),
        html.Div("HF = hard filters — Line<3, MedF, GCS opp, GEE home, AvL>+10%, Rain, LC<50%.",
                 style={"fontSize": "9px", "color": FADED, "fontFamily": MONO, "marginBottom": "10px"}),
        dbc.Row([
            dbc.Col(_line_lc_mini(df25_raw, "2025 RAW"),        md=6),
            dbc.Col(_line_lc_mini(df26_raw, "2026 RAW"),        md=6),
        ], style={"marginBottom": "4px"}),
        dbc.Row([
            dbc.Col(_line_lc_mini(df25_hf,  "2025 HF — TEST"),  md=6),
            dbc.Col(_line_lc_mini(df26_hf,  "2026 HF — TRAIN"), md=6),
        ]),
    ])

    line_avl_ft = _card([
        _label("LINE × AvL — LINE SIZE  ×  AVG vs LINE"),
        html.Div("Does AvL edge hold across all line sizes, or do small/large lines behave differently?",
                 style={"fontSize": "10px", "color": FADED, "fontFamily": MONO, "marginBottom": "4px"}),
        html.Div("HF = hard filters — Line<3, MedF, GCS opp, GEE home, AvL>+10%, Rain, LC<50%.",
                 style={"fontSize": "9px", "color": FADED, "fontFamily": MONO, "marginBottom": "10px"}),
        dbc.Row([
            dbc.Col(_line_avl_mini(df25_raw, "2025 RAW"),        md=6),
            dbc.Col(_line_avl_mini(df26_raw, "2026 RAW"),        md=6),
        ], style={"marginBottom": "4px"}),
        dbc.Row([
            dbc.Col(_line_avl_mini(df25_hf,  "2025 HF — TEST"),  md=6),
            dbc.Col(_line_avl_mini(df26_hf,  "2026 HF — TRAIN"), md=6),
        ]),
    ])

    line_ft = _factor_table(
        "LINE SIZE",
        "HF removes Line<3. The <3 row appears in Raw only.",
        [
            ("< 3.0  (filtered in HF)",  lambda df: df[df['_line'] <  3.0]),
            ("3.0 – 3.9",               lambda df: df[(df['_line'] >= 3.0) & (df['_line'] < 4.0)]),
            ("4.0 – 4.9",               lambda df: df[(df['_line'] >= 4.0) & (df['_line'] < 5.0)]),
            ("5.0 – 5.9",               lambda df: df[(df['_line'] >= 5.0) & (df['_line'] < 6.0)]),
            ("6.0+",                    lambda df: df[df['_line'] >= 6.0]),
        ],
    )

    weather_ft = _factor_table(
        "WEATHER  (HF removes Rain)",
        "Rain is a hard filter — Rain rows appear in Raw only.",
        [
            ("Neutral",     lambda df: df[df['Weather'].str.contains('Neutral', na=False)]),
            ("Rain",        lambda df: df[df['Weather'].str.contains('Rain',    na=False)]),
        ],
    )

    travel_ft = _factor_table(
        "TRAVEL FATIGUE  (parked at n=100)",
        "Long Travel ~9.7pp gap vs Neutral in historical analysis — revisit at n=100.",
        [
            ("Neutral",     lambda df: df[df['Travel Fatigue'].str.contains('Neutral', na=False)]),
            ("Long Travel", lambda df: df[df['Travel Fatigue'].str.contains('Long',    na=False)]),
        ],
    )

    # ── SECTION 5: CALIBRATION META (predicted WR vs actual, year-split) ─────
    calib_divider = html.Div([
        html.Div("HIST WR CALIBRATION — does predicted WR match actual?", style={
            "fontSize": "9px", "color": MUTED, "fontFamily": MONO,
            "letterSpacing": "0.15em", "fontWeight": "600",
        }),
        html.Div(style={"flex": "1", "height": "1px", "background": BORDER, "marginLeft": "12px"}),
    ], style={"display": "flex", "alignItems": "center", "margin": "24px 0 16px"})

    META  = [(df25_hf, "2025 HF (test)"), (df26_hf, "2026 HF (train)")]
    BUCKETS = [("<50%", None, 50), ("50–55%", 50, 55), ("55–60%", 55, 60),
               ("60–65%", 60, 65), ("65–70%", 65, 70), ("70%+", 70, None)]

    meta_headers = [html.Th("Pred bucket", style=TH_L)] + [html.Th(lbl, style=TH) for _, lbl in META]
    meta_rows_built = []
    for bkt_lbl, lo, hi in BUCKETS:
        cells = [html.Td(bkt_lbl, style=TD_L)]
        for _df, _ in META:
            sub = _df.dropna(subset=['_hwr_p'])
            mask = pd.Series([True] * len(sub), index=sub.index)
            if lo is not None: mask &= sub['_hwr_p'] >= lo
            if hi is not None: mask &= sub['_hwr_p'] <  hi
            sub = sub[mask]
            n   = len(sub)
            if n < 3:
                cells.append(html.Td("—", style={**TD, "color": FADED}))
            else:
                wr       = sub['win'].mean() * 100
                se       = _se(wr, n)
                pred_mid = ((lo or 45) + (hi or 75)) / 2
                diff     = wr - pred_mid
                dcol     = WIN if diff >= 0 else LOSS
                cells.append(html.Td([
                    html.Span(f"{wr:.0f}%", style={"color": _wr_color(wr), "fontWeight": "600"}),
                    html.Span(f" ±{se:.0f}", style={"color": FADED, "fontSize": "9px"}),
                    html.Span(f" {diff:+.0f}", style={"color": dcol, "fontSize": "9px", "marginLeft": "3px"}),
                    html.Br(),
                    html.Span(f"n={n}", style={"color": FADED, "fontSize": "9px"}),
                ], style=TD))
        meta_rows_built.append(html.Tr(cells))

    meta_calib_card = _card([
        _label("PREDICTED WR BUCKET vs ACTUAL  ·  2025 HF (test) vs 2026 HF (train)"),
        html.Div("Actual% ±SE  Δ=actual−predicted_mid. Both sides use hard filters only, no LC.",
                 style={"fontSize": "10px", "color": FADED, "fontFamily": MONO, "marginBottom": "12px"}),
        html.Table([
            html.Thead(html.Tr(meta_headers)),
            html.Tbody(meta_rows_built),
        ], style={"width": "100%", "borderCollapse": "collapse"}),
    ])

    thresh_headers = [html.Th("Zone", style=TH_L)] + [html.Th(lbl, style=TH) for _, lbl in META]
    thresh_rows_built = []
    for zone_lbl, min_b, color in [("≥56% predicted  (bet zone)",  56,   WIN),
                                    ("<56% predicted  (avoid zone)", None, LOSS)]:
        cells = [html.Td(zone_lbl, style={**TD_L, "color": color})]
        for _df, _ in META:
            sub = _df.dropna(subset=['_hwr_p'])
            sub = sub[sub['_hwr_p'] >= min_b] if min_b else sub[sub['_hwr_p'] < 56]
            n   = len(sub)
            if n < 3:
                cells.append(html.Td("—", style={**TD, "color": FADED}))
            else:
                wr = sub['win'].mean() * 100
                cells.append(html.Td([
                    html.Span(f"{wr:.0f}%", style={"color": _wr_color(wr), "fontWeight": "600"}),
                    html.Span(f" ({n})", style={"color": FADED, "fontSize": "9px", "marginLeft": "3px"}),
                ], style=TD))
        thresh_rows_built.append(html.Tr(cells))

    thresh_card_built = _card([
        _label("THRESHOLD SPLIT — bet zone (≥56%) vs avoid zone (<56%)"),
        html.Div("Gap between zones measures signal quality of the Hist WR filter.",
                 style={"fontSize": "10px", "color": FADED, "fontFamily": MONO, "marginBottom": "12px"}),
        html.Table([
            html.Thead(html.Tr(thresh_headers)),
            html.Tbody(thresh_rows_built),
        ], style={"width": "100%", "borderCollapse": "collapse"}),
    ])

    # ── ASSEMBLE ──────────────────────────────────────────────────────────────
    return html.Div([
        html.Div([
            html.Span("2026 RESULTS", style={"color": TEXT, "fontWeight": "700",
                                              "fontSize": "14px", "fontFamily": MONO}),
            html.Span(f"  ·  through R{max_round}  ·  qualifying legs only (T0 / T1 / T2)",
                      style={"color": MUTED, "fontSize": "11px", "fontFamily": MONO}),
        ], style={"marginBottom": "14px"}),
        html.Div([
            _card([_label("WIN RATE BY TIER"), tier_table], {"flex": "0 0 auto"}),
            _card([_label("WR BY ROUND  ·  W / placed  (excl pushes)"),
                   html.Div(rnd_table, style={"overflowX": "auto"})],
                  {"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "14px", "alignItems": "flex-start", "marginBottom": "14px"}),
        html.Div([pos_cut, dvp_cut, opp_cut],
                 style={"display": "flex", "gap": "14px", "alignItems": "flex-start", "marginBottom": "14px"}),
        avoid_block,
        pickem_section,
        factor_divider,
        dbc.Row([dbc.Col(pos_ft,     md=6), dbc.Col(dvp_ft,     md=6)]),
        dbc.Row([dbc.Col(avl_ft, md=12)]),
        dbc.Row([dbc.Col(line_ft, md=4), dbc.Col(weather_ft, md=4), dbc.Col(travel_ft, md=4)]),
        calib_divider,
        dbc.Row([dbc.Col(meta_calib_card, md=8), dbc.Col(thresh_card_built, md=4)]),
    ], style={"background": BG, "padding": "18px", "borderRadius": "8px", "minHeight": "100vh"})





# ── Performance + Multi Builder tab callback ─────────────────────────────────

@app.callback(
    Output('excluded-teams-store',  'data'),
    Input({'type': 'team-chip', 'index': ALL}, 'n_clicks'),
    State('excluded-teams-store',   'data'),
    prevent_initial_call=True,
)
def toggle_team(n_clicks_list, excluded):
    excluded = set(excluded or [])
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        team = triggered['index']
        if team in excluded:
            excluded.discard(team)
        else:
            excluded.add(team)
    return list(excluded)


@app.callback(
    Output('excluded-legs-store', 'data'),
    Input('legs-table', 'active_cell'),
    State('legs-table', 'data'),
    State('excluded-legs-store', 'data'),
    prevent_initial_call=True,
)
def toggle_excluded_leg(active_cell, table_data, current_excluded):
    if not active_cell or not table_data:
        return dash.no_update
    if active_cell.get('column_id') != '_excl':
        return dash.no_update
    row_idx = active_cell.get('row', -1)
    if row_idx < 0 or row_idx >= len(table_data):
        return dash.no_update
    row_id = table_data[row_idx].get('id', '')
    if not row_id:
        return dash.no_update
    excluded = set(current_excluded or [])
    if row_id in excluded:
        excluded.discard(row_id)
    else:
        excluded.add(row_id)
    return sorted(excluded)


@app.callback(
    Output('multi-builder-content', 'children'),
    Output('analysis-content',      'children'),
    Output('calibration-content',   'children'),
    Output('pairings-store',        'data'),
    Input('stat-tabs',              'active_tab'),
    Input('rr-top-n-store',         'data'),
    Input('placed-bets-store',      'data'),
    Input('excluded-teams-store',   'data'),
    Input('excluded-legs-store',    'data'),
)
def update_special_tabs(active_tab, rr_top_n, placed_ids, excluded_teams, excluded_legs):
    if active_tab == 'tab-analysis':
        return html.Div(), build_combined_layout(), html.Div(), {}
    else:  # tab-multi (default)
        layout, pairings_data = build_multi_builder_layout(
            checked_ids=placed_ids or [],
            rr_top_n=int(rr_top_n or 4),
            excluded_teams=excluded_teams or [],
            excluded_legs=excluded_legs or [],
        )
        return layout, html.Div(), html.Div(), pairings_data






# ── CSV export ────────────────────────────────────────────────────────────────
@app.callback(
    Output("download-csv",   "data"),
    Input("export-button",   "n_clicks"),
    [State("stat-tabs",      "active_tab")],
    prevent_initial_call=True,
)
def export_data(n_clicks, active_tab):
    if not n_clicks:
        return dash.no_update
    stat_type = {'tab-marks': 'marks', 'tab-tackles': 'tackles'}.get(active_tab, 'disposals')
    df = processed_data_by_stat.get(stat_type)
    if df is not None and not df.empty:
        fname = f"afl_{stat_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return dcc.send_data_frame(df.to_csv, fname, index=False)
    return dash.no_update


# ── debug helpers ─────────────────────────────────────────────────────────────
def debug_pickem_matching(stat_type='disposals'):
    print(f"\n=== DEBUG PICKEM {stat_type.upper()} ===")
    try:
        pickem_data = get_pickem_data_for_dashboard(stat_type)
        print(f"Players retrieved: {len(pickem_data) if pickem_data else 0}")
        if pickem_data:
            print(f"Sample: {list(pickem_data.keys())[:5]}")
    except Exception as e:
        print(f"Error: {e}")
    print("=== END ===\n")


debug_pickem_matching('disposals')
debug_pickem_matching('marks')
debug_pickem_matching('tackles')

def _run_startup_scripts():
    import subprocess, sys, os
    cwd = os.path.dirname(os.path.abspath(__file__))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    scripts = [
        ("update_results.py",           "Update Results"),
        ("_fetch_gbg.py",               "GBG Data"),
        ("_build_tackle_avgs.py",       "Rebuild Tackle Avgs CSV"),
        ("_push_lc_to_sheets.py",       "Line Consistency"),
        ("_push_avl_per_round.py",      "AvL + Avg Tackles"),
        ("_push_l5_per_round.py",       "Last 5 Tackles"),
        ("_push_strategy_to_sheets.py", "Strategy Tiers"),
    ]
    pad = max(len(label) for _, label in scripts)
    print("\n" + "─" * 52)
    print("  AFL Dashboard — startup data refresh")
    print("─" * 52)
    for script, label in scripts:
        script_path = os.path.join(cwd, script)
        if not os.path.exists(script_path):
            print(f"  SKIP  {label:<{pad}}  (script not found)")
            continue
        print(f"  ...   {label:<{pad}}", end="", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                cwd=cwd, timeout=180, env=env,
            )
            if result.returncode == 0:
                last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "ok"
                print(f"\r  OK    {label:<{pad}}  {last_line}")
            else:
                err = (result.stderr or result.stdout).strip().splitlines()[-1] if (result.stderr or result.stdout).strip() else "exit " + str(result.returncode)
                print(f"\r  FAIL  {label:<{pad}}  {err}")
        except subprocess.TimeoutExpired:
            print(f"\r  FAIL  {label:<{pad}}  timed out after 180s")
        except Exception as e:
            print(f"\r  FAIL  {label:<{pad}}  {e}")
    print("─" * 52 + "\n")


if __name__ == '__main__':
    _run_startup_scripts()
    app.run(debug=True)
