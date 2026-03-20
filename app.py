import pandas as pd
import numpy as np
import dash
from dash import html, dcc, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Import from existing modules
from fixture_scraper import scrape_next_round_fixture
from travel_fatigue import build_travel_log
from stadium_locations import STADIUM_COORDS
from data_processor import load_and_prepare_data, calculate_dvp
from dabble_scraper import get_pickem_data_for_dashboard, normalize_player_name

# ===== CONSTANTS =====
OPENWEATHER_API_KEY = "e76003c560c617b8ffb27f2dee7123f4"

# ── Google Sheets config ───────────────────────────────────────────────────────
# 1. Go to https://console.cloud.google.com
# 2. Create a project → enable "Google Sheets API" + "Google Drive API"
# 3. Create credentials → Service Account → download JSON → save as
#    "google_credentials.json" in the same folder as app.py
# 4. Open your Google Sheet → Share → add the service account email as Editor
GOOGLE_SHEET_ID          = "10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w"
GOOGLE_CREDENTIALS_FILE  = "google_credentials.json"
GOOGLE_SHEET_TAB         = "Bet Log"   # change if your tab has a different name
# ──────────────────────────────────────────────────────────────────────────────

POSITION_MAP = {
    "KeyF": ["FF", "CHF"],
    "GenF": ["HFFR", "HFFL", "FPL", "FPR"],
    "Ruck": ["RK"],
    "InsM": ["C", "RR", "R"],
    "Wing": ["WL", "WR"],
    "GenD": ["HBFL", "HBFR", "BPL", "BPR"],
    "KeyD": ["CHB", "FB"]
}

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
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds  = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_FILE, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except FileNotFoundError:
        print(f"⚠️  {GOOGLE_CREDENTIALS_FILE} not found — Sheets sync disabled.")
        return None
    except Exception as e:
        print(f"⚠️  Sheets auth error: {e}")
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
        print(f"⚠️  Could not read existing sheet keys: {e}")
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
        "Line Consistency", "Bet Priority", "Bet Flag", "Actual", "W/L"
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

            new_row = [
                stat_label,                              # Type
                current_year,                            # Year
                current_round,                           # Round
                player,                                  # Player
                str(row.get("Team",             "")),
                str(row.get("Opponent",         "")),
                str(row.get("Position",         "")),
                str(row.get("Travel Fatigue",   "")),
                str(row.get("Weather",          "")),
                str(row.get("DvP",              "")),
                line_val,                                # Line
                str(row.get("Avg vs Line",      "")),
                str(row.get("Line Consistency", "")),
                str(row.get("Bet Priority",     "")),
                str(row.get("Bet Flag",         "")),
                "",   # Actual  — filled mid-week by update_results.py
                "",   # W/L     — filled mid-week by update_results.py
            ]
            rows_to_add.append(new_row)
            existing_keys.add(key)   # prevent same-session dupes

    if not rows_to_add:
        return 0, "✅ Nothing new to push — all flagged bets already in sheet."

    # Batch append for speed
    worksheet.append_rows(rows_to_add, value_input_option="USER_ENTERED")
    return len(rows_to_add), f"✅ {len(rows_to_add)} new bet row(s) pushed to Google Sheets."


# ===== WEATHER FUNCTIONS =====
import requests
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go

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
    else:  # tackles
        if rain_value > 0:
            flag_count, rating_text = -999, "Avoid"
        else:
            flag_count, rating_text = 0, "Neutral"

    factors = []
    if rain_value > 0: factors.append("Rain")
    if wind_value > 0: factors.append("Wind")
    factors_str = ', '.join(factors) if factors else "Clear conditions"

    if stat_type == 'tackles' and rain_value > 0:
        rating = f"🔵 AVOID RAIN ({factors_str})"
    elif flag_count == 0:
        rating = "✅ Neutral"
    elif rating_text == "Medium":
        rating = f"⚠️ Medium Unders Edge ({factors_str})"
    else:
        rating = f"🔴 Strong Unders Edge ({factors_str})"

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


def calculate_score(player_row, team_weather, simplified_dvp, stat_type='disposals'):
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
                travel_points += 1.0
                travel_details.append("Short Break: +1.0")
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
# CORE STRATEGY LOGIC — 6 finalised strategies
# =============================================================================
def calculate_bet_flag(player_row, stat_type='disposals'):
    """
    Priority / Strategy / Win Rate / n
      1  Tackle + Mod Travel + Avg <5%  + Slight Unders or Neutral DvP   95.7%  23
      2  Mark   + Avg <-5%  + No Easy  + Line >5 + LC >60%               90.0%  20
      3  KeyF Mark + Line >5 + No Easy DvP                               85.0%  20
      4  Tackle + Strong Unders DvP + Avg <15%                           79.3%  29
      5  Mark   + Strong Unders DvP + Line >4.5                          73.3%  30
      6  GenF Tackle + exclude Slight Unders DvP                         69.8%  53

    Global kill-switch: Short Break travel → skip all (overs signal, not unders).
    """
    try:
        position             = player_row.get('Position',         player_row.get('position',          ''))
        dvp                  = player_row.get('DvP',              player_row.get('dvp',                ''))
        travel_fatigue       = player_row.get('Travel Fatigue',   player_row.get('travel_fatigue',     ''))
        line_str             = player_row.get('Line',             '')
        avg_vs_line_str      = player_row.get('Avg vs Line',      '')
        line_consistency_str = player_row.get('Line Consistency', '')

        if not line_str or line_str == "":
            return {"priority": "", "description": ""}
        try:
            line_value = float(line_str)
        except (ValueError, TypeError):
            return {"priority": "", "description": ""}

        avg_vs_line_pct = None
        if avg_vs_line_str:
            try:
                avg_vs_line_pct = float(avg_vs_line_str.replace('%', '').replace('+', ''))
            except (ValueError, TypeError):
                pass

        line_consistency_pct = None
        if line_consistency_str:
            try:
                line_consistency_pct = float(line_consistency_str.replace('%', ''))
            except (ValueError, TypeError):
                pass

        # travel
        has_short_break     = 'Short Break'   in travel_fatigue
        has_moderate_travel = 'Moderate'      in travel_fatigue

        # DvP
        has_slight_unders_dvp = 'Slight Unders'   in dvp
        has_strong_unders_dvp = 'Strong Unders'   in dvp
        has_neutral_dvp       = 'Neutral' in dvp and 'Unknown' not in dvp
        has_easy_dvp          = any(x in dvp for x in ['Strong Easy', 'Moderate Easy', 'Slight Easy'])

        # Global kill-switch
        if has_short_break:
            return {"priority": "", "description": ""}

        # ── Strategy 1 ────────────────────────────────────────────────────────
        if (stat_type == 'tackles'
                and has_moderate_travel
                and (has_slight_unders_dvp or has_neutral_dvp)
                and avg_vs_line_pct is not None and avg_vs_line_pct < 5):
            return {"priority": "1",
                    "description": "Tackle + Mod Travel + Avg <5% + Slight Unders/Neutral DvP → 95.7% WR (22/23)"}

        # ── Strategy 2 ────────────────────────────────────────────────────────
        if (stat_type == 'marks'
                and line_value > 5
                and not has_easy_dvp
                and avg_vs_line_pct      is not None and avg_vs_line_pct      < -5
                and line_consistency_pct is not None and line_consistency_pct > 60):
            return {"priority": "2",
                    "description": "Mark + Avg <-5% + No Easy DvP + Line >5 + LC >60% → 90.0% WR (18/20)"}

        # ── Strategy 3 ────────────────────────────────────────────────────────
        if (stat_type == 'marks'
                and position == 'KeyF'
                and line_value > 5
                and not has_easy_dvp):
            return {"priority": "3",
                    "description": "KeyF Mark + Line >5 + No Easy DvP → 85.0% WR (17/20)"}

        # ── Strategy 4 ────────────────────────────────────────────────────────
        if (stat_type == 'tackles'
                and has_strong_unders_dvp
                and avg_vs_line_pct is not None and avg_vs_line_pct < 15):
            return {"priority": "4",
                    "description": "Tackle + Strong Unders DvP + Avg <15% → 79.3% WR (23/29)"}

        # ── Strategy 5 ────────────────────────────────────────────────────────
        if (stat_type == 'marks'
                and has_strong_unders_dvp
                and line_value > 4.5):
            return {"priority": "5",
                    "description": "Mark + Strong Unders DvP + Line >4.5 → 73.3% WR (22/30)"}

        # ── Strategy 6 ────────────────────────────────────────────────────────
        if (stat_type == 'tackles'
                and position == 'GenF'
                and not has_slight_unders_dvp):
            return {"priority": "6",
                    "description": "GenF Tackle + excl Slight Unders DvP → 69.8% WR (37/53)"}

        return {"priority": "", "description": ""}

    except Exception as e:
        print(f"ERROR in calculate_bet_flag: {e}")
        return {"priority": "", "description": "ERROR"}


def add_score_to_dataframe(df, team_weather, simplified_dvp, stat_type='disposals'):
    df['Score']        = ""
    df['ScoreFactors'] = ""
    for idx, row in df.iterrows():
        sd    = calculate_score(row, team_weather, simplified_dvp, stat_type)
        sv    = sd["ScoreValue"]
        label = ("Strong Play" if sv >= 9 else "Good Play" if sv >= 6
                 else "Consider" if sv >= 3 else "Weak" if sv >= 0 else "Avoid")
        df.at[idx, 'Score']        = f"{sv:.1f} - {label}"
        df.at[idx, 'ScoreFactors'] = sd["Factors"]
    return df


STAKING_UNITS = {
    "1": "3 units", "2": "3 units",
    "3": "2 units", "4": "2 units",
    "5": "1 unit",  "6": "1 unit",
}

def add_bet_flag_to_dataframe(df, stat_type='disposals'):
    results            = df.apply(lambda row: calculate_bet_flag(row, stat_type), axis=1)
    df['Bet_Priority'] = results.apply(lambda x: x['priority'])
    df['Bet_Flag']     = results.apply(lambda x: x['description'])
    df['Units']        = df['Bet_Priority'].apply(lambda p: STAKING_UNITS.get(str(p), ""))
    return df


def add_pickem_lines_to_dataframe(df, stat_type='disposals'):
    print(f"🎯 Adding pickem lines for {stat_type}...")
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
        print(f"✅ Matched {matched}/{len(df)} players")
        return df
    except Exception as e:
        print(f"❌ Error adding pickem lines: {e}")
        df['Line'] = ""
        return df


def add_line_analysis_columns(df, stat_type='disposals'):
    print(f"📊 Adding line analysis for {stat_type}...")
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
        print(f"❌ Error adding line analysis: {e}")
        df['Avg vs Line']     = ""
        df['Line Consistency'] = ""
        return df


def process_data_for_dashboard(stat_type='disposals'):
    global dvp_data_by_stat
    try:
        print(f"▶ process_data_for_dashboard({stat_type})")

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

        home_away = {}
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

        def map_pos(pos):
            if pd.isna(pos) or pos == "":
                return "Unknown"
            for role, tags in POSITION_MAP.items():
                if pos in tags:
                    return role
            return "Unknown"

        players["position"] = players.get("namedPosition", pd.Series(["Unknown"] * len(players))).apply(map_pos)

        # ── Fallback: fill Unknown positions from historical stats data ───────
        # If a player's position is Unknown this round (missing namedPosition),
        # look up their most recently recorded namedPosition from the full CSV
        # and use that instead. Players rarely change position roles mid-season.
        unknown_mask = players["position"] == "Unknown"
        if unknown_mask.any():
            try:
                # Build a lookup: player name → most recent known role
                hist = df[["player", "round", "namedPosition"]].copy()
                hist = hist[hist["namedPosition"].notna() & (hist["namedPosition"] != "")]
                hist = hist.sort_values("round", ascending=False)
                hist["role"] = hist["namedPosition"].apply(map_pos)
                hist = hist[hist["role"] != "Unknown"]
                # Keep most recent known role per player
                hist_lookup = (
                    hist.groupby("player")["role"]
                    .first()
                    .to_dict()
                )

                def fill_position(row):
                    if row["position"] != "Unknown":
                        return row["position"]
                    # Try exact match
                    name = row["player"]
                    if name in hist_lookup:
                        return hist_lookup[name]
                    # Try case-insensitive match
                    name_lower = name.lower()
                    for hist_name, role in hist_lookup.items():
                        if hist_name.lower() == name_lower:
                            return role
                    return "Unknown"

                players["position"] = players.apply(fill_position, axis=1)
                filled = unknown_mask.sum() - (players["position"] == "Unknown").sum()
                if filled > 0:
                    print(f"✅ Filled {filled} Unknown positions from historical data")
                still_unknown = (players["position"] == "Unknown").sum()
                if still_unknown > 0:
                    print(f"⚠️  {still_unknown} players still have Unknown position (new players or data gap)")
            except Exception as e:
                print(f"⚠️  Position fallback failed: {e}")
        # ─────────────────────────────────────────────────────────────────────

        DVP_MAP = {
            "Strong Unders":   "🔴 Strong Unders",
            "Moderate Unders": "🟠 Moderate Unders",
            "Slight Unders":   "🟡 Slight Unders",
            "Strong Easy":     "🔵 Strong Easy",
            "Moderate Easy":   "🔷 Moderate Easy",
            "Slight Easy":     "🔹 Slight Easy",
        }

        def get_dvp(row):
            team = row.get('opponent', 'Unknown')
            pos  = row.get('position', 'Unknown')
            if team == 'Unknown' or pos == 'Unknown':
                return "⚠️ Unknown"
            if team in simplified_dvp and pos in simplified_dvp[team]:
                return DVP_MAP.get(simplified_dvp[team][pos]["strength"], "✅ Neutral")
            return "✅ Neutral"

        players['dvp'] = players.apply(get_dvp, axis=1)

        result = add_pickem_lines_to_dataframe(players, stat_type)
        result = add_line_analysis_columns(result, stat_type)

        result = result[['player', 'team', 'opponent', 'position', 'travel_fatigue',
                          'weather', 'dvp', 'Line', 'Avg vs Line', 'Line Consistency']].copy()

        result = add_score_to_dataframe(result, team_weather, simplified_dvp, stat_type)
        result = add_bet_flag_to_dataframe(result, stat_type)

        display = result[['player', 'team', 'opponent', 'position', 'travel_fatigue',
                           'weather', 'dvp', 'Line', 'Avg vs Line', 'Line Consistency',
                           'Bet_Priority', 'Bet_Flag', 'Units']].copy()

        display.columns = ['Player', 'Team', 'Opponent', 'Position', 'Travel Fatigue',
                           'Weather', 'DvP', 'Line', 'Avg vs Line', 'Line Consistency',
                           'Bet Priority', 'Bet Flag', 'Units']

        display = display[display['Line'] != ""].copy()

        def sort_priority(p):
            try:
                return int(p)
            except Exception:
                return 999

        display['_sort'] = display['Bet Priority'].apply(sort_priority)
        display = display.sort_values(['_sort', 'Team']).drop('_sort', axis=1)

        print(f"✅ {stat_type}: {len(display)} rows")
        return display

    except Exception as e:
        print(f"CRITICAL ERROR ({stat_type}): {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame([{
            'Player': f'Error: {e}', 'Team': '', 'Opponent': '', 'Position': '',
            'Travel Fatigue': '', 'Weather': '', 'DvP': '', 'Line': '',
            'Avg vs Line': '', 'Line Consistency': '', 'Bet Priority': '', 'Bet Flag': ''
        }])


# ===== MULTI BUILDER =====

STRATEGY_WR_MAP = {
    "1": 95.7, "2": 90.0, "3": 85.0,
    "4": 79.3,  "5": 73.3, "6": 69.8,
}
MULTI_PAYOUT = 3.20


def generate_pairings(all_bets):
    """
    Generate all two-leg pairings from a list of flagged bets.
    Scores each pairing by EV at $3.20, adjusted for correlation.
    Returns top 60 pairings sorted by EV descending.
    """
    pairings = []

    for i in range(len(all_bets)):
        for j in range(i + 1, len(all_bets)):
            a, b = all_bets[i], all_bets[j]

            wr_a = STRATEGY_WR_MAP.get(str(a.get('Bet Priority', '')), 70) / 100
            wr_b = STRATEGY_WR_MAP.get(str(b.get('Bet Priority', '')), 70) / 100

            # Skip same-team pairings (not allowed)
            if a['Team'] == b['Team']:
                continue

            game_a = frozenset([a['Team'], a['Opponent']])
            game_b = frozenset([b['Team'], b['Opponent']])
            same_game = game_a == game_b

            stat_a = str(a.get('Stat', '')).lower()
            stat_b = str(b.get('Stat', '')).lower()

            # Correlation scoring
            corr_bonus = 0.0
            warning    = ""
            if same_game:
                if stat_a == stat_b:
                    corr_bonus  = 0.08
                    corr_label  = "✅ Same game · same stat"
                elif stat_a in ('disposals', 'marks') and stat_b in ('disposals', 'marks'):
                    corr_bonus  = 0.06
                    corr_label  = "✅ Same game · disposal+mark"
                else:
                    corr_bonus  = -0.05
                    corr_label  = "⚠️ Same game · mixed stats"
                    warning     = "⚠️"
            else:
                corr_label = "— Different games"

            adj_wr = min(wr_a * wr_b + corr_bonus, 0.99)
            ev_pct = round((adj_wr * MULTI_PAYOUT - 1) * 100, 1)

            pairings.append({
                'Flag':        warning,
                'Leg 1':       f"{a['Player']} ({a['Team']}) {a.get('Stat','?')} U{a['Line']}",
                'P1':          f"P{a['Bet Priority']}",
                'Leg 2':       f"{b['Player']} ({b['Team']}) {b.get('Stat','?')} U{b['Line']}",
                'P2':          f"P{b['Bet Priority']}",
                'Correlation': corr_label,
                'Combined WR': f"{adj_wr * 100:.1f}%",
                'EV at $3.20': f"{ev_pct:+.1f}%",
                '_ev':         adj_wr * MULTI_PAYOUT - 1,
                '_p1':         a['Player'],
                '_p2':         b['Player'],
                '_units':      max(
                    int(STAKING_UNITS.get(str(a.get('Bet Priority','')), '1 unit').split()[0]),
                    int(STAKING_UNITS.get(str(b.get('Bet Priority','')), '1 unit').split()[0]),
                ),
            })

    pairings.sort(key=lambda x: x['_ev'], reverse=True)
    return pairings[:60]


def build_portfolio(pairings):
    """
    Greedy selection of non-overlapping pairings — no player used twice.
    Returns the best set of multis for the round.
    """
    used = set()
    portfolio = []
    for pair in pairings:   # already sorted by EV descending
        p1, p2 = pair['_p1'], pair['_p2']
        if p1 not in used and p2 not in used:
            portfolio.append(pair)
            used.add(p1)
            used.add(p2)
    return portfolio


def build_multi_builder_layout():
    # Collect all flagged bets across all stat types
    all_bets = []
    for stat_type, label in [('disposals', 'Disposals'), ('marks', 'Marks'), ('tackles', 'Tackles')]:
        df = processed_data_by_stat.get(stat_type)
        if df is None or df.empty:
            continue
        flagged = df[df['Bet Priority'].astype(str).str.strip() != ''].copy()
        flagged['Stat'] = label
        all_bets.append(flagged)

    if not all_bets:
        return dbc.Alert(
            "No flagged bets found. Load data first or check that Dabble markets are open.",
            color="warning", className="mt-3"
        )

    combined = pd.concat(all_bets, ignore_index=True)

    # ── Individual legs table ────────────────────────────────────────────────
    legs_display = combined[['Bet Priority', 'Units', 'Stat', 'Player', 'Team',
                              'Opponent', 'Position', 'Line', 'DvP',
                              'Avg vs Line', 'Bet Flag']].copy()

    priority_colours = {
        "1": "#28a745", "2": "#28a745",
        "3": "#ffc107", "4": "#ffc107",
        "5": "#17a2b8", "6": "#17a2b8",
    }
    units_colours = {
        "3 units": "#28a745", "2 units": "#ffc107", "1 unit": "#17a2b8",
    }

    legs_style = []
    for p, col in priority_colours.items():
        legs_style.append({
            'if': {'column_id': 'Bet Priority', 'filter_query': f'{{Bet Priority}} = "{p}"'},
            'backgroundColor': col, 'color': 'white' if p in ('1','2','5','6') else 'black',
            'fontWeight': 'bold',
        })
    for u, col in units_colours.items():
        legs_style.append({
            'if': {'column_id': 'Units', 'filter_query': f'{{Units}} = "{u}"'},
            'backgroundColor': col, 'color': 'white' if u in ('3 units', '1 unit') else 'black',
            'fontWeight': 'bold', 'textAlign': 'center',
        })

    legs_table = dash_table.DataTable(
        data=legs_display.to_dict('records'),
        columns=[{"name": c, "id": c} for c in legs_display.columns],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "7px 10px", "fontFamily": "Arial", "fontSize": "13px"},
        style_header={"backgroundColor": "#343a40", "color": "white", "fontWeight": "bold"},
        style_data_conditional=legs_style,
        sort_action="native",
        page_size=25,
    )

    # ── Pairings table ───────────────────────────────────────────────────────
    bets_list = combined.to_dict('records')
    pairings  = generate_pairings(bets_list)

    if not pairings:
        pairings_section = dbc.Alert("Not enough flagged bets to build pairings.", color="info")
    else:
        pairs_df = pd.DataFrame([{k: v for k, v in p.items() if not k.startswith('_')} for p in pairings])

        pair_style = [
            # Green EV
            {'if': {'filter_query': '{EV at $3.20} contains "+"', 'column_id': 'EV at $3.20'},
             'color': '#28a745', 'fontWeight': 'bold'},
            # Red EV
            {'if': {'filter_query': '{EV at $3.20} contains "-"', 'column_id': 'EV at $3.20'},
             'color': '#dc3545', 'fontWeight': 'bold'},
            # Warning rows
            {'if': {'filter_query': '{Flag} = "⚠️"'},
             'backgroundColor': '#fff3cd'},
            # P1-2 priority colouring
            {'if': {'filter_query': '{P1} = "P1" || {P1} = "P2"', 'column_id': 'P1'},
             'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{P2} = "P1" || {P2} = "P2"', 'column_id': 'P2'},
             'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{P1} = "P3" || {P1} = "P4"', 'column_id': 'P1'},
             'backgroundColor': '#ffc107', 'color': 'black', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{P2} = "P3" || {P2} = "P4"', 'column_id': 'P2'},
             'backgroundColor': '#ffc107', 'color': 'black', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{P1} = "P5" || {P1} = "P6"', 'column_id': 'P1'},
             'backgroundColor': '#17a2b8', 'color': 'white', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{P2} = "P5" || {P2} = "P6"', 'column_id': 'P2'},
             'backgroundColor': '#17a2b8', 'color': 'white', 'fontWeight': 'bold'},
        ]

        pairings_section = dash_table.DataTable(
            data=pairs_df.to_dict('records'),
            columns=[{"name": c, "id": c} for c in pairs_df.columns],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "7px 10px", "fontFamily": "Arial", "fontSize": "13px"},
            style_header={"backgroundColor": "#343a40", "color": "white", "fontWeight": "bold"},
            style_data_conditional=pair_style,
            sort_action="native",
            page_size=20,
        )

    # ── Portfolio (non-overlapping multis) ──────────────────────────────────
    portfolio     = build_portfolio(pairings) if pairings else []
    total_units   = sum(p['_units'] for p in portfolio)
    total_ev_pct  = round(sum(p['_ev'] for p in portfolio) / max(len(portfolio), 1) * 100, 1) if portfolio else 0

    priority_badge = {
        'P1': 'success', 'P2': 'success',
        'P3': 'warning',  'P4': 'warning',
        'P5': 'info',     'P6': 'info',
    }

    portfolio_cards = []
    for idx, p in enumerate(portfolio, 1):
        portfolio_cards.append(
            dbc.Card([
                dbc.CardHeader(
                    dbc.Row([
                        dbc.Col(html.Strong(f"Multi {idx}"), width="auto"),
                        dbc.Col(
                            html.Span(f"EV {p['EV at $3.20']}",
                                      className=f"badge bg-{'success' if '+' in p['EV at $3.20'] else 'danger'}"),
                            width="auto"
                        ),
                        dbc.Col(
                            html.Span(f"{p['_units']} unit{'s' if p['_units'] != 1 else ''}",
                                      className=f"badge bg-{'success' if p['_units']==3 else 'warning' if p['_units']==2 else 'info'} text-{'dark' if p['_units']==2 else 'white'}"),
                            width="auto"
                        ),
                        dbc.Col(
                            html.Small(p['Correlation'], className="text-muted"),
                            width="auto"
                        ),
                    ], align="center", className="g-2")
                ),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Span(p['P1'], className=f"badge bg-{priority_badge.get(p['P1'],'secondary')} me-2"),
                            html.Strong(p['Leg 1']),
                        ], className="mb-1"),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Span(p['P2'], className=f"badge bg-{priority_badge.get(p['P2'],'secondary')} me-2"),
                            html.Strong(p['Leg 2']),
                        ]),
                    ]),
                ], className="py-2"),
            ], className="mb-2 shadow-sm",
               color="warning" if p['Flag'] == '⚠️' else "light",
               outline=True)
        )

    portfolio_section = html.Div([
        dbc.Alert([
            html.Strong(f"📋 Round portfolio — {len(portfolio)} multis · {total_units} total units · avg EV {total_ev_pct:+.1f}%"),
            html.Span("  No player appears twice. Place these as separate two-leg multis.",
                      className="text-muted small ms-2"),
        ], color="success" if portfolio else "secondary", className="mb-3"),
        html.Div(portfolio_cards) if portfolio_cards else
            dbc.Alert("Not enough flagged bets to build a portfolio.", color="info"),
    ])

    # ── Staking legend ───────────────────────────────────────────────────────
    staking_legend = dbc.Alert([
        html.Strong("Staking guide: "),
        html.Span("3 units — P1/P2 (90–96% WR)  ", className="badge bg-success me-2"),
        html.Span("2 units — P3/P4 (79–85% WR)  ", className="badge bg-warning text-dark me-2"),
        html.Span("1 unit  — P5/P6 (70–73% WR)  ", className="badge bg-info me-2"),
        "  ·  EV calculated at $3.20 payout  ·  ⚠️ yellow = avoid (mixed stat types same game)",
    ], color="light", className="mb-3 small")

    return html.Div([
        staking_legend,
        html.H5("🎯 Recommended multi portfolio", className="mb-2 mt-2"),
        portfolio_section,
        html.Hr(),
        html.H5(f"All flagged legs ({len(combined)})", className="mb-2 mt-3"),
        legs_table,
        html.Hr(),
        html.H5(f"All valid pairings (top {len(pairings)}, ranked by EV)", className="mb-2 mt-3"),
        pairings_section,
    ])


# ===== DASH APP =====
app    = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

processed_data_by_stat = {'disposals': None, 'marks': None, 'tackles': None}

# Stores simplified_dvp per stat type so edits can recalculate DvP
dvp_data_by_stat = {'disposals': {}, 'marks': {}, 'tackles': {}}

# Tracks when each stat type was last pulled from Dabble
last_pulled_at = {'disposals': None, 'marks': None, 'tackles': None}
current_round_number   = None   # set when data loads, used by Sheets push

app.layout = dbc.Container([
    html.H1("AFL Player Dashboard - Next Round", className="text-center my-4"),

    html.Div(id='loaded-data',        style={'display': 'none'}),
    html.Div(id='sheets-push-status', style={'display': 'none'}),
    dcc.Download(id="download-csv"),

    # Auto-refresh performance tracker every 5 minutes
    dcc.Interval(id='perf-interval', interval=5*60*1000, n_intervals=0),

    dbc.Tabs([
        dbc.Tab(label="Disposals",            tab_id="tab-disposals",    labelClassName="fw-bold"),
        dbc.Tab(label="Marks",                tab_id="tab-marks",        labelClassName="fw-bold"),
        dbc.Tab(label="Tackles",              tab_id="tab-tackles",      labelClassName="fw-bold"),
        dbc.Tab(label="🎯 Multi Builder",     tab_id="tab-multi",        labelClassName="fw-bold text-success"),
        dbc.Tab(label="📊 Performance",       tab_id="tab-performance",  labelClassName="fw-bold text-primary"),
    ], id="stat-tabs", active_tab="tab-disposals", className="mb-3"),

    dbc.Row([
        # ── filters ──────────────────────────────────────────────────────────
        dbc.Col([
            html.Div([
                html.H5("Filter by Team:"),
                dcc.Dropdown(id="team-filter", placeholder="Select team(s)...",
                             clearable=True, multi=True),
                html.Button("Clear Team Filter", id="clear-team-filter",
                            className="btn btn-outline-secondary btn-sm mt-1"),
            ], className="mb-4"),
            html.Div([
                html.H5("Filter by Position:"),
                dcc.Dropdown(id="position-filter", clearable=True, options=[
                    {"label": "Key Forward",     "value": "KeyF"},
                    {"label": "General Forward", "value": "GenF"},
                    {"label": "Ruck",            "value": "Ruck"},
                    {"label": "Inside Mid",      "value": "InsM"},
                    {"label": "Wing",            "value": "Wing"},
                    {"label": "General Defender","value": "GenD"},
                    {"label": "Key Defender",    "value": "KeyD"},
                ], placeholder="Select a position..."),
            ], className="mb-4"),
        ], width=3),

        # ── legend cards ─────────────────────────────────────────────────────
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Travel Fatigue", className="text-center"),
                        html.Div([
                            html.Span("✅ Neutral",     className="badge bg-success me-2"),
                            html.Span("🟠 Long Travel", className="badge bg-warning me-2"),
                        ], className="d-flex justify-content-center")
                    ], className="border rounded p-2 mb-3", id="travel-fatigue-legend",
                    title="Long travel (>1500 km away) = fatigue flag. Short Break = kill-switch, overs signal."),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.H4("Weather", className="text-center"),
                        html.Div([
                            html.Span("✅ Neutral",            className="badge bg-success me-2"),
                            html.Span("⚠️ Medium Unders Edge", className="badge bg-warning me-2"),
                            html.Span("🔴 Strong Unders Edge", className="badge bg-danger me-2"),
                            html.Span("🔵 Avoid",              className="badge bg-primary"),
                        ], className="d-flex justify-content-center flex-wrap")
                    ], className="border rounded p-2 mb-3", id="weather-legend",
                    title="Rain+Wind suppress disposals/marks. Rain on tackles = Avoid (rain boosts tackle counts)."),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("DvP", className="text-center"),
                        html.Div([
                            html.Span("✅ Neutral",       className="badge bg-success me-2"),
                            html.Span("🟡🟠🔴 Unders",   className="badge bg-warning text-dark me-2"),
                            html.Span("🔹🔷🔵 Easy",     className="badge bg-info me-2"),
                        ], className="d-flex justify-content-center flex-wrap")
                    ], className="border rounded p-2 mb-3", id="dvp-legend",
                    title="Defence vs Position. Unders = opponent suppresses the role. Easy = opponent leaks stats. Slight < Moderate < Strong."),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.H4("Bet Strategies", className="text-center"),
                        html.Div([
                            html.Span("P1–2: 90–96% WR", className="badge bg-success me-2"),
                            html.Span("P3–4: 79–85% WR", className="badge bg-warning text-dark me-2"),
                            html.Span("P5–6: 70–73% WR", className="badge bg-info"),
                        ], className="d-flex justify-content-center flex-wrap")
                    ], className="border rounded p-2 mb-3", id="bet-strategy-legend",
                    title=(
                        "P1: Tackle Mod Travel 95.7% | "
                        "P2: Mark multi-confirm 90.0% | "
                        "P3: KeyF Mark 85.0% | "
                        "P4: Tackle Strong Unders 79.3% | "
                        "P5: Mark Strong Unders 73.3% | "
                        "P6: GenF Tackle 69.8% | "
                        "Short Break suppresses all."
                    )),
                ], width=6),
            ]),
        ], width=9),
    ], className="mb-4"),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            html.Div(id="loading-message", children="Loading player data...",
                     className="text-center fs-4"),
            # ── Last pulled indicator ─────────────────────────────────────────
            html.Div(id="last-pulled-message",
                     className="text-center text-muted small mt-1"),
        ], width=6),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    # ── Check for new markets button ──────────────────────────
                    html.Button(
                        "🔄 Check for New Markets",
                        id="refresh-markets-button",
                        className="btn btn-warning w-100",
                        title="Re-scrapes Dabble for updated lines across all 3 stat types. Use this Thu–Sun when new markets open without restarting the dashboard.",
                    ),
                ], width=4),
                dbc.Col([
                    # ── Push to Sheets button ─────────────────────────────────
                    html.Button(
                        "📤 Push to Google Sheets",
                        id="push-sheets-button",
                        className="btn btn-primary w-100",
                        title="Appends all flagged bets (all 3 stat types) to your master Google Sheet. Already-pushed rows are skipped automatically.",
                    ),
                ], width=4),
                dbc.Col([
                    html.Button(
                        "⬇ Export CSV",
                        id="export-button",
                        className="btn btn-success w-100",
                    ),
                ], width=4),
            ]),
            # status messages
            html.Div(id="refresh-markets-message", className="text-center mt-2 small"),
            html.Div(id="sheets-push-message",     className="text-center mt-1 small"),
        ], width=6),
    ], className="mb-3"),

    # ── Bet dashboard (shown for Disposals / Marks / Tackles tabs) ──────────
    html.Div(id="bet-dashboard-content", children=[
        dbc.Alert(
            [
                html.Strong("✏️  Validation step: "),
                "Position and Line cells are editable — click any cell to correct it. ",
                "Bet Priority and Bet Flag recalculate instantly. ",
                "Edited rows are highlighted in ",
                html.Strong("blue", style={"color": "#0d6efd"}),
                ". Validate before pushing to Sheets.",
            ],
            color="info",
            className="mb-2 py-2 small",
        ),
        dash_table.DataTable(
            id="player-table",
            data=[],
            columns=[],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px", "fontFamily": "Arial"},
            style_header={"backgroundColor": "#343a40", "color": "white",
                          "fontWeight": "bold", "textAlign": "center"},
            style_data_conditional=[],
            dropdown={
                "Position": {
                    "options": [
                        {"label": "KeyF — Key Forward",      "value": "KeyF"},
                        {"label": "GenF — General Forward",  "value": "GenF"},
                        {"label": "Ruck",                    "value": "Ruck"},
                        {"label": "InsM — Inside Mid",       "value": "InsM"},
                        {"label": "Wing",                    "value": "Wing"},
                        {"label": "GenD — General Defender", "value": "GenD"},
                        {"label": "KeyD — Key Defender",     "value": "KeyD"},
                        {"label": "Unknown",                 "value": "Unknown"},
                    ]
                }
            },
            page_size=20,
            sort_action="native",
            filter_action="native",
        ),
    ]),

    # ── Multi builder (shown for Multi Builder tab) ───────────────────────
    html.Div(id="multi-builder-content"),

    # ── Performance tracker (shown for Performance tab) ───────────────────
    html.Div(id="performance-content"),

], fluid=True)


# ── load data on startup ──────────────────────────────────────────────────────
@app.callback(
    Output('loaded-data', 'children'),
    Input('loaded-data',  'children'),
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
                        'Bet Priority': '1', 'Bet Flag': 'Test flag',
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

            return "Data loaded"
        except Exception as e:
            print(f"Load error: {e}")
            return "Error loading data"
    return data


# ── push to Google Sheets ─────────────────────────────────────────────────────
@app.callback(
    Output('sheets-push-message', 'children'),
    Input('push-sheets-button',   'n_clicks'),
    prevent_initial_call=True,
)
def handle_push_to_sheets(n_clicks):
    if n_clicks is None:
        return ""

    disposals_df = processed_data_by_stat.get('disposals')
    marks_df     = processed_data_by_stat.get('marks')
    tackles_df   = processed_data_by_stat.get('tackles')

    if all(df is None for df in [disposals_df, marks_df, tackles_df]):
        return "⚠️ Data not loaded yet — please wait and try again."

    rows_added, message = push_to_google_sheets(
        disposals_df, marks_df, tackles_df, current_round_number
    )
    return message


# ── table update ─────────────────────────────────────────────────────────────
# Store edited row indices per stat type so we can highlight them
edited_rows_by_stat = {'disposals': set(), 'marks': set(), 'tackles': set()}

@app.callback(
    [Output('player-table',       'data'),
     Output('player-table',       'columns'),
     Output('player-table',       'style_data_conditional'),
     Output('team-filter',        'options'),
     Output('loading-message',    'children')],
    [Input('stat-tabs',           'active_tab'),
     Input('team-filter',         'value'),
     Input('position-filter',     'value'),
     Input('clear-team-filter',   'n_clicks'),
     Input('loaded-data',         'children'),
     Input('player-table',        'data_timestamp')],
    [State('player-table',        'data'),
     State('player-table',        'active_cell')],
)
def update_table(active_tab, team_filter, position, clear_clicks, loaded_data,
                 data_timestamp, table_data, active_cell):
    if loaded_data != "Data loaded":
        return [], [], [], [], "Loading data..."

    stat_type = {'tab-marks': 'marks', 'tab-tackles': 'tackles'}.get(active_tab, 'disposals')
    data      = processed_data_by_stat.get(stat_type)
    if data is None:
        return [], [], [], [], f"No data for {stat_type}"

    try:
        ctx = dash.callback_context
        triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else ''

        if 'clear-team-filter' in triggered:
            team_filter = None

        # ── Handle cell edits ─────────────────────────────────────────────────
        # If the trigger was a cell edit (data_timestamp fired and we have
        # table_data), apply the user's change back to the master dataframe
        # and recalculate Bet Priority + Bet Flag for the edited row.
        if 'data_timestamp' in triggered and table_data:
            edited_df = pd.DataFrame(table_data)

            # Find which rows changed vs the stored master
            master = processed_data_by_stat[stat_type]
            for i, row in edited_df.iterrows():
                player = row.get('Player', '')
                # Match back to master by Player + Team (unique enough)
                mask = (master['Player'] == player) & (master['Team'] == row.get('Team', ''))
                if not mask.any():
                    continue

                changed = False
                for col in ['Position', 'Line']:
                    if col in master.columns and col in row:
                        if str(master.loc[mask, col].values[0]) != str(row[col]):
                            master.loc[mask, col] = row[col]
                            changed = True

                if changed:
                    # Recalculate Avg vs Line if Line changed
                    new_line = str(row.get('Line', ''))
                    if new_line:
                        try:
                            stats_raw = pd.read_csv("afl_player_stats.csv", skiprows=3).fillna(0)
                            if stat_type not in stats_raw.columns and stat_type == 'disposals':
                                stats_raw[stat_type] = stats_raw.get('kicks', 0) + stats_raw.get('handballs', 0)
                            ps = stats_raw[stats_raw['player'].str.lower() == player.lower()]
                            if not ps.empty and stat_type in ps.columns:
                                avg = ps[stat_type].mean()
                                if avg > 0:
                                    avl = ((float(new_line) / avg) - 1) * 100
                                    master.loc[mask, 'Avg vs Line'] = f"{avl:+.1f}%"
                                below = sum(1 for v in ps[stat_type].values if v < float(new_line))
                                master.loc[mask, 'Line Consistency'] = f"{(below / len(ps)) * 100:.1f}%"
                        except Exception:
                            pass

                    # Recalculate DvP if Position changed
                    new_pos = str(row.get('Position', ''))
                    if new_pos:
                        try:
                            opponent = str(master.loc[mask, 'Opponent'].values[0])
                            sdvp = dvp_data_by_stat.get(stat_type, {})
                            DVP_MAP = {
                                "Strong Unders":   "🔴 Strong Unders",
                                "Moderate Unders": "🟠 Moderate Unders",
                                "Slight Unders":   "🟡 Slight Unders",
                                "Strong Easy":     "🔵 Strong Easy",
                                "Moderate Easy":   "🔷 Moderate Easy",
                                "Slight Easy":     "🔹 Slight Easy",
                            }
                            if opponent and opponent != 'Unknown' and new_pos != 'Unknown':
                                if opponent in sdvp and new_pos in sdvp[opponent]:
                                    new_dvp = DVP_MAP.get(sdvp[opponent][new_pos]["strength"], "✅ Neutral")
                                else:
                                    new_dvp = "✅ Neutral"
                            else:
                                new_dvp = "⚠️ Unknown"
                            master.loc[mask, 'DvP'] = new_dvp
                        except Exception:
                            pass

                    # Recalculate bet flag for this row
                    updated_row = master.loc[mask].iloc[0]
                    result = calculate_bet_flag(updated_row, stat_type)
                    master.loc[mask, 'Bet Priority'] = result['priority']
                    master.loc[mask, 'Bet Flag']     = result['description']

                    # Track this row index for blue highlight
                    edited_rows_by_stat[stat_type].add(i)

            # Write changes back to the global store
            processed_data_by_stat[stat_type] = master

        df = processed_data_by_stat[stat_type].copy()
        if team_filter:
            df = df[df['Team'].isin(team_filter)]
        if position:
            df = df[df['Position'] == position]

        team_options = [{'label': t, 'value': t} for t in sorted(processed_data_by_stat[stat_type]['Team'].unique())]

        # ── Build columns — Position and Line are editable ────────────────────
        EDITABLE_COLS = {'Position', 'Line'}
        columns = [
            {"name": c, "id": c, "editable": c in EDITABLE_COLS,
             "presentation": "dropdown" if c == "Position" else "input"}
            for c in df.columns
        ]

        style = [
            # Travel Fatigue
            {'if': {'column_id': 'Travel Fatigue', 'filter_query': '{Travel Fatigue} contains "Neutral"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
            {'if': {'column_id': 'Travel Fatigue', 'filter_query': '{Travel Fatigue} contains "Long Travel"'},
             'backgroundColor': '#ffecb3', 'color': 'black'},
            # Weather
            {'if': {'column_id': 'Weather', 'filter_query': '{Weather} contains "Neutral"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
            {'if': {'column_id': 'Weather', 'filter_query': '{Weather} contains "Medium"'},
             'backgroundColor': '#fff3cd', 'color': 'black'},
            {'if': {'column_id': 'Weather', 'filter_query': '{Weather} contains "Strong"'},
             'backgroundColor': '#f8d7da', 'color': 'black'},
            {'if': {'column_id': 'Weather', 'filter_query': '{Weather} contains "🔵 Avoid"'},
             'backgroundColor': '#cce5ff', 'color': 'black'},
            # DvP
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Neutral"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Slight"'},
             'backgroundColor': '#fff3cd', 'color': 'black'},
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Moderate"'},
             'backgroundColor': '#ffe066', 'color': 'black'},
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Strong"'},
             'backgroundColor': '#f8d7da', 'color': 'black'},
            # Line
            {'if': {'column_id': 'Line', 'filter_query': '{Line} != ""'},
             'backgroundColor': '#e8f5e8', 'color': 'black', 'fontWeight': 'bold'},
            # Bet Priority  — green P1–2, amber P3–4, blue P5–6
            {'if': {'column_id': 'Bet Priority', 'filter_query': '{Bet Priority} = "1"'},
             'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Priority', 'filter_query': '{Bet Priority} = "2"'},
             'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Priority', 'filter_query': '{Bet Priority} = "3"'},
             'backgroundColor': '#ffc107', 'color': 'black', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Priority', 'filter_query': '{Bet Priority} = "4"'},
             'backgroundColor': '#ffc107', 'color': 'black', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Priority', 'filter_query': '{Bet Priority} = "5"'},
             'backgroundColor': '#17a2b8', 'color': 'white', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Priority', 'filter_query': '{Bet Priority} = "6"'},
             'backgroundColor': '#17a2b8', 'color': 'white', 'fontWeight': 'bold'},
            # Bet Flag — keyed on win-rate % in the description string
            {'if': {'column_id': 'Bet Flag', 'filter_query': '{Bet Flag} contains "95.7%"'},
             'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Flag', 'filter_query': '{Bet Flag} contains "90.0%"'},
             'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Flag', 'filter_query': '{Bet Flag} contains "85.0%"'},
             'backgroundColor': '#ffc107', 'color': 'black', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Flag', 'filter_query': '{Bet Flag} contains "79.3%"'},
             'backgroundColor': '#ffc107', 'color': 'black', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Flag', 'filter_query': '{Bet Flag} contains "73.3%"'},
             'backgroundColor': '#17a2b8', 'color': 'white', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Flag', 'filter_query': '{Bet Flag} contains "69.8%"'},
             'backgroundColor': '#17a2b8', 'color': 'white', 'fontWeight': 'bold'},
        ]

        # ── Blue highlight for edited rows ────────────────────────────────────
        edited = edited_rows_by_stat.get(stat_type, set())
        for row_idx in edited:
            style.append({
                'if': {'row_index': row_idx},
                'backgroundColor': '#cfe2ff',
                'border': '2px solid #0d6efd',
            })

        # ── Position dropdown options ─────────────────────────────────────────
        position_options = {
            "Position": {
                "options": [
                    {"label": "KeyF — Key Forward",     "value": "KeyF"},
                    {"label": "GenF — General Forward", "value": "GenF"},
                    {"label": "Ruck",                   "value": "Ruck"},
                    {"label": "InsM — Inside Mid",      "value": "InsM"},
                    {"label": "Wing",                   "value": "Wing"},
                    {"label": "GenD — General Defender","value": "GenD"},
                    {"label": "KeyD — Key Defender",    "value": "KeyD"},
                    {"label": "Unknown",                "value": "Unknown"},
                ]
            }
        }

        return df.to_dict('records'), columns, style, team_options, ""

    except Exception as e:
        print(f"Table update error ({stat_type}): {e}")
        err = pd.DataFrame([{'Player': f'Error: {e}', 'Team': '', 'Opponent': '',
                              'Position': '', 'Travel Fatigue': '', 'Weather': '',
                              'DvP': '', 'Line': '', 'Bet Priority': '', 'Bet Flag': ''}])
        return err.to_dict('records'), [{"name": c, "id": c} for c in err.columns], [], [], str(e)


# ── Market refresh callback ───────────────────────────────────────────────────
# Tracks when the last successful Dabble scrape ran — used for cooldown
_last_refresh_time = None
REFRESH_COOLDOWN_SECONDS = 300   # 5 minutes — safe buffer for Dabble


@app.callback(
    Output('refresh-markets-message', 'children'),
    Output('loaded-data',             'children',  allow_duplicate=True),
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
        return "", "Data loaded"

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
            for col in ['Line', 'Avg vs Line', 'Line Consistency',
                        'Bet Priority', 'Bet Flag']:
                if col in df.columns:
                    df[col] = ''

            # Rename back to lowercase for the helper functions
            col_map_down = {
                'Player': 'player', 'Team': 'team', 'Opponent': 'opponent',
                'Position': 'position', 'Travel Fatigue': 'travel_fatigue',
                'Weather': 'weather', 'DvP': 'dvp',
            }
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
                'Bet_Priority': 'Bet Priority', 'Bet_Flag': 'Bet Flag',
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
                         'Avg vs Line', 'Line Consistency',
                         'Bet Priority', 'Bet Flag']
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

    summary = " · ".join(results)
    return (
        f"✅ Markets refreshed at {pulled_time} — {summary}",
        "Data loaded",
    )


# ── Last-pulled indicator callback ────────────────────────────────────────────
@app.callback(
    Output('last-pulled-message', 'children'),
    [Input('loaded-data',             'children'),
     Input('refresh-markets-message', 'children'),
     Input('stat-tabs',               'active_tab')],
)
def update_last_pulled(loaded_data, refresh_msg, active_tab):
    """Show when data was last pulled from Dabble for the current tab."""
    if loaded_data != "Data loaded":
        return ""

    stat_type = {'tab-marks': 'marks', 'tab-tackles': 'tackles'}.get(active_tab, 'disposals')

    if active_tab == 'tab-performance':
        return ""

    pulled = last_pulled_at.get(stat_type)
    if not pulled:
        return "📡 Data not yet loaded"

    return f"📡 Dabble lines last pulled at {pulled}"


# ── Performance tracker helpers ───────────────────────────────────────────────

STRATEGY_LABELS = {
    "1": "P1 · Tackle Mod Travel",
    "2": "P2 · Mark multi-confirm",
    "3": "P3 · KeyF Mark",
    "4": "P4 · Tackle Strong Unders",
    "5": "P5 · Mark Strong Unders",
    "6": "P6 · GenF Tackle",
}

STRATEGY_TARGET_WR = {
    "1": 95.7, "2": 90.0, "3": 85.0,
    "4": 79.3, "5": 73.3, "6": 69.8,
}


def load_performance_data():
    """
    Read the master Google Sheet and return a cleaned dataframe
    containing only rows that have both a Bet Priority and a W/L result.
    Returns None if the sheet cannot be read.
    """
    client = get_sheets_client()
    if client is None:
        return None
    try:
        sheet     = client.open_by_key(GOOGLE_SHEET_ID)
        worksheet = sheet.worksheet(GOOGLE_SHEET_TAB)
        records   = worksheet.get_all_records()
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        # Keep only rows with a strategy flag and a result
        df = df[df['Bet Priority'].astype(str).str.strip().isin(['1','2','3','4','5','6'])]
        df = df[df['W/L'].astype(str).str.strip().isin(['1', '-1', '0', '1.0', '-1.0', '0.0'])]
        df['W/L']         = pd.to_numeric(df['W/L'],         errors='coerce')
        df['Round']       = pd.to_numeric(df['Round'],       errors='coerce')
        df['Bet Priority'] = df['Bet Priority'].astype(str).str.strip()
        df['Type']        = df['Type'].astype(str).str.strip()
        return df
    except Exception as e:
        print(f"⚠️  Performance data load error: {e}")
        return None


def make_metric_card(label, value, sub=None, color="#198754"):
    """Small summary stat card."""
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": "#6c757d",
                               "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        html.Div(value, style={"fontSize": "24px", "fontWeight": "500", "color": color}),
        html.Div(sub,   style={"fontSize": "11px", "color": "#6c757d"}) if sub else None,
    ], style={"background": "#f8f9fa", "borderRadius": "8px",
              "padding": "12px 16px", "flex": "1"})


def build_performance_layout(df):
    """Build the full performance tab content from the results dataframe."""
    if df is None:
        return dbc.Alert("⚠️ Could not connect to Google Sheets. "
                         "Check your credentials file.", color="warning")
    if df.empty:
        return dbc.Alert("No completed bets found in your sheet yet. "
                         "Results will appear here once W/L is filled in "
                         "(run update_results.py mid-week).", color="info")

    total_bets = len(df)
    wins       = (df['W/L'] == 1).sum()
    losses     = (df['W/L'] == -1).sum()
    pushes     = (df['W/L'] == 0).sum()
    wr_overall = round(wins / (total_bets - pushes) * 100, 1) if (total_bets - pushes) > 0 else 0

    # EV at $3.20 two-leg multi — approx each leg independently
    ev_pct = round((wr_overall/100)**2 * 3.2 * 100 - 100, 1)

    # ── Summary cards ─────────────────────────────────────────────────────────
    summary_row = html.Div([
        make_metric_card("Total bets",   str(total_bets), f"{wins}W · {losses}L · {pushes}P"),
        make_metric_card("Overall W/R",  f"{wr_overall}%", "excl. pushes",
                         "#198754" if wr_overall >= 65 else "#dc3545"),
        make_metric_card("2-leg EV",     f"{ev_pct:+.0f}%", "at $3.20 payout",
                         "#198754" if ev_pct > 0 else "#dc3545"),
        make_metric_card("Rounds tracked", str(int(df['Round'].nunique())),
                         f"R{int(df['Round'].min())}–R{int(df['Round'].max())}"),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "24px", "flexWrap": "wrap"})

    # ── Per-strategy table ────────────────────────────────────────────────────
    strat_rows = []
    for prio in ['1','2','3','4','5','6']:
        sub = df[df['Bet Priority'] == prio]
        if sub.empty:
            continue
        s_total  = len(sub)
        s_wins   = (sub['W/L'] == 1).sum()
        s_push   = (sub['W/L'] == 0).sum()
        s_wr     = round(s_wins / (s_total - s_push) * 100, 1) if (s_total - s_push) > 0 else 0
        target   = STRATEGY_TARGET_WR.get(prio, 0)
        vs_target = round(s_wr - target, 1)
        trend    = "↑" if vs_target >= 0 else "↓"
        color    = "#198754" if vs_target >= 0 else "#dc3545"
        # rolling last 10
        recent   = sub.tail(10)
        r_wins   = (recent['W/L'] == 1).sum()
        r_push   = (recent['W/L'] == 0).sum()
        r_wr     = round(r_wins / (len(recent) - r_push) * 100, 1) if (len(recent) - r_push) > 0 else 0

        strat_rows.append(html.Tr([
            html.Td(STRATEGY_LABELS.get(prio, prio), style={"fontWeight": "500"}),
            html.Td(f"{s_total}"),
            html.Td(f"{s_wins}W – {s_total - s_wins - s_push}L"),
            html.Td(f"{s_wr}%",     style={"fontWeight": "500",
                                            "color": "#198754" if s_wr >= target else "#dc3545"}),
            html.Td(f"{target}%",   style={"color": "#6c757d"}),
            html.Td(f"{trend} {abs(vs_target)}pp", style={"color": color, "fontWeight": "500"}),
            html.Td(f"{r_wr}%",     style={"color": "#0d6efd"}),
        ]))

    strategy_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Strategy"),
            html.Th("Bets"),
            html.Th("Record"),
            html.Th("Season W/R"),
            html.Th("Target W/R"),
            html.Th("vs Target"),
            html.Th("Last 10 W/R"),
        ]), style={"backgroundColor": "#343a40", "color": "white"}),
        html.Tbody(strat_rows),
    ], bordered=True, hover=True, responsive=True, size="sm", className="mb-4")

    # ── Win rate by round chart ───────────────────────────────────────────────
    round_stats = []
    for rnd in sorted(df['Round'].dropna().unique()):
        sub   = df[df['Round'] == rnd]
        w     = (sub['W/L'] == 1).sum()
        p     = (sub['W/L'] == 0).sum()
        total = len(sub) - p
        wr    = round(w / total * 100, 1) if total > 0 else 0
        round_stats.append({"Round": int(rnd), "WR": wr, "Bets": int(len(sub))})
    round_df = pd.DataFrame(round_stats)

    round_chart = go.Figure()
    if not round_df.empty:
        round_chart.add_trace(go.Bar(
            x=round_df['Round'], y=round_df['Bets'],
            name="Bets placed", marker_color="#dee2e6",
            yaxis="y2", opacity=0.6,
        ))
        round_chart.add_trace(go.Scatter(
            x=round_df['Round'], y=round_df['WR'],
            mode="lines+markers", name="Win rate %",
            line=dict(color="#198754", width=2),
            marker=dict(size=7),
        ))
        round_chart.add_hline(y=65, line_dash="dash", line_color="#dc3545",
                              annotation_text="65% threshold")
    round_chart.update_layout(
        title="Win rate by round",
        xaxis_title="Round",
        yaxis=dict(title="Win rate %", range=[0, 110]),
        yaxis2=dict(title="Bets", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.2),
        height=320, margin=dict(t=40, b=40, l=40, r=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )

    # ── Stat type breakdown ───────────────────────────────────────────────────
    type_rows = []
    for t in ['Disposal', 'Mark', 'Tackle']:
        sub = df[df['Type'] == t]
        if sub.empty:
            continue
        w  = (sub['W/L'] == 1).sum()
        p  = (sub['W/L'] == 0).sum()
        wr = round(w / (len(sub) - p) * 100, 1) if (len(sub) - p) > 0 else 0
        type_rows.append(html.Tr([
            html.Td(t, style={"fontWeight": "500"}),
            html.Td(str(len(sub))),
            html.Td(f"{w}W – {len(sub)-w-p}L"),
            html.Td(f"{wr}%", style={"fontWeight": "500",
                                      "color": "#198754" if wr >= 65 else "#dc3545"}),
        ]))

    type_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Stat type"), html.Th("Bets"),
            html.Th("Record"),    html.Th("Win rate"),
        ]), style={"backgroundColor": "#343a40", "color": "white"}),
        html.Tbody(type_rows),
    ], bordered=True, hover=True, responsive=True, size="sm")

    # ── Rolling last-10 per strategy chart ────────────────────────────────────
    rolling_fig = go.Figure()
    colours = {"1":"#198754","2":"#20c997","3":"#ffc107",
               "4":"#fd7e14","5":"#0d6efd","6":"#6f42c1"}
    for prio in ['1','2','3','4','5','6']:
        sub = df[df['Bet Priority'] == prio].copy()
        if len(sub) < 3:
            continue
        sub = sub.reset_index(drop=True)
        rolling_wr = []
        rounds_x   = []
        for i in range(len(sub)):
            window = sub.iloc[max(0, i-9):i+1]
            w = (window['W/L'] == 1).sum()
            p = (window['W/L'] == 0).sum()
            n = len(window) - p
            rolling_wr.append(round(w/n*100, 1) if n > 0 else None)
            rounds_x.append(sub.iloc[i]['Round'])
        rolling_fig.add_trace(go.Scatter(
            x=rounds_x, y=rolling_wr,
            mode="lines", name=STRATEGY_LABELS.get(prio, prio),
            line=dict(color=colours.get(prio, "#adb5bd"), width=1.5),
        ))
    rolling_fig.add_hline(y=65, line_dash="dash", line_color="#dc3545",
                          annotation_text="65% min")
    rolling_fig.update_layout(
        title="Rolling last-10 win rate per strategy",
        xaxis_title="Round", yaxis_title="Win rate %",
        yaxis=dict(range=[0, 110]),
        legend=dict(orientation="h", y=-0.3),
        height=340, margin=dict(t=40, b=60, l=40, r=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )

    # ── Last refresh timestamp ─────────────────────────────────────────────────
    from datetime import datetime
    refresh_note = html.P(
        f"Last refreshed: {datetime.now().strftime('%H:%M:%S')} "
        f"· Auto-refreshes every 5 min",
        className="text-muted small text-end mb-3",
    )

    return html.Div([
        refresh_note,
        summary_row,
        html.H5("Strategy breakdown", className="mb-2"),
        strategy_table,
        dbc.Row([
            dbc.Col([
                html.H5("Win rate by round", className="mb-2"),
                dcc.Graph(figure=round_chart, config={"displayModeBar": False}),
            ], width=8),
            dbc.Col([
                html.H5("By stat type", className="mb-2"),
                type_table,
            ], width=4),
        ], className="mb-3"),
        html.H5("Rolling last-10 win rate per strategy", className="mb-2"),
        dcc.Graph(figure=rolling_fig, config={"displayModeBar": False}),
    ])


# ── Performance + Multi Builder tab callback ──────────────────────────────────
@app.callback(
    Output('performance-content',   'children'),
    Output('multi-builder-content', 'children'),
    Output('bet-dashboard-content', 'style'),
    [Input('stat-tabs',             'active_tab'),
     Input('perf-interval',         'n_intervals')],
)
def update_special_tabs(active_tab, _n_intervals):
    if active_tab == 'tab-performance':
        df      = load_performance_data()
        content = build_performance_layout(df)
        return content, html.Div(), {"display": "none"}
    elif active_tab == 'tab-multi':
        content = build_multi_builder_layout()
        return html.Div(), content, {"display": "none"}
    else:
        return html.Div(), html.Div(), {"display": "block"}


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

if __name__ == '__main__':
    app.run(debug=True)
