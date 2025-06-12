# Score column - gradient coloring based on score value
import pandas as pd
import numpy as np
import dash
from dash import html, dcc, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Import from existing modules
from fixture_scraper import scrape_next_round_fixture
from travel_fatigue import build_travel_log
from stadium_locations import STADIUM_COORDS
from stat_rules import apply_sensitivity
from data_processor import load_and_prepare_data, calculate_dvp

# ===== CONSTANTS =====s
OPENWEATHER_API_KEY = "e76003c560c617b8ffb27f2dee7123f4"  # From main.py
POSITION_MAP = {
    "KeyF": ["FF", "CHF"],
    "GenF": ["HFFR", "HFFL", "FPL", "FPR"],
    "Ruck": ["RK"],
    "InsM": ["C", "RR", "R"],
    "Wing": ["WL", "WR"],
    "GenD": ["HBFL", "HBFR", "BPL", "BPR"],
    "KeyD": ["CHB", "FB"]
}

# Team name mapping (CSV abbreviation to full name used in fixture)
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

# ===== WEATHER FUNCTIONS =====
# Imported from main.py
import requests
from datetime import datetime, timedelta
import pytz

def get_forecast(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        return res.json().get("list", [])
    except Exception as e:
        print(f"Error fetching forecast: {e}")
        return []

def extract_weather_for_datetime(forecast_list, target_datetime):
    closest = None
    min_diff = timedelta(hours=3)
    for entry in forecast_list:
        dt_txt = entry["dt_txt"]
        dt = datetime.strptime(dt_txt, "%Y-%m-%d %H:%M:%S")
        diff = abs(dt - target_datetime.replace(tzinfo=None))
        if diff < min_diff:
            min_diff = diff
            closest = entry

    if not closest:
        return None

    rain = closest.get("rain", {}).get("3h", 0.0)
    wind = closest.get("wind", {}).get("speed", 0.0)
    humid = closest.get("main", {}).get("humidity", 0.0)
    return {"rain": rain, "wind": wind, "humidity": humid}

def categorize_weather_for_stat(weather, stat_type='disposals'):
    # Get rain and wind values (no humidity)
    rain = float(weather.get('rain', 0)) if weather else 0
    wind = float(weather.get('wind', 0)) if weather else 0
    
    # Categorize rain
    if rain < 1.2:
        rain_value = 0
    elif rain < 3.0:
        rain_value = 1
    elif rain <= 6.0:
        rain_value = 2
    else:
        rain_value = 3
        
    # Categorize wind
    if wind < 15:
        wind_value = 0
    elif wind <= 25:
        wind_value = 1
    else:
        wind_value = 2
    
    # Calculate weather severity (rain + wind, no humidity)
    severity_score = rain_value + wind_value
    
    if severity_score >= 3:
        weather_severity = 2  # Strong
    elif severity_score >= 1:
        weather_severity = 1  # Medium
    else:
        weather_severity = 0  # Neutral
    
    # Apply stat-specific scoring
    if stat_type in ['disposals', 'marks']:
        if weather_severity == 2:
            flag_count = 3.0  # Strong
            rating_text = "Strong"
        elif weather_severity == 1:
            flag_count = 2.0 if stat_type == 'marks' else 1.5  # Medium
            rating_text = "Medium"
        else:
            flag_count = 0  # Neutral
            rating_text = "Neutral"
            
    elif stat_type == 'tackles':
        # Only rain matters for tackles
        if rain_value > 0:  # Any rain ≥1.2mm
            flag_count = -999  # Avoid
            rating_text = "Avoid"
        else:
            flag_count = 0  # Neutral
            rating_text = "Neutral"
    
    # Build display with both rain and wind
    weather_factors = []
    if rain_value > 0:
        weather_factors.append("Rain")
    if wind_value > 0:
        weather_factors.append("Wind")
    
    weather_values_str = ', '.join(weather_factors) if weather_factors else "Clear conditions"
    
    # Create rating display
    if stat_type == 'tackles' and rain_value > 0:
        rating = f"🔵 AVOID RAIN ({weather_values_str})"
    elif flag_count == 0:
        rating = "✅ Neutral"
    elif rating_text == "Medium":
        rating = f"⚠️ Medium Unders Edge ({weather_values_str})"
    elif rating_text == "Strong":
        rating = f"🔴 Strong Unders Edge ({weather_values_str})"
    
    return {
        "Rating": rating,
        "FlagCount": flag_count,
        "RawValues": f"Rain: {rain:.1f}mm, Wind: {wind:.1f}km/h"
    }


# Function to calculate DvP for different stat types
def calculate_dvp_for_stat(processed_df, stat_type='disposals'):
    """Calculate DvP for a specific stat type (disposals, marks, or tackles)"""
    simplified_dvp = {}
    
    # Ensure the stat column exists
    if stat_type not in processed_df.columns:
        if stat_type == 'disposals' and 'kicks' in processed_df.columns and 'handballs' in processed_df.columns:
            processed_df[stat_type] = processed_df['kicks'] + processed_df['handballs']
        else:
            print(f"Warning: {stat_type} column not found. Using zeros.")
            processed_df[stat_type] = 0
            
    # Handle column name differences
    if "opponentTeam" not in processed_df.columns and "opponent" in processed_df.columns:
        processed_df["opponentTeam"] = processed_df["opponent"]
        
    # Calculate role averages
    role_averages = {}
    for role in processed_df['role'].unique():
        role_data = processed_df[processed_df['role'] == role]
        if not role_data.empty:
            role_averages[role] = role_data[stat_type].mean()
    
    # Calculate team-role averages and DvP
    for team in processed_df['opponentTeam'].unique():
        simplified_dvp[team] = {}
        
        for role in processed_df['role'].unique():
            team_role_data = processed_df[(processed_df['opponentTeam'] == team) & 
                                        (processed_df['role'] == role)]
            
            if not team_role_data.empty and role in role_averages:
                team_role_avg = team_role_data[stat_type].mean()
                dvp = team_role_avg - role_averages[role]
                
                # Only track significant unders
                threshold = -0.1
                # Adjust thresholds based on stat type
                if stat_type == 'marks' or stat_type == 'tackles':
                    threshold = -0.05  # Lower threshold for marks/tackles which have lower averages
                    
                if dvp <= threshold:
                    # Adjust strength thresholds based on stat type
                    if stat_type == 'disposals':
                        strength = "Strong" if dvp <= -2.0 else "Moderate" if dvp <= -1.0 else "Slight"
                    elif stat_type in ['marks', 'tackles']:
                        strength = "Strong" if dvp <= -1.0 else "Moderate" if dvp <= -0.5 else "Slight"
                    
                    simplified_dvp[team][role] = {
                        "dvp": dvp,
                        "strength": strength
                    }
    
    return simplified_dvp


def calculate_score(player_row, team_weather, simplified_dvp, stat_type='disposals'):
    score_value = 0
    score_factors = []
    
    # 1. TRAVEL FATIGUE IMPACTS (0 to 3.0 points)
    travel_fatigue = player_row.get('travel_fatigue', '')
    travel_points = 0
    travel_details = []

    if '(' in travel_fatigue:
        flags_part = travel_fatigue.split('(')[1].split(')')[0]
        flags = [flag.strip() for flag in flags_part.split(',')]
        
        # Calculate points based ONLY on flags, ignore fatigue_score
        for flag in flags:
            if "Long Travel" in flag:
                travel_points += 2.0
                travel_details.append("Long Travel: +2.0")
            elif "Short Break" in flag:
                travel_points += 1.0
                travel_details.append("Short Break: +1.0")
            else:
                # Don't add points for other flags
                travel_details.append(f"{flag}: +0.0")

        # Cap at 3.0 (Long Travel + Short Break = 2.0 + 1.0 = 3.0)
        travel_points = min(travel_points, 3.0)

        # Apply travel points if any
        if travel_points > 0:
            score_factors.append(f"Travel: +{travel_points:.1f} ({', '.join(travel_details)})")
            score_value += travel_points
    
    # 2. WEATHER IMPACTS
    team = player_row.get('team', '')
    weather_points = 0
    
    if team in team_weather:
        weather_data = team_weather[team]
        flag_count = weather_data.get('FlagCount', 0)
        
        # Use the flag_count directly (already calculated per stat type in categorize_weather_for_stat)
        weather_points = flag_count
    
    if weather_points > 0:
        score_factors.append(f"Weather: +{weather_points:.1f}")
        score_value += weather_points
    elif weather_points < 0:
        score_factors.append(f"Weather: {weather_points:.1f}")
        score_value += weather_points
    
    # 3. DVP IMPACTS (-1 to 4 points)
    dvp_text = player_row.get('dvp', '')
    dvp_points = 0
    
    if 'Strong' in dvp_text:
        dvp_points = 4
    elif 'Moderate' in dvp_text:
        dvp_points = 2
    elif 'Slight' in dvp_text:
        dvp_points = 1
    else:
        dvp_points = -1  # Penalty for Neutral
    
    # Extract the numeric value if available for display
    dvp_value = ""
    if '(' in dvp_text and ')' in dvp_text:
        dvp_value = dvp_text.split('(')[1].split(')')[0]
    
    if dvp_points > 0:
        score_factors.append(f"DvP: +{dvp_points} ({dvp_value})")
    else:
        score_factors.append(f"DvP: {dvp_points} (Neutral)")
    
    score_value += dvp_points
    
    # Create a summary of all factors
    factors_summary = " | ".join(score_factors)
    
    return {
        "ScoreValue": score_value,
        "Factors": factors_summary
    }

def calculate_bet_flag(player_row, stat_type='disposals'):
    """Calculate bet flag based on updated analysis and filtering criteria"""
    try:
        # Extract values from the row
        position = player_row.get('Position', player_row.get('position', ''))
        weather = player_row.get('Weather', player_row.get('weather', ''))
        dvp = player_row.get('DvP', player_row.get('dvp', ''))
        travel_fatigue = player_row.get('Travel Fatigue', player_row.get('travel_fatigue', ''))
        
        # Parse travel and weather conditions
        has_long_travel = 'Long Travel' in travel_fatigue
        has_short_break = 'Short Break' in travel_fatigue
        has_moderate_travel = 'Moderate' in travel_fatigue
        
        # Check rain levels
        has_any_rain = 'Rain' in weather and 'Neutral' not in weather
        has_medium_weather = 'Medium' in weather
        
        # Check DvP levels
        has_moderate_dvp = 'Moderate' in dvp
        has_strong_dvp = 'Strong' in dvp
        
        # 🚫 AUTO-SKIP RULES (CHECK FIRST)
        if has_short_break:
            return "🚫 SKIP - Short Break"
        
        if has_any_rain and stat_type == 'tackles':
            return "🚫 SKIP - Rain (Tackles)"
        
        if position == 'KeyD':
            return "🚫 SKIP - KeyD Position"
        
        if position == 'Ruck':
            return "🚫 SKIP - Ruck Position"
        
        # 🟢 AUTO-BET RULES (5 New Priority Strategies)
        # #1: KeyF + Mark + Lines >5.0
        if position == 'KeyF' and stat_type == 'marks':
            return "🟢 AUTO-BET - KeyF Mark (Priority #1)"
        
        # #2: GenF + Disposal + Medium Weather
        if position == 'GenF' and stat_type == 'disposals' and has_medium_weather:
            return "🟢 AUTO-BET - GenF Disposal Medium Weather (Priority #2)"
        
        # #3: GenD + Mark + Lines >6.0
        if position == 'GenD' and stat_type == 'marks':
            return "🟢 AUTO-BET - GenD Mark (Priority #3)"
        
        # #4: Wing + Mark
        if position == 'Wing' and stat_type == 'marks':
            return "🟢 AUTO-BET - Wing Mark (Priority #4)"
        
        # #5: GenF + Disposal + Moderate Travel
        if position == 'GenF' and stat_type == 'disposals' and has_moderate_travel:
            return "🟢 AUTO-BET - GenF Disposal Moderate Travel (Priority #5)"
        
        # 🟡 CONSIDER RULES (Former Auto-Bet Rules + Original Consider)
        # Former Auto-Bet #1: GenF + Medium Rain + Disposals
        if position == 'GenF' and has_any_rain and stat_type == 'disposals':
            return "🟡 CONSIDER - GenF Medium Rain Disposals"
        
        # Former Auto-Bet #2: Strong DvP + Tackles
        if has_strong_dvp and stat_type == 'tackles':
            return "🟡 CONSIDER - Strong DvP Tackles"
        
        # Former Auto-Bet #3: Moderate DvP + Long Travel
        if has_moderate_dvp and has_long_travel:
            return "🟡 CONSIDER - Moderate DvP Long Travel"
        
        # Original Consider: GenF + Long Travel + Disposals
        if position == 'GenF' and has_long_travel and stat_type == 'disposals':
            return "🟡 CONSIDER - GenF Long Travel Disposals"
        
        # Default skip for anything else
        return "🚫 SKIP - No Clear Edge"
        
    except Exception as e:
        print(f"ERROR in calculate_bet_flag: {e}")
        import traceback
        traceback.print_exc()
        return "❓ ERROR"

def add_score_to_dataframe(df, team_weather, simplified_dvp, stat_type='disposals'):
    """Add a score to the dataframe based on the specific stat type"""
    score_column = 'Score'
    
    # Create new columns for the score
    df[score_column] = 0
    df['ScoreFactors'] = ""
    
    # Calculate scores for each player
    for idx, row in df.iterrows():
        score_data = calculate_score(row, team_weather, simplified_dvp, stat_type)
        score_value = score_data["ScoreValue"]
        
        # Add a descriptive rating based on score thresholds
        if score_value >= 9:
            rating = "Strong Play"
        elif score_value >= 6:
            rating = "Good Play"
        elif score_value >= 3:
            rating = "Consider"
        elif score_value >= 0:
            rating = "Weak"
        else:
            rating = "Avoid"
            
        # Set the Score column to include both numeric score and rating
        df.at[idx, score_column] = f"{score_value:.1f} - {rating}"
        df.at[idx, 'ScoreFactors'] = score_data["Factors"]
    
    return df

def add_bet_flag_to_dataframe(df, stat_type='disposals'):
    """Add bet flag column to the dataframe"""
    df['Bet_Flag'] = df.apply(lambda row: calculate_bet_flag(row, stat_type), axis=1)
    return df

# Process dashboard data for a specific stat type
def process_data_for_dashboard(stat_type='disposals'):
    try:
        print(f"Starting process_data_for_dashboard for {stat_type}...")
        
        # Step 1: Get next round fixtures
        try:
            fixtures = scrape_next_round_fixture()
            if not fixtures:
                print("⚠️ No fixture data found. Using test data.")
                fixtures = [
                    {
                        "match": "Team A vs Team B",
                        "datetime": datetime.now(pytz.timezone('Australia/Melbourne')),
                        "stadium": "MCG"
                    }
                ]
        except Exception as e:
            print(f"Error fetching fixtures: {e}. Using test data.")
            fixtures = [
                {
                    "match": "Team A vs Team B",
                    "datetime": datetime.now(pytz.timezone('Australia/Melbourne')),
                    "stadium": "MCG"
                }
            ]
        
        print(f"Got {len(fixtures)} fixtures")
        
        # Step 2: Get weather data for each match
        weather_data = {}
        for match in fixtures:
            stadium = match["stadium"]
            latlon = STADIUM_COORDS.get(stadium)
            
            if not latlon:
                for key in STADIUM_COORDS:
                    if key.lower() in stadium.lower():
                        latlon = STADIUM_COORDS[key]
                        break
                if not latlon:
                    print(f"⚠️ Unknown venue: {stadium}. Using default coordinates.")
                    latlon = (-37.8199, 144.9834)  # Default to MCG
            
            try:
                forecast_list = get_forecast(*latlon)
                weather = extract_weather_for_datetime(forecast_list, match["datetime"])
                
                match_key = match["match"]
                if weather:
                    weather_data[match_key] = weather
                else:
                    print(f"⚠️ No forecast found for {match['match']} at {stadium}. Using default values.")
                    weather_data[match_key] = {"rain": 0, "wind": 0, "humidity": 50}
            except Exception as e:
                print(f"Error fetching weather: {e}. Using default values.")
                weather_data[match["match"]] = {"rain": 0, "wind": 0, "humidity": 50}
        
        print(f"Processed weather data for {len(weather_data)} matches")
        print("Weather information for each match:")
        for match_key, weather in weather_data.items():
            weather_rating = categorize_weather_for_stat(weather, stat_type)
            
            try:
                home_team, away_team = match_key.split(" vs ")
            except:
                home_team = "Unknown"
                away_team = "Unknown"
            
            rating = weather_rating.get('Rating', '✅ Neutral')
            raw_values = weather_rating.get('RawValues', 'No data')
            flag_count = weather_rating.get('FlagCount', 0)
            
            print(f" - {match_key}: {raw_values}")
            print(f"   Rating: {rating} (Impact Score: {flag_count:.1f})")
            print("   ---")

        # Step 3: Process travel fatigue data
        try:
            travel_log = build_travel_log()
            print(f"Built travel log with {len(travel_log)} entries")
        except Exception as e:
            print(f"Error building travel log: {e}. Using empty log.")
            travel_log = []
        
        # Step 4: Load player stats
        try:
            df = pd.read_csv("afl_player_stats.csv", skiprows=3)
            df = df.fillna(0)
            print(f"Loaded player stats with {len(df)} rows")
        except Exception as e:
            print(f"Error loading player stats: {e}. Creating test data.")
            df = pd.DataFrame([
                {
                    "player": "Test Player 1", 
                    "team": "Team A",
                    "opponent": "Team B", 
                    "round": 10,
                    "namedPosition": "CHF",
                    "disposals": 20,
                    "marks": 5,
                    "tackles": 4
                },
                {
                    "player": "Test Player 2", 
                    "team": "Team B",
                    "opponent": "Team A", 
                    "round": 10,
                    "namedPosition": "RK",
                    "disposals": 15,
                    "marks": 3,
                    "tackles": 6
                }
            ])
        
        # Step 5: Generate DvP data for the specific stat type
        try:
            processed_df = load_and_prepare_data("afl_player_stats.csv")
            
            if stat_type not in processed_df.columns:
                if stat_type == 'disposals' and 'kicks' in processed_df.columns and 'handballs' in processed_df.columns:
                    processed_df[stat_type] = processed_df['kicks'] + processed_df['handballs']
                else:
                    if 'disposals' in processed_df.columns:
                        if stat_type == 'marks':
                            processed_df[stat_type] = processed_df['disposals'] * 0.2
                        elif stat_type == 'tackles':
                            processed_df[stat_type] = processed_df['disposals'] * 0.15
                    else:
                        processed_df[stat_type] = 0
                    print(f"Warning: {stat_type} column created with fallback values")
            
            simplified_dvp = calculate_dvp_for_stat(processed_df, stat_type)
            print(f"Created {stat_type}-specific DvP data with {len(simplified_dvp)} teams")
        except Exception as e:
            print(f"Warning in {stat_type} DvP calculation: {e}")
            simplified_dvp = {}
        
        # Step 6: Extract team matchups for next round
        team_weather = {}
        team_opponents = {}
        
        print("Next round fixtures:")
        for match in fixtures:
            match_str = match["match"]
            print(f" - {match_str}")
            try:
                home_team, away_team = match_str.split(" vs ")
                
                weather = weather_data.get(match_str)
                if weather:
                    weather_rating = categorize_weather_for_stat(weather, stat_type)
                    team_weather[home_team] = weather_rating
                    team_weather[away_team] = weather_rating
                    
                    # Also store by abbreviation for each team
                    for abbr, name in TEAM_NAME_MAP.items():
                        if name == home_team:
                            team_weather[abbr] = weather_rating
                        if name == away_team:
                            team_weather[abbr] = weather_rating
                
                team_opponents[home_team] = away_team
                team_opponents[away_team] = home_team
            except Exception as e:
                print(f"Error processing match {match_str}: {e}")
        
        # Step 7: Filter for players - include teams with byes
        try:
            latest_round = df['round'].max()
            print(f"Latest round in data: {latest_round}")
            
            # Get players from the last 2 rounds to catch teams that had byes
            recent_rounds = [latest_round, latest_round - 1]
            recent_data = df[df['round'].isin(recent_rounds)].copy()
            print(f"Players from rounds {recent_rounds}: {len(recent_data)}")
            
            # Get the most recent game for each player (handles byes)
            player_teams = recent_data.sort_values('round', ascending=False).groupby(['player', 'team']).first().reset_index()
            print(f"Unique players after including bye teams: {len(player_teams)}")
            
            if len(player_teams) == 0:
                print("No players found in recent rounds. Using all players.")
                player_teams = df.sort_values('round', ascending=False).groupby(['player', 'team']).first().reset_index()
            
            next_round_players = player_teams.copy()
            print(f"Using {len(next_round_players)} players for the dashboard")
        except Exception as e:
            print(f"Error filtering players: {e}. Using full dataset.")
            next_round_players = df.sort_values('round', ascending=False).groupby('player').first().reset_index()
        
        # Step 8: Add travel fatigue data with UPDATED DISPLAY LOGIC
        travel_dict = {}
        
        print("Travel fatigue information:")
        for entry in travel_log:
            round_num = entry.get('round')
            team_name = entry.get('team')
            
            # Find the abbreviation for the team
            team_abbr = None
            for abbr, name in TEAM_NAME_MAP.items():
                if name == team_name:
                    team_abbr = abbr
                    break
            
            team = team_abbr if team_abbr else team_name
            
            # Process only if this is for the next round
            if team and round_num == latest_round + 1:
                fatigue_value = entry.get('fatigue_score')
                if fatigue_value is None:
                    fatigue = 0
                else:
                    try:
                        fatigue = float(fatigue_value)
                    except (ValueError, TypeError):
                        fatigue = 0
                
                travel_flags = []
                
                if entry.get('short_rest', False):
                    travel_flags.append("Short Break")
                
                # Enhanced long travel detection
                long_travel_detected = False
                
                if entry.get('distance_km', 0) > 1000:
                    long_travel_detected = True
                
                if not long_travel_detected:
                    long_distance_routes = [
                        # Victoria to WA
                        ("MEL", "WCE"), ("GEE", "WCE"), ("COL", "WCE"), ("HAW", "WCE"), 
                        ("CAR", "WCE"), ("ESS", "WCE"), ("NTH", "WCE"), ("RIC", "WCE"), 
                        ("STK", "WCE"), ("WBD", "WCE"),
                        # Victoria to FRE
                        ("MEL", "FRE"), ("GEE", "FRE"), ("COL", "FRE"), ("HAW", "FRE"), 
                        ("CAR", "FRE"), ("ESS", "FRE"), ("NTH", "FRE"), ("RIC", "FRE"), 
                        ("STK", "FRE"), ("WBD", "FRE"),
                        # WA to Victoria
                        ("WCE", "MEL"), ("WCE", "GEE"), ("WCE", "COL"), ("WCE", "HAW"), 
                        ("WCE", "CAR"), ("WCE", "ESS"), ("WCE", "NTH"), ("WCE", "RIC"), 
                        ("WCE", "STK"), ("WCE", "WBD"),
                        # FRE to Victoria
                        ("FRE", "MEL"), ("FRE", "GEE"), ("FRE", "COL"), ("FRE", "HAW"), 
                        ("FRE", "CAR"), ("FRE", "ESS"), ("FRE", "NTH"), ("FRE", "RIC"), 
                        ("FRE", "STK"), ("FRE", "WBD"),
                        # East coast to west coast
                        ("SYD", "WCE"), ("SYD", "FRE"), ("GWS", "WCE"), ("GWS", "FRE"),
                        ("BRL", "WCE"), ("BRL", "FRE"), ("GCS", "WCE"), ("GCS", "FRE"),
                        # West coast to east coast
                        ("WCE", "SYD"), ("WCE", "GWS"), ("WCE", "BRL"), ("WCE", "GCS"),
                        ("FRE", "SYD"), ("FRE", "GWS"), ("FRE", "BRL"), ("FRE", "GCS"),
                        # Perth to Adelaide
                        ("WCE", "ADE"), ("WCE", "PTA"), ("FRE", "ADE"), ("FRE", "PTA"),
                        # Adelaide to Perth 
                        ("ADE", "WCE"), ("PTA", "WCE"), ("ADE", "FRE"), ("PTA", "FRE")
                        
                    ]
                    
                    opponent_abbr = ""
                    for abbr, name in TEAM_NAME_MAP.items():
                        if name == team_opponents.get(TEAM_NAME_MAP.get(team, ""), ""):
                            opponent_abbr = abbr
                            break
                    
                    if (team, opponent_abbr) in long_distance_routes:
                        long_travel_detected = True
                
                if long_travel_detected:
                    travel_flags.append("Long Travel")
                
                # UPDATED DISPLAY LOGIC - as requested
                if not travel_flags:
                    emoji = "✅ Neutral"
                elif "Short Break" in travel_flags and "Long Travel" in travel_flags:
                    emoji = "🔴 Strong (Short Break + Long Travel)"
                elif "Long Travel" in travel_flags:
                    emoji = "🟠 Moderate (Long Travel)"
                elif "Short Break" in travel_flags:
                    emoji = "🟡 Slight (Short Break)"
                else:
                    # Fallback for other flags
                    emoji = f"⚠️ Unknown ({', '.join(travel_flags)})"
                
                travel_dict[team] = emoji
                print(f" - {team_name} ({team}): Round {round_num}, Fatigue {fatigue} → {emoji}")
        
        # For any teams not in the travel log, use a default value
        for team_abbr in next_round_players['team'].unique():
            if team_abbr not in travel_dict:
                full_name = TEAM_NAME_MAP.get(team_abbr, team_abbr)
                if full_name in team_opponents:
                    travel_dict[team_abbr] = "✅ Neutral"
                    print(f" - {full_name} ({team_abbr}): No travel data found, using default")
                    
        print(f"Added travel fatigue data for {len(travel_dict)} teams")
        
        # Add travel fatigue with fallback for missing teams
        next_round_players['travel_fatigue'] = next_round_players['team'].map(
            lambda x: travel_dict.get(x, "✅ Neutral")
        )
        
        # Step 9: Add weather data with fallback for all teams
        next_round_players['weather'] = next_round_players['team'].map(
            lambda x: team_weather.get(x, {"Rating": "✅ Neutral"}).get('Rating', "✅ Neutral")
        )

        print("\nTeam weather mapping check:")
        for team_abbr in next_round_players['team'].unique():
            team_full = TEAM_NAME_MAP.get(team_abbr, team_abbr)
            weather_rating = team_weather.get(team_abbr, {"Rating": "✅ Neutral"}).get('Rating', "✅ Neutral")
            print(f" - {team_abbr} ({team_full}): {weather_rating}")
        
        # Step 10: Add opponent for DvP with fallback
        def get_team_full_name(team_abbr):
            return TEAM_NAME_MAP.get(team_abbr, team_abbr)
        
        def get_team_opponent(team_abbr):
            full_name = get_team_full_name(team_abbr)
            opponent_full_name = team_opponents.get(full_name, "Unknown")
            
            # Find the abbreviation for the opponent if possible
            for abbr, name in TEAM_NAME_MAP.items():
                if name == opponent_full_name:
                    return abbr
            
            return opponent_full_name
        
        # Apply the mapping
        next_round_players['opponent'] = next_round_players['team'].apply(get_team_opponent)
        
        # If we still have "Unknown" opponents, log them
        unknown_opponents = next_round_players[next_round_players['opponent'] == "Unknown"]['team'].unique()
        if len(unknown_opponents) > 0:
            print(f"Warning: {len(unknown_opponents)} teams have unknown opponents:")
            for team in unknown_opponents:
                print(f" - {team}: No opponent found in fixture (full name: {get_team_full_name(team)})")
        
        # Step 11: Add role positions with default for unknown
        def map_position(pos):
            if pd.isna(pos) or pos == "":
                return "Unknown"
            for role, tags in POSITION_MAP.items():
                if pos in tags:
                    return role
            return "Unknown"
        
        if 'namedPosition' in next_round_players.columns:
            next_round_players["position"] = next_round_players["namedPosition"].apply(map_position)
        else:
            print("Warning: namedPosition column missing. Using default positions.")
            next_round_players["position"] = "Unknown"
            
        print("Added position mapping")
        
        # Step 12: Add DvP indicators with better error handling
        def get_dvp_rating(row):
            try:
                team = row['opponent'] if 'opponent' in row else "Unknown"
                pos = row['position'] if 'position' in row else "Unknown"
                
                # Handle missing or None values
                if team is None or pos is None or team == "Unknown" or pos == "Unknown":
                    return "⚠️ Unknown"
                    
                # Use the simplified_dvp dictionary instead of dataframe lookups
                if team in simplified_dvp and pos in simplified_dvp[team]:
                    dvp_info = simplified_dvp[team][pos]
                    strength = dvp_info["strength"]
                    dvp_value = dvp_info["dvp"]
                    
                    # Format the DvP value to show in the rating
                    formatted_dvp = f"{dvp_value:.1f}"
                    
                    if strength == "Strong":
                        return f"🔴 Strong Unders ({formatted_dvp})"
                    elif strength == "Moderate":
                        return f"🟠 Moderate Unders ({formatted_dvp})"
                    elif strength == "Slight":
                        return f"🟡 Slight Unders ({formatted_dvp})"
                
                return "✅ Neutral"
            except Exception as e:
                print(f"Error in DvP rating: {e}")
                return "⚠️ Unknown"
        
        next_round_players['dvp'] = next_round_players.apply(get_dvp_rating, axis=1)
        
        # Step 13: Clean up and select final columns for display (REMOVED Score column)
        result_df = next_round_players[['player', 'team', 'opponent', 'position', 'travel_fatigue', 
                                        'weather', 'dvp']].copy()
        
        # Calculate the Unders Score for each player (for bet flag calculation only)
        result_df = add_score_to_dataframe(result_df, team_weather, simplified_dvp, stat_type)
        
        # Add bet flag based on filtering criteria
        result_df = add_bet_flag_to_dataframe(result_df, stat_type)
        
        # Final columns for display (REMOVED Score column)
        display_df = result_df[['player', 'team', 'opponent', 'position', 
                                'travel_fatigue', 'weather', 'dvp', 'Bet_Flag']]
        
        # Rename columns for display (REMOVED Score column)
        column_mapping = {
            'player': 'Player', 
            'team': 'Team', 
            'opponent': 'Opponent', 
            'position': 'Position',
            'travel_fatigue': 'Travel Fatigue', 
            'weather': 'Weather', 
            'dvp': 'DvP',
            'Bet_Flag': 'Bet Flag'
        }
        display_df.columns = list(column_mapping.values())
        
        # Sort by team and player (no score-based sorting)
        display_df = display_df.sort_values(['Team', 'Player'], ascending=[True, True])
        
        print(f"Final dashboard data for {stat_type} has {len(display_df)} rows")
        return display_df
        
    except Exception as e:
        print(f"CRITICAL ERROR in process_data_for_dashboard for {stat_type}: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a minimal valid dataframe
        return pd.DataFrame([{
            'Player': f'Error: {str(e)}',
            'Team': 'Error',
            'Opponent': 'Error',
            'Position': 'Error',
            'Travel Fatigue': 'Error',
            'Weather': 'Error', 
            'DvP': 'Error',
            'Bet Flag': 'Error'
        }])

# ===== DASH APP =====
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Store separate processed data for each stat type
processed_data_by_stat = {
    'disposals': None,
    'marks': None,
    'tackles': None
}

# Define the layout with Export to CSV feature (UPDATED TRAVEL FATIGUE LEGEND)
app.layout = dbc.Container([
    html.H1("AFL Player Dashboard - Next Round", className="text-center my-4"),
    
    # Hidden div to store the loaded data
    html.Div(id='loaded-data', style={'display': 'none'}),
    
    # Add Download component for CSV export
    dcc.Download(id="download-csv"),
    
    # Add tabs for stat selection
    dbc.Tabs([
        dbc.Tab(label="Disposals", tab_id="tab-disposals", labelClassName="fw-bold"),
        dbc.Tab(label="Marks", tab_id="tab-marks", labelClassName="fw-bold"),
        dbc.Tab(label="Tackles", tab_id="tab-tackles", labelClassName="fw-bold"),
    ], id="stat-tabs", active_tab="tab-disposals", className="mb-3"),
    
    # Rest of the layout
    dbc.Row([
        # Left column - filters
        dbc.Col([
            html.Div([
                html.H5("Filter by Team:"),
                dcc.Dropdown(
                    id="team-filter",
                    placeholder="Select team(s)...",
                    clearable=True,
                    multi=True
                ),
                html.Button(
                    "Clear Team Filter", 
                    id="clear-team-filter", 
                    className="btn btn-outline-secondary btn-sm mt-1"
                )
            ], className="mb-4"),
            
            html.Div([
                html.H5("Filter by Position:"),
                dcc.Dropdown(
                    id="position-filter",
                    options=[
                        {"label": "Key Forward", "value": "KeyF"},
                        {"label": "General Forward", "value": "GenF"},
                        {"label": "Ruck", "value": "Ruck"},
                        {"label": "Inside Mid", "value": "InsM"},
                        {"label": "Wing", "value": "Wing"},
                        {"label": "General Defender", "value": "GenD"},
                        {"label": "Key Defender", "value": "KeyD"}
                    ],
                    placeholder="Select a position...",
                    clearable=True
                )
            ], className="mb-4")
        ], width=3),
        
        # Legend cards - simplified layout
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Travel Fatigue", className="text-center"),
                        html.Div([
                            html.Span("✅ Neutral", className="badge bg-success me-2"),
                            html.Span("🟡 Slight", className="badge bg-warning text-dark me-2"),
                            html.Span("🟠 Moderate", className="badge bg-warning me-2"),
                            html.Span("🔴 Strong", className="badge bg-danger")
                        ], className="d-flex justify-content-center")
                    ], className="border rounded p-2 mb-3", 
                    id="travel-fatigue-legend",
                    title="Travel fatigue impacts: ✅ Neutral (no flags), 🟡 Slight (Short Break), 🟠 Moderate (Long Travel), 🔴 Strong (Short Break + Long Travel)")
                ], width=6),
                
                dbc.Col([
                    html.Div([
                        html.H4("Weather", className="text-center"),
                        html.Div([
                            html.Span("✅ Neutral", className="badge bg-success me-2"),
                            html.Span("⚠️ Medium Unders Edge", className="badge bg-warning me-2"),
                            html.Span("🔴 Strong Unders Edge", className="badge bg-danger me-2"),
                            html.Span("🔵 Avoid", className="badge bg-primary")
                        ], className="d-flex justify-content-center")
                    ], className="border rounded p-2 mb-3",
                    id="weather-legend",
                    title="Weather impacts: Rain + Wind affect disposals/marks negatively, but rain increases tackles (avoid tackles unders in rain).")
                ], width=6),
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("DvP", className="text-center"),
                        html.Div([
                            html.Span("✅ Neutral", className="badge bg-success me-2"),
                            html.Span("🟡 Slight Unders", className="badge bg-warning text-dark me-2"),
                            html.Span("🟠 Moderate Unders", className="badge bg-warning me-2"),
                            html.Span("🔴 Strong Unders", className="badge bg-danger")
                        ], className="d-flex justify-content-center flex-wrap")
                    ], className="border rounded p-2 mb-3",
                    id="dvp-legend",
                    title="Defenders vs Position shows historical matchup difficulty; +1 for Slight, +2 for Moderate, +4 for Strong unders, -1 for Neutral")
                ], width=6),
                
                dbc.Col([
                    html.Div([
                        html.H4("Bet Flag", className="text-center"),
                        html.Div([
                            html.Span("🟢 AUTO-BET", className="badge bg-success me-2"),
                            html.Span("🟡 CONSIDER", className="badge bg-warning text-dark me-2"),
                            html.Span("🚫 SKIP", className="badge bg-danger")
                        ], className="d-flex justify-content-center flex-wrap")
                    ], className="border rounded p-2 mb-3",
                    id="bet-flag-legend",
                    title="Updated bet logic: 🟢 AUTO-BET for KeyF+Marks, GenF+Disposal+Medium Weather, GenD+Marks, Wing+Marks, GenF+Disposal+Moderate Travel. 🟡 CONSIDER for former auto-bets (GenF+Rain+Disposals, Strong DvP+Tackles, Moderate DvP+Long Travel, GenF+Long Travel+Disposals). 🚫 SKIP for Short Break, Rain+Tackles, KeyD/Ruck positions.")
                ], width=6),
            ])
        ], width=9)
    ], className="mb-4"),
    
    html.Hr(),
    
    # Export to CSV button row
    dbc.Row([
        dbc.Col([
            html.Div(id="loading-message", children="Loading player data...", className="text-center fs-4")
        ], width=8),
        dbc.Col([
            html.Button(
                "Export to CSV",
                id="export-button",
                className="btn btn-success float-end"
            )
        ], width=4)
    ], className="mb-3"),
    
    dash_table.DataTable(
        id='player-table',
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '8px',
            'fontFamily': 'Arial'
        },
        style_header={
            'backgroundColor': '#343a40',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data_conditional=[],  # Will be updated dynamically
        page_size=20,
        sort_action='native',
        filter_action='native',
    ),
], fluid=True)

# Callback to load data for all stat types
@app.callback(
    Output('loaded-data', 'children'),
    Input('loaded-data', 'children')
)
def load_data(data):
    if data is None:
        try:
            # Process data for each stat type
            for stat_type in ['disposals', 'marks', 'tackles']:
                df = process_data_for_dashboard(stat_type)
                
                print(f"Processed data for {stat_type} shape: {df.shape}")
                if df.empty:
                    print(f"WARNING: Processed {stat_type} dataframe is empty!")
                    df = pd.DataFrame([{
                        'Player': 'Test Player',
                        'Team': 'Test Team',
                        'Opponent': 'Test Opponent',
                        'Position': 'KeyF',
                        'Travel Fatigue': '✅ Neutral',
                        'Weather': '✅ Neutral',
                        'DvP': '✅ Neutral',
                        'Bet Flag': '🚫 SKIP - Low Score'
                    }])
                    print(f"Created test data row for {stat_type}.")
                
                # Cache the data globally
                processed_data_by_stat[stat_type] = df
            
            return "Data loaded"
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Error loading data"
    return data

# Callback to filter and display data based on stat type
@app.callback(
    [Output('player-table', 'data'),
     Output('player-table', 'columns'),
     Output('player-table', 'style_data_conditional'),
     Output('team-filter', 'options'),
     Output('loading-message', 'children')],
    [Input('stat-tabs', 'active_tab'),
     Input('team-filter', 'value'),
     Input('position-filter', 'value'),
     Input('clear-team-filter', 'n_clicks'),
     Input('loaded-data', 'children')]
)
def update_table(active_tab, team_filter, position, clear_clicks, loaded_data):
    # Check if data is loaded
    if loaded_data != "Data loaded":
        return [], [], [], [], "Loading data..."
    
    # Determine which stat type is selected
    stat_type = 'disposals'
    if active_tab == "tab-marks":
        stat_type = 'marks'
    elif active_tab == "tab-tackles":
        stat_type = 'tackles'
    
    # Get corresponding data
    processed_data = processed_data_by_stat.get(stat_type)
    if processed_data is None:
        return [], [], [], [], f"No data available for {stat_type}"
    
    try:
        # Check if the clear button was clicked
        ctx = dash.callback_context
        if ctx.triggered and 'clear-team-filter' in ctx.triggered[0]['prop_id']:
            team_filter = None
        
        # Work with a copy of the processed data
        df = processed_data.copy()
        
        # Apply filters if provided
        if team_filter and len(team_filter) > 0:
            df = df[df['Team'].isin(team_filter)]
        if position:
            df = df[df['Position'] == position]
        
        # Create team options for dropdown
        team_options = [{'label': t, 'value': t} for t in sorted(processed_data['Team'].unique())]
        
        # Define columns
        columns = [{"name": i, "id": i} for i in df.columns]
        
        # Create the conditional styling (UPDATED TRAVEL FATIGUE STYLING)
        style_data_conditional = [
            # Travel Fatigue - UPDATED
            {'if': {'column_id': 'Travel Fatigue', 'filter_query': '{Travel Fatigue} contains "Neutral"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
            {'if': {'column_id': 'Travel Fatigue', 'filter_query': '{Travel Fatigue} contains "Slight"'},
             'backgroundColor': '#fff9c4', 'color': 'black'},
            {'if': {'column_id': 'Travel Fatigue', 'filter_query': '{Travel Fatigue} contains "Moderate"'},
             'backgroundColor': '#ffecb3', 'color': 'black'},
            {'if': {'column_id': 'Travel Fatigue', 'filter_query': '{Travel Fatigue} contains "Strong"'},
             'backgroundColor': '#f8d7da', 'color': 'black'},
            
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
             
            # Bet Flag colors
            {'if': {'column_id': 'Bet Flag', 'filter_query': '{Bet Flag} contains "AUTO-BET"'},
             'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Flag', 'filter_query': '{Bet Flag} contains "CONSIDER"'},
             'backgroundColor': '#ffc107', 'color': 'black', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Flag', 'filter_query': '{Bet Flag} contains "SKIP"'},
             'backgroundColor': '#f8d7da', 'color': 'black'},
        ]
        
        return df.to_dict('records'), columns, style_data_conditional, team_options, ""
    
    except Exception as e:
        print(f"Error updating table for {stat_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return error data
        error_df = pd.DataFrame([{
            'Player': f'Error: {str(e)}',
            'Team': 'N/A',
            'Opponent': 'N/A',
            'Position': 'N/A',
            'Travel Fatigue': 'N/A',
            'Weather': 'N/A', 
            'DvP': 'N/A',
            'Bet Flag': 'N/A'
        }])
        columns = [{"name": i, "id": i} for i in error_df.columns]
        return error_df.to_dict('records'), columns, [], [], f"Error filtering {stat_type} data: {str(e)}"

# Export data callback
@app.callback(
    Output("download-csv", "data"),
    Input("export-button", "n_clicks"),
    [State("stat-tabs", "active_tab")],
    prevent_initial_call=True
)
def export_data(n_clicks, active_tab):
    if n_clicks is None:
        return dash.no_update
    
    # Get the current stat type
    stat_type = 'disposals'
    if active_tab == "tab-marks":
        stat_type = 'marks'
    elif active_tab == "tab-tackles":
        stat_type = 'tackles'
    
    # Get the full dataset for this stat type
    full_data = processed_data_by_stat.get(stat_type)
    
    # If we have data, export it
    if full_data is not None and not full_data.empty:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"afl_{stat_type}_dashboard_{current_time}.csv"
        
        return dcc.send_data_frame(full_data.to_csv, filename, index=False)
    
    return dash.no_update

# Run the app
if __name__ == '__main__':
    app.run(debug=True)