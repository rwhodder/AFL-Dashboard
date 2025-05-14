import pandas as pd
import numpy as np
import dash
from dash import html, dcc, dash_table, Input, Output, callback_context
import dash_bootstrap_components as dbc

# Import from existing modules
from fixture_scraper import scrape_next_round_fixture
from travel_fatigue import build_travel_log
from stadium_locations import STADIUM_COORDS
from stat_rules import apply_sensitivity
from data_processor import load_and_prepare_data, calculate_dvp

# ===== CONSTANTS =====
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

def categorize_weather(weather):
    try:
        # Handle potential None values
        rain = weather.get('rain', 0) if weather else 0
        wind = weather.get('wind', 0) if weather else 0
        humidity = weather.get('humidity', 0) if weather else 0
        
        # Ensure numeric values
        rain = float(rain) if rain is not None else 0
        wind = float(wind) if wind is not None else 0
        humidity = float(humidity) if humidity is not None else 0
        
        # Categorize
        rain_cat = "Low" if rain == 0 else "Med" if rain <= 2 else "High"
        wind_cat = "Low" if wind <= 10 else "Med" if wind <= 20 else "High"
        humid_cat = "Low" if humidity <= 60 else "Med" if humidity <= 70 else "High"  # Changed from 85 to 70
        
        # Track which flags are hit for detailed display
        flags_hit = []
        
        # Count the flags - high rain, high wind, high humidity
        flag_count = 0
        if rain > 2:
            flag_count += 1
            flags_hit.append("High Rain")
        if wind > 6:
            flag_count += 1
            flags_hit.append("High Wind")
        if humidity > 70:  # Changed from 85 to 70
            flag_count += 1
            flags_hit.append("High Humidity")
            
        # Set rating based on flag count
        if flag_count == 0:
            rating = "✅ Neutral"
        elif flag_count == 1:
            rating = f"⚠️ Medium Unders Edge ({flags_hit[0]})"
        else:
            rating = f"🔴 Strong Unders Edge ({', '.join(flags_hit)})"
            
        return {
            "Rain": rain_cat,
            "Wind": wind_cat,
            "Humidity": humid_cat,
            "Rating": rating,
            "FlagCount": flag_count,
            "Flags": flags_hit,
            "RawValues": f"Rain: {rain:.1f}mm, Wind: {wind:.1f}km/h, Humidity: {int(humidity)}%"
        }
    except Exception as e:
        print(f"Error in categorize_weather: {e}")
        return {
            "Rain": "Low",
            "Wind": "Low",
            "Humidity": "Low",
            "Rating": "✅ Neutral",
            "FlagCount": 0,
            "Flags": [],
            "RawValues": "No data available"
        }

# ===== WEIGHTED SCORE MODEL =====
def calculate_unders_score(player_row, team_weather, simplified_dvp):
    unders_score = 0
    score_factors = []
    
    # 1. TRAVEL FATIGUE (1 point per flag)
    travel_fatigue = player_row.get('travel_fatigue', '')
    if '(' in travel_fatigue:
        # Extract flags from the travel fatigue string
        flags_part = travel_fatigue.split('(')[1].split(')')[0]
        flags = [flag.strip() for flag in flags_part.split(',')]
        travel_points = len(flags)
        
        if travel_points > 0:
            score_factors.append(f"Travel: +{travel_points} ({', '.join(flags)})")
            unders_score += travel_points
    
    # 2. WEATHER (2 points for rain, 1 point each for wind/humidity)
    team = player_row.get('team', '')
    weather_points = 0
    weather_flags = []
    
    if team in team_weather:
        weather_data = team_weather[team]
        raw_values = weather_data.get('RawValues', '')
        flags = weather_data.get('Flags', [])
        
        for flag in flags:
            if 'Rain' in flag:
                weather_points += 2
                weather_flags.append('Rain(2)')
            elif 'Wind' in flag:
                weather_points += 1
                weather_flags.append('Wind(1)')
            elif 'Humidity' in flag:
                weather_points += 1
                weather_flags.append('Humidity(1)')
    
    if weather_points > 0:
        score_factors.append(f"Weather: +{weather_points} ({', '.join(weather_flags)})")
        unders_score += weather_points
    
    # 3. DVP (0 to -3 points based on strength)
    dvp_text = player_row.get('dvp', '')
    dvp_points = 0
    
    if 'Strong' in dvp_text:
        dvp_points = 3
    elif 'Moderate' in dvp_text:
        dvp_points = 2
    elif 'Slight' in dvp_text:
        dvp_points = 1
    
    if dvp_points > 0:
        # Extract the numeric value if available
        dvp_value = ""
        if '(' in dvp_text and ')' in dvp_text:
            dvp_value = dvp_text.split('(')[1].split(')')[0]
        
        score_factors.append(f"DvP: +{dvp_points} ({dvp_value})")
        unders_score += dvp_points
    
    # 4. CBA TREND (0 to 1 points)
    cba_trend = player_row.get('CBA_Trend', '')
    cba_points = 0
    
    if 'Declining' in cba_trend:
        cba_points = 1
    
    if cba_points > 0:
        score_factors.append(f"CBA Trend: +{cba_points} (Declining)")
        unders_score += cba_points
    
    # 5. TOG TREND (0 to 1 points) - Updated logic
    tog_trend = player_row.get('TOG_Trend', '')
    tog_points = 0
    
    if 'Declining' in tog_trend:
        tog_points = 1  # Changed from 2 to 1
    
    if tog_points > 0:
        factor = "Declining"
        score_factors.append(f"TOG Trend: +{tog_points} ({factor})")
        unders_score += tog_points
    
    # 6. ROLE STATUS (1 or 0 points) - Updated logic
    role_status = player_row.get('Role_Status', '')
    role_points = 0
    
    if 'LOW USAGE' in role_status:  # Target unders
        role_points = 1
        factor = "Low Usage"
        score_factors.append(f"Role: +{role_points} ({factor})")
        unders_score += role_points
    
    # No points added for STABLE or UNSTABLE (both now 0)
    
    # Create a summary of all factors
    factors_summary = " | ".join(score_factors)
    
    return {
        "UnderScore": unders_score,
        "Factors": factors_summary
    }

# Function to add unders score to the dataframe
def add_unders_score_to_dataframe(df, team_weather, simplified_dvp):
    # Create new columns for the score
    df['UnderScore'] = 0
    df['ScoreFactors'] = ""
    
    # Calculate scores for each player
    for idx, row in df.iterrows():
        score_data = calculate_unders_score(row, team_weather, simplified_dvp)
        score_value = score_data["UnderScore"]
        
        # Add a descriptive rating based on the score
        if score_value >= 8:
            rating = "Strong Play"
        elif score_value >= 6:
            rating = "Good Play"
        elif score_value >= 4:
            rating = "Consider"
        elif score_value >= 2:
            rating = "Weak"
        else:
            rating = "Avoid"
            
        # Set the UnderScore column to include both numeric score and rating
        df.at[idx, 'UnderScore'] = f"{score_value} - {rating}"
        df.at[idx, 'ScoreFactors'] = score_data["Factors"]  # Keep for possible filtering but won't display
    
    return df
def calculate_tog_cba_trends(df):
    # Ensure tog and cbas columns exist and are numeric
    required_columns = ['round', 'tog', 'cbas']
    for col in required_columns:
        if col not in df.columns:
            print(f"WARNING: Missing required column '{col}' in TOG/CBA calculation")
            # Create a minimal dataframe with required columns
            return pd.DataFrame(columns=['player', 'team', 'TOG_slope', 'CBA_slope', 
                                         'TOG_Trend', 'CBA_Trend', 'TOG_avg', 'TOG_std', 
                                         'CBA_avg', 'CBA_std', 'Role_Status'])
    
    # Convert to numeric and handle errors
    for col in ['tog', 'cbas']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Filter for last 5 rounds
    try:
        latest_round = df['round'].max()
        recent_df = df[df['round'] >= latest_round - 4]
        
        # Handle case with too few rounds
        if recent_df['round'].nunique() < 2:
            print("WARNING: Not enough unique rounds for trend calculation")
            recent_df = df.copy()  # Use all available data
    except Exception as e:
        print(f"WARNING: Error filtering rounds: {e}")
        recent_df = df.copy()  # Fallback to using all data
    
    # Calculate TOG/CBA trend slopes with error handling
    def calc_trend_slope(group):
        try:
            rounds = group['round']
            # Only calculate slope if we have enough data points
            if len(rounds) >= 2:
                tog_slope = np.polyfit(rounds, group['tog'], 1)[0]
                cba_slope = np.polyfit(rounds, group['cbas'], 1)[0]
            else:
                tog_slope = None
                cba_slope = None
                
            return pd.Series({'TOG_slope': tog_slope, 'CBA_slope': cba_slope})
        except Exception as e:
            print(f"WARNING: Error calculating slope: {e}")
            return pd.Series({'TOG_slope': None, 'CBA_slope': None})
    
    try:
        slopes = recent_df.groupby(['player', 'team']).apply(calc_trend_slope, include_groups=False).reset_index()
    except Exception as e:
        print(f"ERROR in groupby.apply: {e}")
        # Create a minimal dataframe if groupby fails
        player_teams = recent_df[['player', 'team']].drop_duplicates()
        slopes = player_teams.copy()
        slopes['TOG_slope'] = None
        slopes['CBA_slope'] = None
    
    # Convert slopes to emoji indicators
    def slope_to_icon(slope):
        if slope > 1:
            return '📈 Increasing'
        elif slope < -1:
            return '📉 Declining'
        else:
            return '⚠️ Flat'
    
    slopes['TOG_Trend'] = slopes['TOG_slope'].apply(slope_to_icon)
    slopes['CBA_Trend'] = slopes['CBA_slope'].apply(slope_to_icon)
    
    # Aggregate stability stats
    trend = recent_df.groupby(['player', 'team']).agg({
        'tog': ['mean', 'std'],
        'cbas': ['mean', 'std']
    }).reset_index()
    
    trend.columns = ['player', 'team', 'TOG_avg', 'TOG_std', 'CBA_avg', 'CBA_std']
    
    # Merge in slope trend indicators
    trend = trend.merge(slopes[['player', 'team', 'TOG_Trend', 'CBA_Trend']],
                        on=['player', 'team'], how='left')
    
    # Role stability classification
    def flag_risk(row):
        if row['CBA_std'] > 5 or row['TOG_std'] > 6:
            return "⚠️ UNSTABLE"
        elif row['CBA_avg'] < 5:
            return "📉 LOW USAGE"
        else:
            return "🎯 STABLE"
    
    trend['Role_Status'] = trend.apply(flag_risk, axis=1)
    
    return trend

# ===== DATA PROCESSING =====
def process_data_for_dashboard():
    try:
        # Add more debug output
        print("Starting process_data_for_dashboard...")
        
        # Step 1: Get next round fixtures
        try:
            fixtures = scrape_next_round_fixture()
            if not fixtures:
                print("⚠️ No fixture data found. Using test data.")
                # Create test fixture data
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
            df = df.fillna(0)  # Fill NA values with 0 for numerical calculations
            print(f"Loaded player stats with {len(df)} rows")
        except Exception as e:
            print(f"Error loading player stats: {e}. Creating test data.")
            # Create test data with minimum required columns
            df = pd.DataFrame([
                {
                    "player": "Test Player 1", 
                    "team": "Team A",
                    "opponent": "Team B", 
                    "round": 10,
                    "namedPosition": "CHF",
                    "tog": 80,
                    "cbas": 10
                },
                {
                    "player": "Test Player 2", 
                    "team": "Team B",
                    "opponent": "Team A", 
                    "round": 10,
                    "namedPosition": "RK",
                    "tog": 75,
                    "cbas": 5
                }
            ])
        
        # Step 5: Process the data for TOG/CBA trends - add explicit fallback columns
        for col in ['tog', 'cbas']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0
                print(f"Warning: '{col}' column missing, added with default values")
                
        try:
            trend_data = calculate_tog_cba_trends(df)
            print(f"Calculated TOG/CBA trends for {len(trend_data)} players")
        except Exception as e:
            print(f"Warning in TOG/CBA calculation: {e}")
            # Create dummy trend data with necessary columns
            trend_data = pd.DataFrame([
                {"player": "Test Player 1", "team": "Team A", "TOG_Trend": "⚠️ Flat", "CBA_Trend": "⚠️ Flat", "Role_Status": "🎯 STABLE"},
                {"player": "Test Player 2", "team": "Team B", "TOG_Trend": "⚠️ Flat", "CBA_Trend": "⚠️ Flat", "Role_Status": "🎯 STABLE"}
            ])
        
        # Step 6: Generate DvP data using a simplified approach
        try:
            # Instead of using the complicated DvP function, let's create a simplified version
            # directly in the main code
            simplified_dvp = {}
            
            # Get the processed data
            processed_df = load_and_prepare_data("afl_player_stats.csv")
            
            # Ensure all required columns exist
            if "disposals" not in processed_df.columns:
                if "kicks" in processed_df.columns and "handballs" in processed_df.columns:
                    processed_df["disposals"] = processed_df["kicks"] + processed_df["handballs"]
                else:
                    processed_df["disposals"] = 0
                    
            # Handle column name differences
            if "opponentTeam" not in processed_df.columns and "opponent" in processed_df.columns:
                processed_df["opponentTeam"] = processed_df["opponent"]
                
            # Calculate role averages
            role_averages = {}
            for role in processed_df['role'].unique():
                role_data = processed_df[processed_df['role'] == role]
                if not role_data.empty:
                    role_averages[role] = role_data['disposals'].mean()
            
            # Calculate team-role averages and DvP
            for team in processed_df['opponentTeam'].unique():
                simplified_dvp[team] = {}
                
                for role in processed_df['role'].unique():
                    team_role_data = processed_df[(processed_df['opponentTeam'] == team) & 
                                                (processed_df['role'] == role)]
                    
                    if not team_role_data.empty and role in role_averages:
                        team_role_avg = team_role_data['disposals'].mean()
                        dvp = team_role_avg - role_averages[role]
                        
                        # Only track significant unders
                        if dvp <= -0.1:
                            strength = "Strong" if dvp <= -2.0 else "Moderate" if dvp <= -1.0 else "Slight"
                            simplified_dvp[team][role] = {
                                "dvp": dvp,
                                "strength": strength
                            }
            
            # Use this simplified_dvp dictionary instead of the dvp_disposals DataFrame
            print(f"Created simplified DvP data with {len(simplified_dvp)} teams")
        except Exception as e:
            print(f"Warning in simplified DvP calculation: {e}")
            simplified_dvp = {}
        
        # Step 7: Extract team matchups for next round
        teams_playing = []
        team_weather = {}
        team_opponents = {}
        
        # Print the fixtures to help with debugging
        print("Next round fixtures:")
        for match in fixtures:
            match_str = match["match"]
            print(f" - {match_str}")
            try:
                home_team, away_team = match_str.split(" vs ")
                teams_playing.extend([home_team, away_team])
                
                # Map teams to weather
                weather = weather_data.get(match_str)
                if weather:
                    weather_rating = categorize_weather(weather)
                    team_weather[home_team] = weather_rating
                    team_weather[away_team] = weather_rating
                
                # Map teams to opponents
                team_opponents[home_team] = away_team
                team_opponents[away_team] = home_team
            except Exception as e:
                print(f"Error processing match {match_str}: {e}")
        
        # Print the team matchups for debugging
        print("Team opponents for next round:")
        for team, opponent in team_opponents.items():
            print(f" - {team} vs {opponent}")
        
        # Step 8: Filter for players in the upcoming round only
        try:
            # Instead of filtering for next round (which doesn't exist in the data yet)
            # Use the most recent round data and assume these players will play next round
            latest_round = df['round'].max()
            print(f"Latest round in data: {latest_round}")
            
            # Get all players from the latest round
            latest_data = df[df['round'] == latest_round].copy()
            print(f"Players in latest round: {len(latest_data)}")
            
            # Check if we have any data
            if len(latest_data) == 0:
                print("No players found in latest round. Using all players.")
                latest_data = df.copy()
                
            # Get unique player-team combinations (one row per player)
            player_teams = latest_data.groupby(['player', 'team']).first().reset_index()
            print(f"Unique player-team combinations: {len(player_teams)}")
            
            # Instead of filtering by teams_playing, keep all players since we can't predict
            # who's playing in the next round with certainty
            next_round_players = player_teams.copy()
            print(f"Using {len(next_round_players)} players for the dashboard")
        except Exception as e:
            print(f"Error filtering players: {e}. Using full dataset.")
            # Use the entire dataset, but take only the latest record for each player
            next_round_players = df.sort_values('round', ascending=False).groupby('player').first().reset_index()
        
        # Step 9: Add travel fatigue data with improved long travel detection
        travel_dict = {}
        
        print("Travel fatigue information:")
        for entry in travel_log:
            # We want entries for the NEXT round (latest_round + 1)
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
                # Make sure fatigue is a number, not None
                fatigue_value = entry.get('fatigue_score')
                if fatigue_value is None:
                    fatigue = 0
                else:
                    try:
                        fatigue = float(fatigue_value)
                    except (ValueError, TypeError):
                        fatigue = 0
                
                # Track specific travel flags
                travel_flags = []
                
                # Extract all flag information from the entry
                if entry.get('back_to_back', False):
                    travel_flags.append("Back to Back")
                if entry.get('timezone_change', False) or entry.get('time_zone_change', False):
                    travel_flags.append("Time Zone")
                if entry.get('short_break', False) or entry.get('days_break', 0) < 6:
                    travel_flags.append("Short Break")
                
                # Enhanced long travel detection - check travel_distance or directly detect known long trips
                long_travel_detected = False
                
                # Method 1: Check if travel_distance is directly available
                if entry.get('travel_distance', 0) > 1000 or entry.get('long_travel', False):
                    long_travel_detected = True
                
                # Method 2: Check for specific known long travel combinations
                if not long_travel_detected:
                    # STK to WCE case - traveling to Perth
                    if (team == "STK" and team_opponents.get(TEAM_NAME_MAP.get("STK", ""), "") == TEAM_NAME_MAP.get("WCE", "")) or \
                       (team == "WCE" and team_opponents.get(TEAM_NAME_MAP.get("WCE", ""), "") == TEAM_NAME_MAP.get("STK", "")):
                        if "Time Zone" in travel_flags:  # If time zone change is detected, it's almost certainly long travel
                            long_travel_detected = True
                    
                    # Other specific known long travel routes
                    long_distance_routes = [
                        # Victoria to WA
                        ("MEL", "WCE"), ("GEE", "WCE"), ("COL", "WCE"), ("HAW", "WCE"), 
                        ("CAR", "WCE"), ("ESS", "WCE"), ("NTH", "WCE"), ("RIC", "WCE"), 
                        ("STK", "WCE"), ("WBD", "WCE"),
                        # Victoria to FRE
                        ("MEL", "FRE"), ("GEE", "FRE"), ("COL", "FRE"), ("HAW", "FRE"), 
                        ("CAR", "FRE"), ("ESS", "FRE"), ("NTH", "FRE"), ("RIC", "FRE"), 
                        ("STK", "FRE"), ("WBD", "FRE"),
                        # East coast to west coast
                        ("SYD", "WCE"), ("SYD", "FRE"), ("GWS", "WCE"), ("GWS", "FRE"),
                        ("BRL", "WCE"), ("BRL", "FRE"), ("GCS", "WCE"), ("GCS", "FRE"),
                        # West coast to east coast
                        ("WCE", "SYD"), ("WCE", "GWS"), ("WCE", "BRL"), ("WCE", "GCS"),
                        ("FRE", "SYD"), ("FRE", "GWS"), ("FRE", "BRL"), ("FRE", "GCS"),
                        # Perth to Adelaide (smaller but still significant)
                        ("WCE", "ADE"), ("WCE", "PTA"), ("FRE", "ADE"), ("FRE", "PTA"),
                    ]
                    
                    # Check if the team and opponent combination matches any long distance route
                    opponent_abbr = ""
                    for abbr, name in TEAM_NAME_MAP.items():
                        if name == team_opponents.get(TEAM_NAME_MAP.get(team, ""), ""):
                            opponent_abbr = abbr
                            break
                    
                    if (team, opponent_abbr) in long_distance_routes:
                        long_travel_detected = True
                
                # Add long travel flag if detected by any method
                if long_travel_detected:
                    travel_flags.append("Long Travel")
                    
                # If no specific flags but fatigue score is significant, add generic flag
                if not travel_flags and fatigue >= 1:
                    travel_flags.append(f"Fatigue Score: {fatigue:.1f}")
                    
                # Now use the fatigue information for rating with improved consistency
                if fatigue < 1 and not travel_flags:
                    emoji = "✅ Low"
                elif len(travel_flags) >= 2:
                    # Any combination of 2+ flags should be High
                    emoji = f"🔴 High ({', '.join(travel_flags)})"
                elif len(travel_flags) == 1:
                    # Single flag is Medium
                    emoji = f"⚠️ Medium ({', '.join(travel_flags)})"
                elif fatigue >= 1:
                    # No flags but significant fatigue score
                    emoji = f"⚠️ Medium (Fatigue Score: {fatigue:.1f})"
                else:
                    # Fallback
                    emoji = "✅ Low"
                
                # Store the value and print debug info
                travel_dict[team] = emoji
                print(f" - {team_name} ({team}): Round {round_num}, Fatigue {fatigue} → {emoji}")
        
        # For any teams not in the travel log, use a default value
        for team_abbr in next_round_players['team'].unique():
            if team_abbr not in travel_dict:
                full_name = TEAM_NAME_MAP.get(team_abbr, team_abbr)
                if full_name in team_opponents:  # Only add if team is actually playing
                    travel_dict[team_abbr] = "⚠️ Unknown"
                    print(f" - {full_name} ({team_abbr}): No travel data found, using default")
                    
        print(f"Added travel fatigue data for {len(travel_dict)} teams")
        
        # Add travel fatigue with fallback for missing teams
        next_round_players['travel_fatigue'] = next_round_players['team'].map(
            lambda x: travel_dict.get(x, "⚠️ Unknown")
        )
        
        # Step 10: Add weather data with fallback for all teams
        # For teams not playing next round, just use neutral weather
        next_round_players['weather'] = next_round_players['team'].map(
            lambda x: team_weather.get(x, {"Rating": "✅ Neutral"}).get('Rating', "✅ Neutral")
        )
        
        # Step 11: Add opponent for DvP with fallback
        # Map team abbreviations to full names for lookup
        def get_team_full_name(team_abbr):
            return TEAM_NAME_MAP.get(team_abbr, team_abbr)
        
        def get_team_opponent(team_abbr):
            full_name = get_team_full_name(team_abbr)
            opponent_full_name = team_opponents.get(full_name, "Unknown")
            
            # Find the abbreviation for the opponent if possible
            for abbr, name in TEAM_NAME_MAP.items():
                if name == opponent_full_name:
                    return abbr
            
            return opponent_full_name  # Return full name if no abbreviation found
        
        # Apply the mapping
        next_round_players['opponent'] = next_round_players['team'].apply(get_team_opponent)
        
        # If we still have "Unknown" opponents, log them
        unknown_opponents = next_round_players[next_round_players['opponent'] == "Unknown"]['team'].unique()
        if len(unknown_opponents) > 0:
            print(f"Warning: {len(unknown_opponents)} teams have unknown opponents:")
            for team in unknown_opponents:
                print(f" - {team}: No opponent found in fixture (full name: {get_team_full_name(team)})")
        
        # Step 12: Add role positions with default for unknown
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
        
        # Step 13: Add DvP indicators with better error handling
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
        
        # Step 14: Add TOG/CBA trends - handle potential missing columns in trend_data
        if 'TOG_Trend' not in trend_data.columns or 'CBA_Trend' not in trend_data.columns or 'Role_Status' not in trend_data.columns:
            print("Missing columns in trend_data. Adding defaults.")
            for col in ['TOG_Trend', 'CBA_Trend', 'Role_Status']:
                if col not in trend_data.columns:
                    trend_data[col] = '⚠️ Unknown'
                    
        try:
            # Check if we need to convert column names to lowercase
            player_col = 'player' if 'player' in trend_data.columns else 'Player'
            team_col = 'team' if 'team' in trend_data.columns else 'Team'
            
            # For merge to work, ensure column names match in both dataframes
            trend_columns = [player_col, team_col, 'TOG_Trend', 'CBA_Trend', 'Role_Status']
            
            # Rename columns in trend_data if needed
            if player_col != 'player' or team_col != 'team':
                trend_data = trend_data.rename(columns={player_col: 'player', team_col: 'team'})
            
            # Get only the columns we need
            trend_subset = trend_data[['player', 'team', 'TOG_Trend', 'CBA_Trend', 'Role_Status']]
            
            # Now do the merge
            next_round_players = next_round_players.merge(
                trend_subset,
                on=['player', 'team'],
                how='left'
            )
            print("Added TOG/CBA trends data")
        except Exception as e:
            print(f"Error adding TOG/CBA trends: {e}")
            # Add placeholder columns
            next_round_players['TOG_Trend'] = '⚠️ Unknown'
            next_round_players['CBA_Trend'] = '⚠️ Unknown'
            next_round_players['Role_Status'] = '⚠️ Unknown'
            
        # Step 15: Fill any missing values with placeholders
        for col in ['TOG_Trend', 'CBA_Trend', 'Role_Status']:
            if col in next_round_players.columns:
                next_round_players[col] = next_round_players[col].fillna('⚠️ Unknown')
            else:
                next_round_players[col] = '⚠️ Unknown'
        
        # Step 16: Clean up and select final columns for display
        result_df = next_round_players[['player', 'team', 'opponent', 'position', 'travel_fatigue', 
                                        'weather', 'dvp', 'TOG_Trend', 'CBA_Trend', 'Role_Status']].copy()
        
        # Calculate the Unders Score for each player
        result_df = add_unders_score_to_dataframe(result_df, team_weather, simplified_dvp)
        
        # Final columns for display - include only the Under Score column, not Score Factors
        display_df = result_df[['player', 'team', 'opponent', 'position', 
                                'travel_fatigue', 'weather', 'dvp', 'TOG_Trend', 'CBA_Trend', 
                                'Role_Status', 'UnderScore']]
        
        # Rename columns for display
        display_df.columns = ['Player', 'Team', 'Opponent', 'Position', 
                            'Travel Fatigue', 'Weather', 'DvP', 'TOG Trend', 'CBA Trend', 
                            'Role Status', 'Under Score']
        
        # Create a numeric-only version of the Under Score for sorting
        display_df['SortScore'] = display_df['Under Score'].apply(lambda x: int(x.split(' - ')[0]) if isinstance(x, str) else 0)
        
        # Sort by the numeric Under Score (descending) and then by team and player
        display_df = display_df.sort_values(['SortScore', 'Team', 'Player'], ascending=[False, True, True])
        
        # Remove the sorting column as it's not needed for display
        display_df = display_df.drop(columns=['SortScore'])
        
        print(f"Final dashboard data has {len(display_df)} rows")
        return display_df
        
    except Exception as e:
        print(f"CRITICAL ERROR in process_data_for_dashboard: {e}")
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
            'TOG Trend': 'Error',
            'CBA Trend': 'Error',
            'Role Status': 'Error'
        }])

# ===== DASH APP =====
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Store the processed data in a global variable to avoid reprocessing
processed_data = None

# Define the layout
app.layout = dbc.Container([
    html.H1("AFL Player Dashboard - Next Round", className="text-center my-4"),
    
    # Hidden div to store the loaded data
    html.Div(id='loaded-data', style={'display': 'none'}),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Filter by Team:"),
                dcc.Dropdown(
                    id="team-filter",
                    placeholder="Select team(s)...",
                    clearable=True,
                    multi=True  # Enable multi-select
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
        
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Travel Fatigue", className="text-center"),
                        html.Div([
                            html.Span("✅ Low", className="badge bg-success me-2"),
                            html.Span("⚠️ Medium", className="badge bg-warning me-2"),
                            html.Span("🔴 High", className="badge bg-danger")
                        ], className="d-flex justify-content-center")
                    ], className="border rounded p-2")
                ]),
                dbc.Col([
                    html.Div([
                        html.H4("Weather", className="text-center"),
                        html.Div([
                            html.Span("✅ Neutral", className="badge bg-success me-2"),
                            html.Span("⚠️ Medium Unders Edge", className="badge bg-warning me-2"),
                            html.Span("🔴 Strong Unders Edge", className="badge bg-danger")
                        ], className="d-flex justify-content-center")
                    ], className="border rounded p-2")
                ]),
                dbc.Col([
                    html.Div([
                        html.H4("DvP", className="text-center"),
                        html.Div([
                            html.Span("✅ Neutral", className="badge bg-success me-2"),
                            html.Span("🟡 Slight Unders", className="badge bg-warning text-dark me-2"),
                            html.Span("🟠 Moderate Unders", className="badge bg-warning me-2"),
                            html.Span("🔴 Strong Unders", className="badge bg-danger")
                        ], className="d-flex justify-content-center")
                    ], className="border rounded p-2")
                ])
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("TOG & CBA Trends", className="text-center"),
                        html.Div([
                            html.Span("📈 Increasing", className="badge bg-success me-2"),
                            html.Span("⚠️ Flat", className="badge bg-warning text-dark me-2"),
                            html.Span("📉 Declining", className="badge bg-danger")
                        ], className="d-flex justify-content-center")
                    ], className="border rounded p-2")
                ]),
                dbc.Col([
                    html.Div([
                        html.H4("Role Status", className="text-center"),
                        html.Div([
                            html.Span("🎯 STABLE", className="badge bg-success me-2"),
                            html.Span("⚠️ UNSTABLE", className="badge bg-warning text-dark me-2"),
                            html.Span("📉 LOW USAGE", className="badge bg-danger")
                        ], className="d-flex justify-content-center")
                    ], className="border rounded p-2")
                ])
            ], className="mb-3")
        ], width=9)
    ], className="mb-4"),
    
    html.Hr(),
    
    html.Div(id="loading-message", children="Loading player data...", className="text-center fs-4 mb-3"),
    
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
        style_data_conditional=[
            # Travel Fatigue
            {'if': {'column_id': 'Travel Fatigue', 'filter_query': '{Travel Fatigue} contains "Low"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
            {'if': {'column_id': 'Travel Fatigue', 'filter_query': '{Travel Fatigue} contains "Medium"'},
             'backgroundColor': '#fff3cd', 'color': 'black'},
            {'if': {'column_id': 'Travel Fatigue', 'filter_query': '{Travel Fatigue} contains "High"'},
             'backgroundColor': '#f8d7da', 'color': 'black'},
            
            # Weather
            {'if': {'column_id': 'Weather', 'filter_query': '{Weather} contains "Neutral"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
            {'if': {'column_id': 'Weather', 'filter_query': '{Weather} contains "Medium"'},
             'backgroundColor': '#fff3cd', 'color': 'black'},
            {'if': {'column_id': 'Weather', 'filter_query': '{Weather} contains "Strong"'},
             'backgroundColor': '#f8d7da', 'color': 'black'},
            
            # DvP
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Neutral"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Slight"'},
             'backgroundColor': '#fff3cd', 'color': 'black'},
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Moderate"'},
             'backgroundColor': '#ffe066', 'color': 'black'},
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Strong"'},
             'backgroundColor': '#f8d7da', 'color': 'black'},
             
            # TOG Trend
            {'if': {'column_id': 'TOG Trend', 'filter_query': '{TOG Trend} contains "Increasing"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
            {'if': {'column_id': 'TOG Trend', 'filter_query': '{TOG Trend} contains "Flat"'},
             'backgroundColor': '#fff3cd', 'color': 'black'},
            {'if': {'column_id': 'TOG Trend', 'filter_query': '{TOG Trend} contains "Declining"'},
             'backgroundColor': '#f8d7da', 'color': 'black'},
             
            # CBA Trend
            {'if': {'column_id': 'CBA Trend', 'filter_query': '{CBA Trend} contains "Increasing"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
            {'if': {'column_id': 'CBA Trend', 'filter_query': '{CBA Trend} contains "Flat"'},
             'backgroundColor': '#fff3cd', 'color': 'black'},
            {'if': {'column_id': 'CBA Trend', 'filter_query': '{CBA Trend} contains "Declining"'},
             'backgroundColor': '#f8d7da', 'color': 'black'},
            
            # Role Status
            {'if': {'column_id': 'Role Status', 'filter_query': '{Role Status} contains "STABLE"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
            {'if': {'column_id': 'Role Status', 'filter_query': '{Role Status} contains "UNSTABLE"'},
             'backgroundColor': '#fff3cd', 'color': 'black'},
            {'if': {'column_id': 'Role Status', 'filter_query': '{Role Status} contains "LOW USAGE"'},
             'backgroundColor': '#f8d7da', 'color': 'black'},
             
            # Under Score - gradient coloring based on score value in combined format (e.g., "8 - Strong Play")
            {'if': {'column_id': 'Under Score', 'filter_query': '{Under Score} contains "Strong Play"'},
             'backgroundColor': '#f8d7da', 'color': 'black', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Under Score', 'filter_query': '{Under Score} contains "Good Play"'},
             'backgroundColor': '#ffecb3', 'color': 'black', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Under Score', 'filter_query': '{Under Score} contains "Consider"'},
             'backgroundColor': '#fff9c4', 'color': 'black'},
            {'if': {'column_id': 'Under Score', 'filter_query': '{Under Score} contains "Weak"'},
             'backgroundColor': '#e8f5e9', 'color': 'black'},
            {'if': {'column_id': 'Under Score', 'filter_query': '{Under Score} contains "Avoid"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
        ],
        page_size=20,
        sort_action='native',
        filter_action='native',
    ),
], fluid=True)

# Define callbacks for the dashboard
# First callback to load data once at startup
@app.callback(
    Output('loaded-data', 'children'),
    Input('loaded-data', 'children')
)
def load_data(data):
    if data is None:
        try:
            # Process data once
            df = process_data_for_dashboard()
            
            # Add debug information
            print(f"Processed data shape: {df.shape}")
            if df.empty:
                print("WARNING: Processed dataframe is empty!")
                # Create sample data for testing
                df = pd.DataFrame([{
                    'Player': 'Test Player',
                    'Team': 'Test Team',
                    'Opponent': 'Test Opponent',
                    'Position': 'KeyF',
                    'Travel Fatigue': '✅ Low',
                    'Weather': '✅ Neutral',
                    'DvP': '✅ Neutral',
                    'TOG Trend': '⚠️ Flat',
                    'CBA Trend': '⚠️ Flat',
                    'Role Status': '🎯 STABLE',
                    'Under Score': '0 - Avoid'
                }])
                print("Created test data row.")
            
            # Cache the data globally
            global processed_data
            processed_data = df
            
            # Store data has been loaded
            return "Data loaded"
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Error loading data"
    return data  # Return unchanged if data already loaded

# Second callback to filter and display data
@app.callback(
    [Output('player-table', 'data'),
     Output('player-table', 'columns'),
     Output('team-filter', 'options'),
     Output('loading-message', 'children')],
    [Input('team-filter', 'value'),
     Input('position-filter', 'value'),
     Input('clear-team-filter', 'n_clicks'),
     Input('loaded-data', 'children')]  # Ensure this runs after data is loaded
)
def update_table(team_filter, position, clear_clicks, loaded_data):
    # Check if data is loaded
    if loaded_data != "Data loaded" or processed_data is None:
        return [], [], [], "Loading data..."
    
    try:
        # Check if the clear button was clicked
        ctx = dash.callback_context
        if ctx.triggered and 'clear-team-filter' in ctx.triggered[0]['prop_id']:
            team_filter = None  # Clear the team filter
        
        # Work with a copy of the processed data
        df = processed_data.copy()
        
        # Apply filters if provided
        if team_filter and len(team_filter) > 0:
            df = df[df['Team'].isin(team_filter)]
        if position:
            df = df[df['Position'] == position]
        
        # Create team options for dropdown (from the FULL dataset, not filtered)
        team_options = [{'label': t, 'value': t} for t in sorted(processed_data['Team'].unique())]
        
        # Define columns
        columns = [{"name": i, "id": i} for i in df.columns]
        
        return df.to_dict('records'), columns, team_options, ""
    
    except Exception as e:
        print(f"Error updating table: {str(e)}")
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
            'TOG Trend': 'N/A',
            'CBA Trend': 'N/A',
            'Role Status': 'N/A',
            'Under Score': 'N/A'
        }])
        columns = [{"name": i, "id": i} for i in error_df.columns]
        return error_df.to_dict('records'), columns, [], f"Error filtering data: {str(e)}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)