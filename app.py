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
    """Categorize weather considering its effect on different stats (disposals, marks, tackles)"""
    try:
        # Handle potential None values
        rain = weather.get('rain', 0) if weather else 0
        wind = weather.get('wind', 0) if weather else 0
        humidity = weather.get('humidity', 0) if weather else 0
        
        # Ensure numeric values
        rain = float(rain) if rain is not None else 0
        wind = float(wind) if wind is not None else 0
        humidity = float(humidity) if humidity is not None else 0
        
        # RAIN CATEGORIZATION LOGIC
        if rain < 0.3:
            rain_value = 0
            rain_cat = "Low"
        elif rain < 1:
            rain_value = 1
            rain_cat = "Light"
        elif rain <= 3:
            rain_value = 2
            rain_cat = "Moderate"
        else:
            rain_value = 3
            rain_cat = "Heavy"
        
        # WIND CATEGORIZATION
        if wind < 15:
            wind_value = 0
            wind_cat = "Low"
        elif wind <= 25:
            wind_value = 1
            wind_cat = "Moderate"
        else:
            wind_value = 2
            wind_cat = "High"
        
        # HUMIDITY CATEGORIZATION
        if humidity <= 65:
            humid_value = 0
            humid_cat = "Low"
        elif humidity <= 75:
            humid_value = 1
            humid_cat = "Moderate"
        else:
            humid_value = 2
            humid_cat = "High"
        
        # Track which flags are hit for detailed display
        flags_hit = []
        
        # NEW WEATHER SCORING LOGIC
        flag_count = 0
        
        # Determine overall weather severity (0 = Neutral, 1 = Medium, 2 = Strong)
        weather_severity = 0
        
        # Calculate severity based on combined weather factors
        severity_score = rain_value + wind_value + humid_value
        
        if severity_score >= 4:  # Strong conditions
            weather_severity = 2
        elif severity_score >= 2:  # Medium conditions  
            weather_severity = 1
        else:  # Neutral conditions
            weather_severity = 0
        
        # Apply stat-specific scoring - REBALANCED TO 3 POINTS MAX
        if stat_type == 'marks':
            if weather_severity == 2:
                flag_count = 3  # Strong
                rating_text = "Strong"
            elif weather_severity == 1:
                flag_count = 2  # Medium
                rating_text = "Medium"
            else:
                flag_count = 0  # Neutral
                rating_text = "Neutral"
        elif stat_type == 'disposals':
            if weather_severity == 2:
                flag_count = 3  # Strong
                rating_text = "Strong"
            elif weather_severity == 1:
                flag_count = 1.5  # Medium
                rating_text = "Medium"
            else:
                flag_count = 0  # Neutral
                rating_text = "Neutral"
        elif stat_type == 'tackles':
            if weather_severity == 2:
                flag_count = -3  # Strong (negative for tackles)
                rating_text = "Strong"
            elif weather_severity == 1:
                flag_count = -1.5  # Medium (negative for tackles)
                rating_text = "Medium"
            else:
                flag_count = 0  # Neutral
                rating_text = "Neutral"
        
        # Build flags description
        weather_factors = []
        if rain_value > 0:
            weather_factors.append(f"Rain: {rain:.1f}mm")
        if wind_value > 0:
            weather_factors.append(f"Wind: {wind:.1f}km/h")
        if humid_value > 0:
            weather_factors.append(f"Humidity: {int(humidity)}%")
        
        weather_values_str = ', '.join(weather_factors) if weather_factors else "Clear conditions"
        
        # Create rating display
        if stat_type == 'tackles' and flag_count < 0:
            rating = f"🔵 {rating_text} Avoid ({weather_values_str})"
        elif flag_count == 0:
            rating = "✅ Neutral"
        elif rating_text == "Medium":
            rating = f"⚠️ Medium Unders Edge ({weather_values_str})"
        elif rating_text == "Strong":
            rating = f"🔴 Strong Unders Edge ({weather_values_str})"
        else:
            rating = "✅ Neutral"
            
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
    
    # 1. TRAVEL FATIGUE IMPACTS (0 to 2.5 points - UPDATED CAP)
    travel_fatigue = player_row.get('travel_fatigue', '')
    if '(' in travel_fatigue:
        flags_part = travel_fatigue.split('(')[1].split(')')[0]
        flags = [flag.strip() for flag in flags_part.split(',')]
        
        travel_points = 0
        travel_details = []
        
        # Assign points based on specific travel flags
        for flag in flags:
            if "Long Travel" in flag:
                travel_points += 2.0
                travel_details.append("Long Travel: +2.0")
            elif "Short Break" in flag:
                travel_points += 1.0
                travel_details.append("Short Break: +1.0")
            # REMOVED: Time Zone scoring (was +0.5)
            # elif "Time Zone" in flag:
            #     travel_points += 0.5
            #     travel_details.append("Time Zone: +0.5")
            else:
                # Don't add points for other flags (including Time Zone)
                travel_details.append(f"{flag}: +0.0")
        
        # Apply travel points cap at 2.5 instead of 3.5
        travel_points = min(travel_points, 2.5)
        
        # Apply travel points if any
        if travel_points > 0:
            score_factors.append(f"Travel: +{travel_points:.1f} ({', '.join(travel_details)})")
            score_value += travel_points
    
    # 2. WEATHER IMPACTS - NEW SIMPLIFIED LOGIC
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
        dvp_points = 4  # Increased from 3 to 4
    elif 'Moderate' in dvp_text:
        dvp_points = 2  # Stays at 2
    elif 'Slight' in dvp_text:
        dvp_points = 1  # Stays at 1
    else:
        dvp_points = -1  # New penalty for Neutral
    
    # Extract the numeric value if available for display
    dvp_value = ""
    if '(' in dvp_text and ')' in dvp_text:
        dvp_value = dvp_text.split('(')[1].split(')')[0]
    
    if dvp_points > 0:
        score_factors.append(f"DvP: +{dvp_points} ({dvp_value})")
    else:
        score_factors.append(f"DvP: {dvp_points} (Neutral)")
    
    score_value += dvp_points
    
    # 4. TOG TREND - REMOVED FROM SCORING (keep for display only)
    # tog_trend = player_row.get('TOG_Trend', '')
    # No scoring logic here anymore
    
    # 5. CBA TREND - REMOVED FROM SCORING (keep for display only)  
    # cba_trend = player_row.get('CBA_Trend', '')
    # No scoring logic here anymore
    
    # 6. ROLE STATUS (-1 to 2 points)
    role_status = player_row.get('Role_Status', '')
    role_points = 0
    
    if 'UNSTABLE' in role_status:
        role_points = 2  # Major change from -1 to 2
        factor = "Unstable"
    elif 'LOW USAGE' in role_status:
        role_points = 1  # Stays at 1
        factor = "Low Usage"
    elif 'STABLE' in role_status:
        role_points = -1  # New penalty from 0 to -1
        factor = "Stable"
    
    if role_points > 0:
        score_factors.append(f"Role: +{role_points} ({factor})")
    elif role_points < 0:
        score_factors.append(f"Role: {role_points} ({factor})")
    
    score_value += role_points
    
    # 7. BET TYPE FACTOR (0 to 2 points) - NEW
    bet_type_points = 0
    
    if stat_type == 'tackles':
        bet_type_points = 2
    elif stat_type == 'marks':
        bet_type_points = 1
    # Disposals: 0 points (no adjustment)
    
    if bet_type_points > 0:
        score_factors.append(f"Bet Type: +{bet_type_points} ({stat_type.capitalize()})")
        score_value += bet_type_points
    
    # Create a summary of all factors
    factors_summary = " | ".join(score_factors)
    
    return {
        "ScoreValue": score_value,
        "Factors": factors_summary
    }

def add_score_to_dataframe(df, team_weather, simplified_dvp, stat_type='disposals'):
    """Add a score to the dataframe based on the specific stat type"""
    # Always calculating unders for all stat types now
    score_column = 'Score'
    
    # Create new columns for the score
    df[score_column] = 0
    df['ScoreFactors'] = ""
    
    # Calculate scores for each player
    for idx, row in df.iterrows():
        score_data = calculate_score(row, team_weather, simplified_dvp, stat_type)
        score_value = score_data["ScoreValue"]
        
        # Add a descriptive rating based on the UPDATED score thresholds
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
        df.at[idx, 'ScoreFactors'] = score_data["Factors"]  # Keep for possible filtering but won't display
    
    return df

# Calculate TOG/CBA trends
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
        if slope is None:
            return '⚠️ Unknown'
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

# Process dashboard data for a specific stat type
def process_data_for_dashboard(stat_type='disposals'):
    try:
        print(f"Starting process_data_for_dashboard for {stat_type}...")
        
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
        print("Weather information for each match:")
        for match_key, weather in weather_data.items():
            # Get the weather categorization for this match
            weather_rating = categorize_weather_for_stat(weather, stat_type)
            
            # Extract teams from the match key
            try:
                home_team, away_team = match_key.split(" vs ")
            except:
                home_team = "Unknown"
                away_team = "Unknown"
            
            # Print detailed weather info
            rating = weather_rating.get('Rating', '✅ Neutral')
            raw_values = weather_rating.get('RawValues', 'No data')
            flags = weather_rating.get('Flags', [])
            flag_count = weather_rating.get('FlagCount', 0)
            
            # Format the flags for display
            flags_text = ", ".join(flags) if flags else "None"
            
            # Print the formatted weather information
            print(f" - {match_key}: {raw_values}")
            print(f"   Rating: {rating} (Impact Score: {flag_count:.1f})")
            if flags:
                print(f"   Flags: {flags_text}")
            
            # Add a separator between matches for readability
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
                    "cbas": 10,
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
                    "tog": 75,
                    "cbas": 5,
                    "disposals": 15,
                    "marks": 3,
                    "tackles": 6
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
        
        # Step 6: Generate DvP data using for the specific stat type
        try:
            # Get the processed data
            processed_df = load_and_prepare_data("afl_player_stats.csv")
            
            # Ensure we have the stat columns
            if stat_type not in processed_df.columns:
                if stat_type == 'disposals' and 'kicks' in processed_df.columns and 'handballs' in processed_df.columns:
                    processed_df[stat_type] = processed_df['kicks'] + processed_df['handballs']
                else:
                    # For testing, add a dummy column if needed
                    if 'disposals' in processed_df.columns:
                        # Create placeholder values based on disposals
                        if stat_type == 'marks':
                            processed_df[stat_type] = processed_df['disposals'] * 0.2  # Rough estimate
                        elif stat_type == 'tackles':
                            processed_df[stat_type] = processed_df['disposals'] * 0.15  # Rough estimate
                    else:
                        processed_df[stat_type] = 0
                    print(f"Warning: {stat_type} column created with fallback values")
            
            # Calculate DvP for the specific stat type
            simplified_dvp = calculate_dvp_for_stat(processed_df, stat_type)
            print(f"Created {stat_type}-specific DvP data with {len(simplified_dvp)} teams")
        except Exception as e:
            print(f"Warning in {stat_type} DvP calculation: {e}")
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
                
# Map teams to weather with stat-specific categorization
                weather = weather_data.get(match_str)
                if weather:
                    weather_rating = categorize_weather_for_stat(weather, stat_type)
                    # Store by full name
                    team_weather[home_team] = weather_rating
                    team_weather[away_team] = weather_rating
                    
                    # Also store by abbreviation for each team
                    for abbr, name in TEAM_NAME_MAP.items():
                        if name == home_team:
                            team_weather[abbr] = weather_rating
                        if name == away_team:
                            team_weather[abbr] = weather_rating
                
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
                # REMOVED: Time Zone flag from scoring (still appears in display)
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

# Debug: Print the team name mappings for weather data
        print("\nTeam weather mapping check:")
        for team_abbr in next_round_players['team'].unique():
            # Use TEAM_NAME_MAP directly instead of calling get_team_full_name
            team_full = TEAM_NAME_MAP.get(team_abbr, team_abbr)
            weather_rating = team_weather.get(team_abbr, {"Rating": "✅ Neutral"}).get('Rating', "✅ Neutral")
            print(f" - {team_abbr} ({team_full}): {weather_rating}")
        
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
        result_df = add_score_to_dataframe(result_df, team_weather, simplified_dvp, stat_type)
        
        # Final columns for display - include only the Score column, not Score Factors
        display_df = result_df[['player', 'team', 'opponent', 'position', 
                                'travel_fatigue', 'weather', 'dvp', 'TOG_Trend', 'CBA_Trend', 
                                'Role_Status', 'Score']]
        
        # Rename columns for display
        score_column_name = f"{stat_type.capitalize()} Score"
        column_mapping = {
            'player': 'Player', 
            'team': 'Team', 
            'opponent': 'Opponent', 
            'position': 'Position',
            'travel_fatigue': 'Travel Fatigue', 
            'weather': 'Weather', 
            'dvp': 'DvP', 
            'TOG_Trend': 'TOG Trend', 
            'CBA_Trend': 'CBA Trend',
            'Role_Status': 'Role Status', 
            'Score': score_column_name
        }
        display_df.columns = list(column_mapping.values())
        
        # Create a numeric-only version of the Score for sorting
        display_df['SortScore'] = display_df[score_column_name].apply(lambda x: float(x.split(' - ')[0]) if isinstance(x, str) else 0)
        
        # Sort by the numeric Score (descending) and then by team and player
        display_df = display_df.sort_values(['SortScore', 'Team', 'Player'], ascending=[False, True, True])
        
        # Remove the sorting column as it's not needed for display
        display_df = display_df.drop(columns=['SortScore'])
        
        print(f"Final dashboard data for {stat_type} has {len(display_df)} rows")
        return display_df
        
    except Exception as e:
        print(f"CRITICAL ERROR in process_data_for_dashboard for {stat_type}: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a minimal valid dataframe
        score_column_name = f"{stat_type.capitalize()} Score"
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
            'Role Status': 'Error',
            score_column_name: 'Error'
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

# Define the layout with Export to CSV feature
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
        
        # First column of legend cards
        dbc.Col([
            # First column - Travel Fatigue, Weather, DvP
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Travel Fatigue", className="text-center"),
                        html.Div([
                            html.Span("✅ Low", className="badge bg-success me-2"),
                            html.Span("⚠️ Medium", className="badge bg-warning me-2"),
                            html.Span("🔴 High", className="badge bg-danger")
                        ], className="d-flex justify-content-center")
                    ], className="border rounded p-2 mb-3", 
                    id="travel-fatigue-legend",
                    title="Travel fatigue impacts player performance; +2 for Long Travel, +1 for Short Break (Time Zone removed from scoring but still displayed). Capped at +2.5 total.")
                ], width=12),
                
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
title="Weather scoring: Marks +6/+4/0, Disposals +4/+2/0, Tackles -4/-2/0 (for Strong/Medium/Neutral). Tackles show negative scores as weather increases tackle counts.")
                ], width=12),
                
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
                ], width=12)
            ])
        ], width=3),
        
        # Second column of legend cards
        dbc.Col([
            # Second column - TOG/CBA, Role Status, Edge Score
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("TOG & CBA Trends", className="text-center"),
                        html.Div([
                            html.Span("📈 Increasing", className="badge bg-success me-2"),
                            html.Span("⚠️ Flat", className="badge bg-warning text-dark me-2"),
                            html.Span("📉 Declining", className="badge bg-danger")
                        ], className="d-flex justify-content-center")
                    ], className="border rounded p-2 mb-3",
                    id="tog-cba-legend",
                    title="Time on Ground and Center Bounce Attendance trends - displayed for information only, no longer contribute to scoring")
                ], width=12),
                
                dbc.Col([
                    html.Div([
                        html.H4("Role Status", className="text-center"),
                        html.Div([
                            html.Span("🎯 STABLE", className="badge bg-success me-2"),
                            html.Span("⚠️ UNSTABLE", className="badge bg-warning text-dark me-2"),
                            html.Span("📉 LOW USAGE", className="badge bg-danger")
                        ], className="d-flex justify-content-center")
                    ], className="border rounded p-2 mb-3",
                    id="role-legend",
                    title="Player's role consistency and usage rate; +2 for UNSTABLE, +1 for LOW USAGE, -1 for STABLE")
                ], width=12),
                
                dbc.Col([
                    # This will be dynamically updated based on selected stat
                    html.Div([
    html.H4(id="score-legend-title", className="text-center"),
    html.Div([
        html.Span("9.0+ - Strong Play", className="badge bg-danger me-2"),
        html.Span("6.0-8.9 - Good Play", className="badge bg-warning me-2"),
        html.Span("3.0-5.9 - Consider", className="badge bg-warning text-dark me-2"),
        html.Span("0.0-2.9 - Weak", className="badge bg-success text-dark me-2"),
        html.Span("Below 0 - Avoid", className="badge bg-success me-2")
    ], className="d-flex justify-content-center flex-wrap gap-1")
], className="border rounded p-2 mb-3",
id="score-legend",
title="Combined score from all factors; higher scores indicate stronger unders play. Updated thresholds: Strong Play 9+, Good Play 6-8.9, Consider 3-5.9, Weak 0-2.9, Avoid <0.")
                ], width=12)
            ])
        ], width=3),
        
        # Add a smaller spacer column to maintain layout
        dbc.Col([], width=3)
    ], className="mb-4"),
    
    html.Hr(),
    
    # Add Export to CSV button row
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
                
                # Add debug information
                print(f"Processed data for {stat_type} shape: {df.shape}")
                if df.empty:
                    print(f"WARNING: Processed {stat_type} dataframe is empty!")
                    # Create sample data for testing
                    score_column = f"{stat_type.capitalize()} Score"
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
                        score_column: '0 - Avoid'
                    }])
                    print(f"Created test data row for {stat_type}.")
                
                # Cache the data globally
                processed_data_by_stat[stat_type] = df
            
            # Store that data has been loaded
            return "Data loaded"
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Error loading data"
    return data  # Return unchanged if data already loaded

# Callback to update the score legend title when tab changes
@app.callback(
    Output('score-legend-title', 'children'),
    Input('stat-tabs', 'active_tab')
)
def update_score_legend_title(active_tab):
    if active_tab == "tab-disposals":
        return "Disposals Unders Score"
    elif active_tab == "tab-marks":
        return "Marks Unders Score"
    elif active_tab == "tab-tackles":
        return "Tackles Unders Score"
    return "Unders Score"

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
     Input('loaded-data', 'children')]  # Ensure this runs after data is loaded
)
def update_table(active_tab, team_filter, position, clear_clicks, loaded_data):
    # Check if data is loaded
    if loaded_data != "Data loaded":
        return [], [], [], [], "Loading data..."
    
    # Determine which stat type is selected
    stat_type = 'disposals'  # default
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
        
        # Create the conditional styling based on stat type
        score_column = f"{stat_type.capitalize()} Score"
        style_data_conditional = [
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
             # Add this new condition for blue dot Avoid
            {'if': {'column_id': 'Weather', 'filter_query': '{Weather} contains "🔵 Avoid"'},
             'backgroundColor': '#cce5ff', 'color': 'black'},  # Light blue for Avoid

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
             
            # Score column - gradient coloring based on score value
{'if': {'column_id': score_column, 'filter_query': '{' + score_column + '} contains "Strong Play"'},
 'backgroundColor': '#f8d7da', 'color': 'black', 'fontWeight': 'bold'},
{'if': {'column_id': score_column, 'filter_query': '{' + score_column + '} contains "Good Play"'},
 'backgroundColor': '#ffecb3', 'color': 'black', 'fontWeight': 'bold'},
{'if': {'column_id': score_column, 'filter_query': '{' + score_column + '} contains "Consider"'},
 'backgroundColor': '#fff9c4', 'color': 'black'},
{'if': {'column_id': score_column, 'filter_query': '{' + score_column + '} contains "Weak"'},
 'backgroundColor': '#e8f5e9', 'color': 'black'},
{'if': {'column_id': score_column, 'filter_query': '{' + score_column + '} contains "Avoid"'},
 'backgroundColor': '#d4edda', 'color': 'black'},
        ]
        
        return df.to_dict('records'), columns, style_data_conditional, team_options, ""
    
    except Exception as e:
        print(f"Error updating table for {stat_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return error data
        score_column = f"{stat_type.capitalize()} Score"
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
            score_column: 'N/A'
        }])
        columns = [{"name": i, "id": i} for i in error_df.columns]
        return error_df.to_dict('records'), columns, [], [], f"Error filtering {stat_type} data: {str(e)}"

# Update the export_data callback
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
    stat_type = 'disposals'  # default
    if active_tab == "tab-marks":
        stat_type = 'marks'
    elif active_tab == "tab-tackles":
        stat_type = 'tackles'
    
    # Get the full dataset for this stat type
    full_data = processed_data_by_stat.get(stat_type)
    
    # If we have data, export it
    if full_data is not None and not full_data.empty:
        # Create a filename with the stat type and current datetime
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"afl_{stat_type}_dashboard_{current_time}.csv"
        
        # Return the data as a CSV download
        return dcc.send_data_frame(full_data.to_csv, filename, index=False)
    
    return dash.no_update

# Run the app
if __name__ == '__main__':
    app.run(debug=True)