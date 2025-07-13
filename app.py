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
from dabble_scraper import get_pickem_data_for_dashboard, normalize_player_name  # NEW IMPORT

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
                
                # Define thresholds based on stat type
                if stat_type == 'disposals':
                    strong_threshold = 2.0  # Strong Easy/Unders threshold
                    moderate_threshold = 1.0  # Moderate Easy/Unders threshold
                    slight_threshold = 0.1  # Slight Easy/Unders threshold
                elif stat_type in ['marks', 'tackles']:
                    strong_threshold = 1.0  # Lower thresholds for marks/tackles
                    moderate_threshold = 0.5
                    slight_threshold = 0.05
                
                # Categorize DvP (both positive and negative)
                if dvp >= strong_threshold:
                    strength = "Strong Easy"  # Strong positive DvP = Easy overs
                elif dvp >= moderate_threshold:
                    strength = "Moderate Easy"  # Moderate positive DvP
                elif dvp >= slight_threshold:
                    strength = "Slight Easy"  # Slight positive DvP
                elif dvp <= -strong_threshold:
                    strength = "Strong Unders"  # Strong negative DvP = Unders
                elif dvp <= -moderate_threshold:
                    strength = "Moderate Unders"  # Moderate negative DvP
                elif dvp <= -slight_threshold:
                    strength = "Slight Unders"  # Slight negative DvP
                else:
                    strength = "Neutral"  # Between -slight and +slight thresholds
                
                # Only store if it's not neutral (to keep the dictionary smaller)
                if strength != "Neutral":
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
    """Calculate bet flag based on proven strategies with win rates - ONLY THE 7 SPECIFIED STRATEGIES"""
    try:
        # Extract values from the row
        position = player_row.get('Position', player_row.get('position', ''))
        weather = player_row.get('Weather', player_row.get('weather', ''))
        dvp = player_row.get('DvP', player_row.get('dvp', ''))
        travel_fatigue = player_row.get('Travel Fatigue', player_row.get('travel_fatigue', ''))
        line_str = player_row.get('Line', '')
        avg_vs_line_str = player_row.get('Avg vs Line', '')
        
        # Parse the line value
        line_value = None
        if line_str and line_str != "":
            try:
                line_value = float(line_str)
            except (ValueError, TypeError):
                line_value = None
        
        # Parse the Avg vs Line percentage
        avg_vs_line_pct = None
        if avg_vs_line_str and avg_vs_line_str != "":
            try:
                # Remove % and + signs, convert to float
                avg_vs_line_pct = float(avg_vs_line_str.replace('%', '').replace('+', ''))
            except (ValueError, TypeError):
                avg_vs_line_pct = None
        
        # Parse travel and weather conditions
        has_long_travel = 'Long Travel' in travel_fatigue
        has_moderate_travel = 'Moderate' in travel_fatigue
        
        # Check rain and weather conditions
        has_rain = 'Rain' in weather and 'Neutral' not in weather
        has_neutral_weather = 'Neutral' in weather
        
        # Check DvP levels - UPDATED TO EXCLUDE EASY DVP
        has_moderate_unders_dvp = 'Moderate Unders' in dvp
        has_strong_unders_dvp = 'Strong Unders' in dvp
        has_easy_dvp = any(x in dvp for x in ['Strong Easy', 'Moderate Easy', 'Slight Easy'])
        
        # AUTO-SKIP RULES (CHECK FIRST - HIGHEST PRIORITY)
        # SKIP if no line available - RETURN BLANK VALUES
        if line_value is None:
            return {"priority": "", "tier": "", "description": ""}
        
        # ONLY THE 7 SPECIFIED STRATEGIES - NOW WITH AUTOMATED AVG CRITERIA
        
        # MARKS STRATEGIES
        if stat_type == 'marks':
            # Strategy #1: KeyF + Mark + Line >5.0 + No Easy DvP → 92% WR (13 bets)
            if position == 'KeyF' and line_value > 5.0 and not has_easy_dvp:
                return {"priority": "1", "tier": "🥇", "description": "KeyF + Mark + Line >5.0 + No Easy DvP"}
            
            # Strategy #3: GenD + Mark + Avg <0% + No Easy DvP + Line >5 → 91.7% WR (12 bets)
            if (position == 'GenD' and line_value > 5 and not has_easy_dvp and 
                avg_vs_line_pct is not None and avg_vs_line_pct < 0):
                return {"priority": "3", "tier": "🥇", "description": "GenD + Mark + Avg <0% + No Easy DvP + Line >5"}
        
        # TACKLE STRATEGIES
        elif stat_type == 'tackles':
            # Strategy #2: Tackle + Strong Unders DvP + No Rain + Avg <15% → 93.3% WR (15 bets)
            if (has_strong_unders_dvp and not has_rain and 
                avg_vs_line_pct is not None and avg_vs_line_pct < 15):
                return {"priority": "2", "tier": "🥇", "description": "Tackle + Strong Unders DvP + No Rain + Avg <15%"}
            
            # Strategy #5: Tackle + Moderate Travel + Avg <5% + No Easy DvP + No Rain → 83.3% WR (17 bets)
            if (has_moderate_travel and not has_easy_dvp and not has_rain and 
                avg_vs_line_pct is not None and avg_vs_line_pct < 5):
                return {"priority": "5", "tier": "🥈", "description": "Tackle + Moderate Travel + Avg <5% + No Easy DvP + No Rain"}
        
        # DISPOSAL STRATEGIES
        elif stat_type == 'disposals':
            # Strategy #6: Disposal + Line >27 + Strong/Moderate Unders DvP Only → 75.0% WR (8 bets)
            if line_value > 27 and (has_strong_unders_dvp or has_moderate_unders_dvp) and not has_easy_dvp:
                return {"priority": "6", "tier": "🥉", "description": "Disposal + Line >27 + Strong/Moderate Unders DvP Only"}
            
            # Strategy #7: Long Travel + Disposal + Avg ≤-10% + No Easy DvP → 73.3% WR (15 bets)
            if (has_long_travel and not has_easy_dvp and 
                avg_vs_line_pct is not None and avg_vs_line_pct <= -10):
                return {"priority": "7", "tier": "🥉", "description": "Long Travel + Disposal + Avg ≤-10% + No Easy DvP"}
        
        # Default skip for anything that doesn't match the 7 strategies
        return {"priority": "", "tier": "", "description": "SKIP - No Strategy"}
        
    except Exception as e:
        print(f"ERROR in calculate_bet_flag: {e}")
        import traceback
        traceback.print_exc()
        return {"priority": "", "tier": "", "description": "ERROR"}

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
        df[score_column] = df[score_column].astype(str)  # Ensure column is string type
        df.at[idx, score_column] = f"{score_value:.1f} - {rating}"
        df.at[idx, 'ScoreFactors'] = score_data["Factors"]
    
    return df

def add_bet_flag_to_dataframe(df, stat_type='disposals'):
    """Add bet flag columns to the dataframe"""
    bet_results = df.apply(lambda row: calculate_bet_flag(row, stat_type), axis=1)
    
    df['Bet_Priority'] = bet_results.apply(lambda x: x['priority'])
    df['Bet_Tier'] = bet_results.apply(lambda x: x['tier'])
    df['Bet_Flag'] = bet_results.apply(lambda x: x['description'])
    
    return df

# NEW FUNCTION: Add pickem lines to dataframe
def add_pickem_lines_to_dataframe(df, stat_type='disposals'):
    """Add pickem line data to the dataframe with enhanced name matching"""
    print(f"🎯 Adding pickem lines for {stat_type}...")
    
    try:
        # Get pickem data for this stat type
        pickem_data = get_pickem_data_for_dashboard(stat_type)
        
        if not pickem_data:
            print(f"⚠️ No pickem data found for {stat_type}, adding empty Line column")
            df['Line'] = ""
            return df
        
        print(f"📊 Got {len(pickem_data)} pickem lines for {stat_type}")
        
        # Create a more robust mapping function
        def get_player_line(player_name):
            if not player_name or pd.isna(player_name):
                return ""
            
            player_name = str(player_name).strip()
            
            # Method 1: Direct exact match (case insensitive)
            for pickem_player, line_value in pickem_data.items():
                if player_name.lower() == pickem_player.lower():
                    print(f"  ✅ Direct match: '{player_name}' → {line_value}")
                    return str(line_value)
            
            # Method 2: Try normalized versions
            try:
                normalized_name = normalize_player_name(player_name)
                for pickem_player, line_value in pickem_data.items():
                    normalized_pickem = normalize_player_name(pickem_player)
                    if normalized_name.lower() == normalized_pickem.lower():
                        print(f"  ✅ Normalized match: '{player_name}' → '{pickem_player}' → {line_value}")
                        return str(line_value)
            except Exception as e:
                print(f"  ⚠️ Error in normalize_player_name: {e}")
            
            # Method 3: Remove spaces and compare
            player_no_space = player_name.replace(" ", "").lower()
            for pickem_player, line_value in pickem_data.items():
                pickem_no_space = pickem_player.replace(" ", "").lower()
                if player_no_space == pickem_no_space:
                    print(f"  ✅ No-space match: '{player_name}' → '{pickem_player}' → {line_value}")
                    return str(line_value)
            
            # Method 4: Last name + first initial matching
            try:
                player_parts = player_name.split()
                if len(player_parts) >= 2:
                    player_last = player_parts[-1].lower()
                    player_first_initial = player_parts[0][0].lower()
                    
                    for pickem_player, line_value in pickem_data.items():
                        pickem_parts = pickem_player.split()
                        if len(pickem_parts) >= 2:
                            pickem_last = pickem_parts[-1].lower()
                            pickem_first_initial = pickem_parts[0][0].lower()
                            
                            if (player_last == pickem_last and 
                                player_first_initial == pickem_first_initial):
                                print(f"  ✅ Initial match: '{player_name}' → '{pickem_player}' → {line_value}")
                                return str(line_value)
            except (IndexError, AttributeError):
                pass
            
            # Method 5: Partial name matching (for nicknames, etc.)
            for pickem_player, line_value in pickem_data.items():
                # Check if either name contains the other (for cases like "Sam" vs "Samuel")
                if (player_name.lower() in pickem_player.lower() or 
                    pickem_player.lower() in player_name.lower()):
                    # Make sure it's not a substring match that's too short
                    if min(len(player_name), len(pickem_player)) >= 4:
                        print(f"  ✅ Partial match: '{player_name}' → '{pickem_player}' → {line_value}")
                        return str(line_value)
            
            # No match found
            return ""
        
        # Apply the mapping to add Line column
        print(f"🔍 Attempting to match {len(df)} players...")
        df['Line'] = df['player'].apply(get_player_line)
        
        # Count successful matches
        matched_count = sum(1 for line in df['Line'] if line != "")
        total_players = len(df)
        
        print(f"✅ Successfully matched {matched_count}/{total_players} players with pickem lines")
        
        # Show some examples of successful and failed matches
        matched_players = df[df['Line'] != ""][['player', 'Line']].head(5)
        unmatched_players = df[df['Line'] == ""][['player']].head(5)
        
        if not matched_players.empty:
            print("📋 Sample successful matches:")
            for _, row in matched_players.iterrows():
                print(f"   ✅ {row['player']}: {row['Line']}")
        
        if not unmatched_players.empty:
            print("📋 Sample unmatched players:")
            for _, row in unmatched_players.iterrows():
                print(f"   ❌ {row['player']}")
            
            # Show what pickem players we have for comparison
            print("📋 Sample available pickem players:")
            for player in list(pickem_data.keys())[:5]:
                print(f"   📊 {player}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error adding pickem lines for {stat_type}: {e}")
        import traceback
        traceback.print_exc()
        df['Line'] = ""
        return df


def add_line_analysis_columns(df, stat_type='disposals'):
    """Add Avg vs Line and Line Consistency columns to the dataframe"""
    print(f"📊 Adding line analysis columns for {stat_type}...")
    
    try:
        # Load the full season stats
        stats_df = pd.read_csv("afl_player_stats.csv", skiprows=3)
        stats_df = stats_df.fillna(0)
        
        # Ensure we have the stat column
        if stat_type not in stats_df.columns:
            if stat_type == 'disposals' and 'kicks' in stats_df.columns and 'handballs' in stats_df.columns:
                stats_df[stat_type] = stats_df['kicks'] + stats_df['handballs']
            else:
                print(f"⚠️ Warning: {stat_type} column not found in stats data")
                df['Avg vs Line'] = ""
                df['Line Consistency'] = ""
                return df
        
        # Initialize the new columns
        df['Avg vs Line'] = ""
        df['Line Consistency'] = ""
        
        print(f"🔍 Processing {len(df)} players for line analysis...")
        
        # Process each player in the dataframe
        for idx, row in df.iterrows():
            player_name = row.get('Player', row.get('player', ''))
            line_str = row.get('Line', '')
            
            # Skip if no line available
            if not line_str or line_str == "":
                df.at[idx, 'Avg vs Line'] = ""
                df.at[idx, 'Line Consistency'] = ""
                continue
            
            # Parse line value
            try:
                line_value = float(line_str)
            except (ValueError, TypeError):
                df.at[idx, 'Avg vs Line'] = ""
                df.at[idx, 'Line Consistency'] = ""
                continue
            
            # Find player's season stats
            player_stats = stats_df[stats_df['player'].str.lower() == player_name.lower()]
            
            if player_stats.empty:
                # Try fuzzy matching if exact match fails
                for stats_player in stats_df['player'].unique():
                    if (stats_player.lower().replace(" ", "") == player_name.lower().replace(" ", "") or
                        (len(player_name.split()) >= 2 and len(stats_player.split()) >= 2 and
                         player_name.split()[-1].lower() == stats_player.split()[-1].lower() and
                         player_name.split()[0][0].lower() == stats_player.split()[0][0].lower())):
                        player_stats = stats_df[stats_df['player'] == stats_player]
                        break
            
            if player_stats.empty:
                print(f"   ⚠️ No stats found for {player_name}")
                df.at[idx, 'Avg vs Line'] = ""
                df.at[idx, 'Line Consistency'] = ""
                continue
            
            # Calculate season average
            stat_values = player_stats[stat_type].values
            season_avg = stat_values.mean()
            
            # 1. Calculate Avg vs Line: (Line / Avg) - 1
            if season_avg > 0:
                avg_vs_line_ratio = (line_value / season_avg) - 1
                avg_vs_line_pct = avg_vs_line_ratio * 100
                df.at[idx, 'Avg vs Line'] = f"{avg_vs_line_pct:+.1f}%"
            else:
                df.at[idx, 'Avg vs Line'] = ""
            
            # 2. Calculate Line Consistency: % of games below the line
            games_below_line = sum(1 for value in stat_values if value < line_value)
            total_games = len(stat_values)
            
            if total_games > 0:
                consistency_pct = (games_below_line / total_games) * 100
                df.at[idx, 'Line Consistency'] = f"{consistency_pct:.1f}%"
            else:
                df.at[idx, 'Line Consistency'] = ""
            
            # Log some examples for verification
            if idx < 5:  # Log first 5 for debugging
                print(f"   ✅ {player_name}: Line={line_value}, Avg={season_avg:.1f}, "
                      f"Avg vs Line={df.at[idx, 'Avg vs Line']}, "
                      f"Below Line={games_below_line}/{total_games} = {df.at[idx, 'Line Consistency']}")
        
        # Count successful calculations
        successful_avg = sum(1 for val in df['Avg vs Line'] if val != "")
        successful_consistency = sum(1 for val in df['Line Consistency'] if val != "")
        
        print(f"✅ Successfully calculated:")
        print(f"   - Avg vs Line: {successful_avg}/{len(df)} players")
        print(f"   - Line Consistency: {successful_consistency}/{len(df)} players")
        
        return df
        
    except Exception as e:
        print(f"❌ Error adding line analysis columns: {e}")
        import traceback
        traceback.print_exc()
        df['Avg vs Line'] = ""
        df['Line Consistency'] = ""
        return df


def interpret_line_analysis(avg_vs_line_str, consistency_str):
    """Helper function to interpret the line analysis values"""
    interpretation = []
    
    # Interpret Avg vs Line
    if avg_vs_line_str and avg_vs_line_str != "":
        try:
            avg_vs_line = float(avg_vs_line_str.replace('%', '').replace('+', ''))
            
            if avg_vs_line > 10:
                interpretation.append("🔴 Line Much Higher Than Avg (Strong Unders)")
            elif avg_vs_line > 5:
                interpretation.append("🟠 Line Higher Than Avg (Unders)")
            elif avg_vs_line > -5:
                interpretation.append("🟡 Line Close To Avg (Neutral)")
            elif avg_vs_line > -10:
                interpretation.append("🔵 Line Lower Than Avg (Overs)")
            else:
                interpretation.append("🟢 Line Much Lower Than Avg (Strong Overs)")
        except:
            pass
    
    # Interpret Line Consistency
    if consistency_str and consistency_str != "":
        try:
            consistency = float(consistency_str.replace('%', ''))
            
            if consistency >= 70:
                interpretation.append("🎯 Very Consistent Unders (70%+ below line)")
            elif consistency >= 60:
                interpretation.append("📈 Consistent Unders (60%+ below line)")
            elif consistency >= 40:
                interpretation.append("⚖️ Balanced (40-60% below line)")
            elif consistency >= 30:
                interpretation.append("📉 Consistent Overs (30%+ above line)")
            else:
                interpretation.append("🚀 Very Consistent Overs (70%+ above line)")
        except:
            pass
    
    return " | ".join(interpretation) if interpretation else ""


# Process dashboard data for a specific stat type
def process_data_for_dashboard(stat_type='disposals'):
    try:
        print("🔴🔴🔴 UPDATED CODE IS RUNNING - VERSION 2.0 🔴🔴🔴")
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
            # The print statements we're seeing are coming from build_travel_log() itself
            # We need to process the data here without relying on its output
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
        
        # Step 8: Add travel fatigue using hardcoded long travel pairs
        travel_dict = {}

        print("\n" + "="*80)
        print("PROCESSING TRAVEL FATIGUE USING HARDCODED PAIRS ONLY:")
        print("="*80)

        long_distance_pairs = set([
            # WA → Victoria
            ("WCE", "MEL"), ("WCE", "GEE"), ("WCE", "COL"), ("WCE", "HAW"),
            ("WCE", "CAR"), ("WCE", "ESS"), ("WCE", "NTH"), ("WCE", "RIC"),
            ("WCE", "STK"), ("WCE", "WBD"),
            ("FRE", "MEL"), ("FRE", "GEE"), ("FRE", "COL"), ("FRE", "HAW"),
            ("FRE", "CAR"), ("FRE", "ESS"), ("FRE", "NTH"), ("FRE", "RIC"),
            ("FRE", "STK"), ("FRE", "WBD"),

            # WA → East coast
            ("WCE", "SYD"), ("WCE", "GWS"), ("WCE", "BRL"), ("WCE", "GCS"),
            ("FRE", "SYD"), ("FRE", "GWS"), ("FRE", "BRL"), ("FRE", "GCS"),

            # Adelaide → Perth
            ("ADE", "WCE"), ("ADE", "FRE"), ("PTA", "WCE"), ("PTA", "FRE"),

            # Perth → Adelaide
            ("WCE", "ADE"), ("FRE", "ADE"), ("WCE", "PTA"), ("FRE", "PTA"),

            # QLD → Victoria
            ("BRL", "MEL"), ("BRL", "GEE"), ("BRL", "COL"), ("BRL", "HAW"),
            ("BRL", "CAR"), ("BRL", "ESS"), ("BRL", "NTH"), ("BRL", "RIC"),
            ("BRL", "STK"), ("BRL", "WBD"),
            ("GCS", "MEL"), ("GCS", "GEE"), ("GCS", "COL"), ("GCS", "HAW"),
            ("GCS", "CAR"), ("GCS", "ESS"), ("GCS", "NTH"), ("GCS", "RIC"),
            ("GCS", "STK"), ("GCS", "WBD"),

            # QLD → Adelaide
            ("BRL", "ADE"), ("BRL", "PTA"), ("GCS", "ADE"), ("GCS", "PTA"),

            # East coast → WA
            ("SYD", "WCE"), ("SYD", "FRE"), ("GWS", "WCE"), ("GWS", "FRE"),
            ("BRL", "WCE"), ("BRL", "FRE"), ("GCS", "WCE"), ("GCS", "FRE"),

            # Victoria → WA
            ("MEL", "WCE"), ("GEE", "WCE"), ("COL", "WCE"), ("HAW", "WCE"),
            ("CAR", "WCE"), ("ESS", "WCE"), ("NTH", "WCE"), ("RIC", "WCE"),
            ("STK", "WCE"), ("WBD", "WCE"),
            ("MEL", "FRE"), ("GEE", "FRE"), ("COL", "FRE"), ("HAW", "FRE"),
            ("CAR", "FRE"), ("ESS", "FRE"), ("NTH", "FRE"), ("RIC", "FRE"),
            ("STK", "FRE"), ("WBD", "FRE"),

            # Victoria → QLD
            ("MEL", "BRL"), ("GEE", "BRL"), ("COL", "BRL"), ("HAW", "BRL"),
            ("CAR", "BRL"), ("ESS", "BRL"), ("NTH", "BRL"), ("RIC", "BRL"),
            ("STK", "BRL"), ("WBD", "BRL"),
            ("MEL", "GCS"), ("GEE", "GCS"), ("COL", "GCS"), ("HAW", "GCS"),
            ("CAR", "GCS"), ("ESS", "GCS"), ("NTH", "GCS"), ("RIC", "GCS"),
            ("STK", "GCS"), ("WBD", "GCS"),

            # Adelaide → QLD
            ("ADE", "BRL"), ("ADE", "GCS"), ("PTA", "BRL"), ("PTA", "GCS")
        ])

        latest_round = df['round'].max()

        # Build travel fatigue for teams playing in next round
        # First, determine home/away teams from fixture data
        home_away_mapping = {}
        for match in fixtures:
            match_str = match["match"]
            try:
                home_team_full, away_team_full = match_str.split(" vs ")
                
                # Find abbreviations
                home_abbr = None
                away_abbr = None
                for abbr, name in TEAM_NAME_MAP.items():
                    if name == home_team_full:
                        home_abbr = abbr
                    if name == away_team_full:
                        away_abbr = abbr
                
                if home_abbr and away_abbr:
                    home_away_mapping[home_abbr] = {'opponent': away_abbr, 'is_home': True}
                    home_away_mapping[away_abbr] = {'opponent': home_abbr, 'is_home': False}
                    
            except Exception as e:
                print(f"Error parsing match for home/away: {match_str}")

        # Now process travel fatigue - only flag away teams that are traveling long distance
        for team_abbr in next_round_players['team'].unique():
            full_name = TEAM_NAME_MAP.get(team_abbr, team_abbr)
            
            # Get opponent and home/away status
            team_info = home_away_mapping.get(team_abbr, {})
            opponent_abbr = team_info.get('opponent')
            is_home = team_info.get('is_home', True)  # Default to home if unknown
            
            # Only check travel fatigue for away teams
            if not is_home and opponent_abbr:
                pair = (team_abbr, opponent_abbr)
                flagged = pair in long_distance_pairs
                
                # DEBUG: print exactly what's being checked
                print(f"Checking travel for: {team_abbr} (AWAY) vs {opponent_abbr} (HOME) → Flagged: {flagged}")
                
                emoji = "🟠 Moderate (Long Travel)" if flagged else "✅ Neutral"
            else:
                # Home teams don't get travel fatigue
                flagged = False
                status = "HOME" if is_home else "UNKNOWN"
                print(f"Checking travel for: {team_abbr} ({status}) vs {opponent_abbr} → Flagged: {flagged}")
                emoji = "✅ Neutral"
            
            travel_dict[team_abbr] = emoji
            print(f" - {full_name} ({team_abbr}): {emoji}")

        # Map travel fatigue to players
        next_round_players['travel_fatigue'] = next_round_players['team'].map(
            lambda x: travel_dict.get(x, "✅ Neutral")
        )

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
                    
                    # Return DvP rating with blue icons for Easy and red for Unders
                    if strength == "Strong Unders":
                        return "🔴 Strong Unders"
                    elif strength == "Moderate Unders":
                        return "🟠 Moderate Unders"
                    elif strength == "Slight Unders":
                        return "🟡 Slight Unders"
                    elif strength == "Strong Easy":
                        return "🔵 Strong Easy"
                    elif strength == "Moderate Easy":
                        return "🔷 Moderate Easy"
                    elif strength == "Slight Easy":
                        return "🔹 Slight Easy"
                
                return "✅ Neutral"
            except Exception as e:
                print(f"Error in DvP rating: {e}")
                return "⚠️ Unknown"
        
        next_round_players['dvp'] = next_round_players.apply(get_dvp_rating, axis=1)
        
        # Step 13: Add pickem lines (NEW STEP)
        result_df = add_pickem_lines_to_dataframe(next_round_players, stat_type)
        
        # Step 13.5: Add line analysis columns (NEW STEP)
        result_df = add_line_analysis_columns(result_df, stat_type)

        # Step 14: Clean up and select final columns for display (UPDATED to include Line)
        result_df = result_df[['player', 'team', 'opponent', 'position', 'travel_fatigue', 
                      'weather', 'dvp', 'Line', 'Avg vs Line', 'Line Consistency']].copy()
        
        # Calculate the Unders Score for each player (for bet flag calculation only)
        result_df = add_score_to_dataframe(result_df, team_weather, simplified_dvp, stat_type)
        
        # Add bet flag based on filtering criteria
        result_df = add_bet_flag_to_dataframe(result_df, stat_type)
        
        # Final columns for display (ADDED Line column)
        # This selects 13 columns to match your 13-item mapping
        display_df = result_df[['player', 'team', 'opponent', 'position', 
                        'travel_fatigue', 'weather', 'dvp', 'Line', 
                        'Avg vs Line', 'Line Consistency',
                        'Bet_Priority', 'Bet_Tier', 'Bet_Flag']].copy()
        
        # Rename columns for display (ADDED Line column)
        column_mapping = {
    'player': 'Player', 
    'team': 'Team', 
    'opponent': 'Opponent', 
    'position': 'Position',
    'travel_fatigue': 'Travel Fatigue', 
    'weather': 'Weather', 
    'dvp': 'DvP',
    'Line': 'Line',
    'Avg vs Line': 'Avg vs Line',
    'Line Consistency': 'Line Consistency',
    'Bet_Priority': 'Bet Priority',
    'Bet_Tier': 'Bet Tier',
    'Bet_Flag': 'Bet Flag'
}
        # Apply the column renaming
        display_df.columns = list(column_mapping.values())
        
        # Filter out rows with no line data
        display_df = display_df[display_df['Line'] != ""].copy()
        print(f"After filtering out players with no lines: {len(display_df)} rows remaining")
        
        # Sort by Bet Priority (ascending) then by Team (ascending)
        # First, handle the sorting properly for Bet Priority column
        def sort_bet_priority(priority_str):
            """Convert bet priority to numeric for proper sorting"""
            if pd.isna(priority_str) or priority_str == "" or priority_str == "SKIP":
                return 999  # Put non-priorities at the end
            try:
                return int(priority_str)
            except (ValueError, TypeError):
                return 999  # Put invalid values at the end
        
        # Create a temporary column for sorting
        display_df['_sort_priority'] = display_df['Bet Priority'].apply(sort_bet_priority)
        
        # Sort by priority first (ascending), then by team (ascending)
        display_df = display_df.sort_values(['_sort_priority', 'Team'], ascending=[True, True])
        
        # Remove the temporary sorting column
        display_df = display_df.drop('_sort_priority', axis=1)
        
        print(f"Final dashboard data for {stat_type} has {len(display_df)} rows")
        print(f"Filtered to show only players with betting lines available")
        print(f"Sorted by: Bet Priority (ascending), then Team (ascending)")
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
            'Line': 'Error',
            'Bet Priority': 'Error',
            'Bet Tier': 'Error',
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
                            html.Span("🟠 Long Travel", className="badge bg-warning me-2")
                        ], className="d-flex justify-content-center")
                    ], className="border rounded p-2 mb-3", 
                    id="travel-fatigue-legend",
                    title="Travel fatigue now only considers long-distance travel (>1500km). Teams traveling across the country get the Long Travel flag.")
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
                            html.Span("🟡🟠🔴 Unders", className="badge bg-warning text-dark me-2"),
                            html.Span("🔹🔷🔵 Easy", className="badge bg-info me-2")
                        ], className="d-flex justify-content-center flex-wrap")
                    ], className="border rounded p-2 mb-3",
                    id="dvp-legend",
                    title="Defense vs Position shows historical matchup difficulty. 🔴🟠🟡 Unders = opponent allows fewer stats than average (good for unders bets). 🔵🔷🔹 Easy = opponent allows more stats than average (good for overs bets). Intensity shows strength: Slight < Moderate < Strong.")
                ], width=6),
                
                dbc.Col([
                    html.Div([
                        html.H4("Bet Tier", className="text-center"),
                        html.Div([
                            html.Span("🥇 Tier 1 (90%+)", className="badge bg-success me-2"),
                            html.Span("🥈 Tier 2 (80-90%)", className="badge bg-warning text-dark me-2"),
                            html.Span("🥉 Tier 3 (65-80%)", className="badge bg-info")
                        ], className="d-flex justify-content-center flex-wrap")
                    ], className="border rounded p-2 mb-3",
                    id="bet-tier-legend",
                    title="Bet tiers based on win rates: 🥇 Tier 1 (90%+), 🥈 Tier 2 (80-90%), 🥉 Tier 3 (65-80%). Higher tiers have better historical performance.")
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
                        'Line': '25.5',
                        'Bet Priority': '#1',
                        'Bet Tier': '🥇',
                        'Bet Flag': 'Tackle + Moderate Travel + Avg <5%'
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
        
        # Create the conditional styling (UPDATED FOR NEW COLUMNS)
        style_data_conditional = [
            # Travel Fatigue - UPDATED
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

            # DvP - Updated to handle both Unders and Easy
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Neutral"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Slight"'},
             'backgroundColor': '#fff3cd', 'color': 'black'},
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Moderate"'},
             'backgroundColor': '#ffe066', 'color': 'black'},
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Strong"'},
             'backgroundColor': '#f8d7da', 'color': 'black'},
             
            # Line column - highlight based on whether line exists (NEW)
            {'if': {'column_id': 'Line', 'filter_query': '{Line} != ""'},
             'backgroundColor': '#e8f5e8', 'color': 'black', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Line', 'filter_query': '{Line} = ""'},
             'backgroundColor': '#f8f9fa', 'color': '#6c757d'},
             
            # Bet Tier colors - UPDATED FOR NEW TIER SYSTEM
            {'if': {'column_id': 'Bet Tier', 'filter_query': '{Bet Tier} contains "🥇"'},
             'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Tier', 'filter_query': '{Bet Tier} contains "🥈"'},
             'backgroundColor': '#ffc107', 'color': 'black', 'fontWeight': 'bold'},
            {'if': {'column_id': 'Bet Tier', 'filter_query': '{Bet Tier} contains "🥉"'},
             'backgroundColor': '#cd7f32', 'color': 'white', 'fontWeight': 'bold'},
             
            # Bet Flag colors - UPDATED FOR COMBO STRATEGIES
            {'if': {'column_id': 'Bet Flag', 'filter_query': '{Bet Flag} contains "COMBO"'},
             'backgroundColor': '#e6ccff', 'color': 'black', 'fontWeight': 'bold'},
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
            'Line': 'N/A',
            'Bet Priority': 'N/A',
            'Bet Tier': 'N/A',
            'Bet Flag': 'N/A'
        }])
        columns = [{"name": i, "id": i} for i in error_df.columns]
        return error_df.to_dict('records'), columns, [], [], f"Error filtering {stat_type} data: {str(e)}"

# Add this debug code temporarily to your app.py to test what's happening
def debug_pickem_matching(stat_type='disposals'):
    """Debug function to see what's happening with pickem data"""
    print(f"\n=== DEBUGGING PICKEM DATA FOR {stat_type.upper()} ===")
    
    try:
        # Test 1: Check if get_pickem_data_for_dashboard works
        from dabble_scraper import get_pickem_data_for_dashboard, normalize_player_name
        pickem_data = get_pickem_data_for_dashboard(stat_type)
        
        print(f"1. Pickem data retrieved: {len(pickem_data) if pickem_data else 0} players")
        if pickem_data:
            print(f"   Sample pickem players: {list(pickem_data.keys())[:5]}")
            print(f"   Sample pickem lines: {list(pickem_data.values())[:5]}")
        else:
            print("   ❌ No pickem data returned!")
            return
        
        # Test 2: Check dataframe player names
        try:
            df = pd.read_csv("afl_player_stats.csv", skiprows=3)
            latest_round = df['round'].max()
            recent_players = df[df['round'] == latest_round]['player'].unique()[:5]
            print(f"2. Sample dataframe players: {recent_players.tolist()}")
        except Exception as e:
            print(f"   ❌ Error reading dataframe: {e}")
            return
        
        # Test 3: Test normalize_player_name function
        test_names = ["Patrick Cripps", "Marcus Bontempelli", "Lachie Neale"]
        print("3. Testing normalize_player_name:")
        for name in test_names:
            normalized = normalize_player_name(name)
            print(f"   '{name}' → '{normalized}'")
        
        # Test 4: Try manual matching
        print("4. Testing manual matching:")
        for df_player in recent_players[:3]:
            normalized_df = normalize_player_name(df_player)
            found_match = False
            
            for pickem_player in list(pickem_data.keys())[:10]:  # Check first 10
                normalized_pickem = normalize_player_name(pickem_player)
                
                if normalized_df.lower() == normalized_pickem.lower():
                    print(f"   ✅ MATCH: '{df_player}' ↔ '{pickem_player}'")
                    found_match = True
                    break
            
            if not found_match:
                print(f"   ❌ NO MATCH: '{df_player}' (normalized: '{normalized_df}')")
        
        print("=== END DEBUG ===\n")
        
    except Exception as e:
        print(f"❌ Debug error: {e}")
        import traceback
        traceback.print_exc()

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

debug_pickem_matching('disposals')
debug_pickem_matching('marks') 
debug_pickem_matching('tackles')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)# Score column - gradient coloring based on score value