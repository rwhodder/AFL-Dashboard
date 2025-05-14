import pandas as pd
import numpy as np
import dash
from dash import html, dcc, dash_table, Input, Output
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
    rain = weather.get('rain', 0)
    wind = weather.get('wind', 0)
    humidity = weather.get('humidity', 0)
    
    rain_cat = "Low" if rain == 0 else "Med" if rain <= 2 else "High"
    wind_cat = "Low" if wind <= 10 else "Med" if wind <= 20 else "High"
    humid_cat = "Low" if humidity <= 60 else "Med" if humidity <= 85 else "High"
    
    rating = "✅ Neutral"
    if rain > 2 or wind > 6 or humidity > 85:
        rating = "⚠️ Strong Unders Edge"
        
    return {
        "Rain": rain_cat,
        "Wind": wind_cat,
        "Humidity": humid_cat,
        "Rating": rating
    }

# ===== TOG/CBA FUNCTIONS =====
def calculate_tog_cba_trends(df):
    # Filter for last 5 rounds
    latest_round = df['round'].max()
    recent_df = df[df['round'] >= latest_round - 4]
    
    # Calculate TOG/CBA trend slopes
    def calc_trend_slope(group):
        rounds = group['round']
        tog_slope = np.polyfit(rounds, group['tog'], 1)[0] if len(group) >= 2 else 0
        cba_slope = np.polyfit(rounds, group['cbas'], 1)[0] if len(group) >= 2 else 0
        return pd.Series({'TOG_slope': tog_slope, 'CBA_slope': cba_slope})
    
    slopes = recent_df.groupby(['player', 'team']).apply(calc_trend_slope).reset_index()
    
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
    # Step 1: Get next round fixtures
    fixtures = scrape_next_round_fixture()
    if not fixtures:
        print("⚠️ No fixture data found.")
        return pd.DataFrame()  # Empty DataFrame as fallback
    
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
                print(f"⚠️ Unknown venue: {stadium}")
                continue
        
        forecast_list = get_forecast(*latlon)
        weather = extract_weather_for_datetime(forecast_list, match["datetime"])
        
        match_key = match["match"]
        if weather:
            weather_data[match_key] = weather
        else:
            print(f"⚠️ No forecast found for {match['match']} at {stadium}")
    
    # Step 3: Process travel fatigue data
    travel_log = build_travel_log()
    
    # Step 4: Load player stats
    df = pd.read_csv("afl_player_stats.csv", skiprows=3)
    df = df.fillna(0)  # Fill NA values with 0 for numerical calculations
    
    # Step 5: Calculate TOG/CBA trends
    trend_data = calculate_tog_cba_trends(df)
    
    # Step 6: Generate DvP stats 
    processed_df = load_and_prepare_data("afl_player_stats.csv")
    dvp_disposals = calculate_dvp(processed_df, "disposals")
    
    # Step 7: Extract team matchups for next round
    teams_playing = []
    team_weather = {}
    team_opponents = {}
    for match in fixtures:
        match_str = match["match"]
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
    
    # Step 8: Filter for players in the upcoming round only
    latest_round = df['round'].max()
    latest_data = df[df['round'] == latest_round]
    
    next_round_players = latest_data[latest_data['team'].isin(teams_playing)].copy()
    
    # Step 9: Add travel fatigue data
    travel_dict = {}
    for entry in travel_log:
        if entry.get('round') == latest_round + 1:  # next round
            team = entry.get('team')
            fatigue = entry.get('fatigue_score', 0)
            emoji = "✅ Low" if fatigue < 1 else "⚠️ Medium" if fatigue < 2 else "🔴 High"
            travel_dict[team] = emoji
    
    next_round_players['travel_fatigue'] = next_round_players['team'].map(travel_dict)
    next_round_players['travel_fatigue'] = next_round_players['travel_fatigue'].fillna("⚠️ Unknown")
    
    # Step 10: Add weather data
    next_round_players['weather'] = next_round_players['team'].map(lambda x: team_weather.get(x, {}).get('Rating', "⚠️ Unknown"))
    
    # Step 11: Add opponent for DvP
    next_round_players['opponent'] = next_round_players['team'].map(team_opponents)
    
    # Step 12: Add role positions
    def map_position(pos):
        for role, tags in POSITION_MAP.items():
            if pos in tags:
                return role
        return "Unknown"
    
    next_round_players["position"] = next_round_players["namedPosition"].apply(map_position)
    
    # Step 13: Add DvP indicators
    def get_dvp_rating(row):
        team = row['opponent']
        pos = row['position']
        
        # Look for matching DvP entries
        matching = dvp_disposals[(dvp_disposals['opponentTeam'] == team) & 
                                (dvp_disposals['role'] == pos)]
        
        if not matching.empty:
            dvp_value = matching.iloc[0]['dvp']
            if dvp_value <= -4.0:
                return "🔴 Strong Unders"
            elif dvp_value <= -2.0:
                return "🟠 Moderate Unders"
            else:
                return "🟡 Slight Unders"
        return "✅ Neutral"
    
    next_round_players['dvp'] = next_round_players.apply(get_dvp_rating, axis=1)
    
    # Step 14: Add TOG/CBA trends
    next_round_players = next_round_players.merge(
        trend_data[['player', 'team', 'TOG_Trend', 'CBA_Trend', 'Role_Status']],
        on=['player', 'team'],
        how='left'
    )
    
    # Step 15: Fill any missing values with placeholders
    next_round_players['TOG_Trend'] = next_round_players['TOG_Trend'].fillna('⚠️ Unknown')
    next_round_players['CBA_Trend'] = next_round_players['CBA_Trend'].fillna('⚠️ Unknown')
    next_round_players['Role_Status'] = next_round_players['Role_Status'].fillna('⚠️ Unknown')
    
    # Step 16: Clean up and select final columns for display
    result_df = next_round_players[['player', 'team', 'opponent', 'position', 'travel_fatigue', 
                                    'weather', 'dvp', 'TOG_Trend', 'CBA_Trend', 'Role_Status']].copy()
    
    # Combine TOG/CBA trend into one column
    result_df['tog_cba_trend'] = result_df.apply(
        lambda x: f"TOG: {x['TOG_Trend']} | CBA: {x['CBA_Trend']}", axis=1
    )
    
    # Final columns for display
    display_df = result_df[['player', 'team', 'opponent', 'position', 
                            'travel_fatigue', 'weather', 'dvp', 'tog_cba_trend', 'Role_Status']]
    
    # Rename columns for display
    display_df.columns = ['Player', 'Team', 'Opponent', 'Position', 
                          'Travel Fatigue', 'Weather', 'DvP', 'TOG/CBA Trend', 'Role Status']
    
    # Sort by team and player
    display_df = display_df.sort_values(['Team', 'Player'])
    
    return display_df

# ===== DASH APP =====
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define the layout
app.layout = dbc.Container([
    html.H1("AFL Player Dashboard - Next Round", className="text-center my-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Filter by Team:"),
                dcc.Dropdown(
                    id="team-filter",
                    placeholder="Select a team...",
                    clearable=True
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
                            html.Span("⚠️ Strong Unders Edge", className="badge bg-warning")
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
                        html.H4("TOG/CBA Trend", className="text-center"),
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
            {'if': {'column_id': 'Weather', 'filter_query': '{Weather} contains "Strong Unders"'},
             'backgroundColor': '#fff3cd', 'color': 'black'},
            
            # DvP
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Neutral"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Slight"'},
             'backgroundColor': '#fff3cd', 'color': 'black'},
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Moderate"'},
             'backgroundColor': '#ffe066', 'color': 'black'},
            {'if': {'column_id': 'DvP', 'filter_query': '{DvP} contains "Strong"'},
             'backgroundColor': '#f8d7da', 'color': 'black'},
            
            # Role Status
            {'if': {'column_id': 'Role Status', 'filter_query': '{Role Status} contains "STABLE"'},
             'backgroundColor': '#d4edda', 'color': 'black'},
            {'if': {'column_id': 'Role Status', 'filter_query': '{Role Status} contains "UNSTABLE"'},
             'backgroundColor': '#fff3cd', 'color': 'black'},
            {'if': {'column_id': 'Role Status', 'filter_query': '{Role Status} contains "LOW USAGE"'},
             'backgroundColor': '#f8d7da', 'color': 'black'},
        ],
        page_size=20,
        sort_action='native',
        filter_action='native',
    ),
], fluid=True)

# Define callback to update table data
@app.callback(
    [Output('player-table', 'data'),
     Output('player-table', 'columns'),
     Output('team-filter', 'options'),
     Output('loading-message', 'children')],
    [Input('team-filter', 'value'),
     Input('position-filter', 'value')]
)
def update_table(team, position):
    try:
        # Process data
        df = process_data_for_dashboard()
        
        # Apply filters
        if team:
            df = df[df['Team'] == team]
        if position:
            df = df[df['Position'] == position]
        
        # Create team options for dropdown
        team_options = [{'label': t, 'value': t} for t in sorted(df['Team'].unique())]
        
        # Define columns
        columns = [{"name": i, "id": i} for i in df.columns]
        
        return df.to_dict('records'), columns, team_options, ""
    
    except Exception as e:
        print(f"Error updating table: {e}")
        return [], [], [], f"Error loading data: {str(e)}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)