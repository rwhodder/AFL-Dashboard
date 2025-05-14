import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

# Load and preprocess CSV
df = pd.read_csv('afl_player_stats.csv', skiprows=3)
df.columns = df.columns.str.lower()
df = df[['player', 'round', 'team', 'tog', 'cbas']]

# Only keep last 5 rounds for trend analysis
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

trend.columns = ['Player', 'Team', 'TOG_avg', 'TOG_std', 'CBA_avg', 'CBA_std']

# Merge in slope trend indicators
trend = trend.merge(slopes[['player', 'team', 'TOG_Trend', 'CBA_Trend']],
                    left_on=['Player', 'Team'],
                    right_on=['player', 'team'],
                    how='left').drop(columns=['player', 'team'])

# Role stability classification
def flag_risk(row):
    if row['CBA_std'] > 5 or row['TOG_std'] > 6:
        return "BE CAUTIOUS"
    elif row['CBA_avg'] < 5:
        return "BEST FOR UNDERS"
    else:
        return "TARGET"

trend['Role_Status'] = trend.apply(flag_risk, axis=1)

# Add icons
status_icons = {
    "TARGET": "🎯 Target (STABLE)",
    "BE CAUTIOUS": "⚠️ Caution (UNSTABLE)",
    "BEST FOR UNDERS": "📉 Unders (LOW USAGE)"
}
trend['Role_Status_Icon'] = trend['Role_Status'].map(status_icons)

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "TOG & CBA Dashboard"

team_options = [{'label': team, 'value': team} for team in sorted(trend['Team'].unique())]

app.layout = html.Div([
    html.H2("TOG & CBA Trends - Last 5 Rounds"),

    html.Div([
        html.Label("Filter by Teams:"),
        dcc.Dropdown(
            id='team-filter',
            options=team_options,
            multi=True,
            placeholder="Select one or more teams..."
        ),
    ], style={'width': '50%', 'margin-bottom': '20px'}),

    html.Div([
        html.H4("📋 Player Role Table"),
        html.Button("ℹ️", id="open-modal", n_clicks=0, style={'fontSize': '20px', 'marginLeft': '10px'}),
    ]),

    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Role Status Guide")),
            dbc.ModalBody([
                html.Pre("""🟩 STABLE — Consistent Role, Reliable Projections
What it means:
The player's TOG and CBA have stayed consistent over recent games.

🟨 UNSTABLE — Volatile Role, Unpredictable Output
What it means:
The player's TOG or CBA is jumping around — maybe subbed, injured, or role-shifted.

🟥 LOW USAGE — Barely Involved, Low Stat Floor
What it means:
The player is on-field (decent TOG), but not part of key plays (low CBA or usage).""",
                          style={"whiteSpace": "pre-wrap"})
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
            ),
        ],
        id="modal",
        is_open=False,
        size="lg",
        scrollable=True,
    ),

    dash_table.DataTable(
        id='role-table',
        columns=[
            {"name": "Player", "id": "Player"},
            {"name": "Team", "id": "Team"},
            {"name": "Avg TOG (%)", "id": "TOG_avg", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Avg CBA", "id": "CBA_avg", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "TOG std", "id": "TOG_std", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "CBA std", "id": "CBA_std", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "TOG Trend", "id": "TOG_Trend"},
            {"name": "CBA Trend", "id": "CBA_Trend"},
            {"name": "Status", "id": "Role_Status_Icon"}
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
        sort_action="native",
        filter_action="native",
        page_size=25,
    ),

    html.Hr(),
    dcc.Graph(id='tog-cba-plot')
])

@app.callback(
    Output("modal", "is_open"),
    [Input("open-modal", "n_clicks"), Input("close-modal", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

@app.callback(
    Output('role-table', 'data'),
    Output('tog-cba-plot', 'figure'),
    Input('team-filter', 'value')
)
def update_outputs(selected_teams):
    if not selected_teams:
        filtered = trend
    else:
        filtered = trend[trend['Team'].isin(selected_teams)]

    fig = px.scatter(
        filtered,
        x='TOG_avg',
        y='CBA_avg',
        color='Role_Status',
        hover_data=['Player', 'Team', 'TOG_std', 'CBA_std', 'TOG_Trend', 'CBA_Trend'],
        title='Player Role Trends',
        labels={'TOG_avg': 'Avg TOG (%)', 'CBA_avg': 'Avg CBA'}
    )
    fig.update_layout(height=600)

    return filtered.to_dict('records'), fig

if __name__ == '__main__':
    app.run(debug=True)
