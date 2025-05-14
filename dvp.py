import pandas as pd
import dash
from dash import html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc

# === Load Data ===
df = pd.read_csv("afl_player_stats.csv", skiprows=3)

# === Handle missing namedPosition by filling INT with previous known ===
df['namedPosition'] = df.groupby('player')['namedPosition'] \
    .transform(lambda x: x.mask(x == "INT").ffill()).infer_objects(copy=False)

df = df[df["namedPosition"] != "INT"]

# === Position Groups ===
position_map = {
    "KeyF": ["FF", "CHF"],
    "GenF": ["HFFR", "HFFL", "FPL", "FPR"],
    "Ruck": ["RK"],
    "InsM": ["C", "RR", "R"],
    "Wing": ["WL", "WR"],
    "GenD": ["HBFL", "HBFR", "BPL", "BPR"],
    "KeyD": ["CHB", "FB"],
}
position_lookup = {pos: group for group, pos_list in position_map.items() for pos in pos_list}
df["positionGroup"] = df["namedPosition"].map(position_lookup)

# === Add Calculated Stats ===
df["disposals"] = df["kicks"] + df["handballs"]

# === Clean Team Column ===
df["opponent"] = df["opponent"].str.strip().str.upper()

# === Calculate League Averages ===
league_averages = df.groupby("positionGroup")[["disposals", "marks", "tackles"]].mean().to_dict()

# === Build DvP Table ===
dvp_outputs = {}
min_samples = 5  # %

stat_settings = {
    "Disposals": {"col": "disposals", "threshold": -1.5},
    "Marks": {"col": "marks", "threshold": -0.5},
    "Tackles": {"col": "tackles", "threshold": -0.5},
}

for stat, settings in stat_settings.items():
    stat_col = settings["col"]
    threshold = settings["threshold"]
    df_stat = df[["opponent", "positionGroup", stat_col]].copy()
    df_stat["count"] = 1

    agg = df_stat.groupby(["opponent", "positionGroup"]).agg(
        avg_stat=(stat_col, "mean"),
        count=("count", "sum")
    ).reset_index()

    # Compute DvP
    agg["league_avg"] = agg["positionGroup"].map(league_averages[stat_col])
    agg["dvp"] = agg["avg_stat"] - agg["league_avg"]

    # Compute sample %
    total_counts = df.groupby("opponent").size().to_dict()
    agg["sample_pct"] = agg["opponent"].map(total_counts)
    agg["sample_pct"] = (agg["count"] / agg["sample_pct"] * 100).round(0)

    # Filter for strong unders (No-Go Zones)
    filtered = agg[
        (agg["dvp"] <= threshold) & (agg["sample_pct"] >= min_samples)
    ].copy()

    filtered = filtered.rename(
        columns={
            "opponent": "Team",
            "positionGroup": "Position",
            "dvp": "Avg DvP",
            "sample_pct": "Sample %",
        }
    )[["Team", "Position", "Avg DvP", "Sample %"]]

    filtered["Avg DvP"] = filtered["Avg DvP"].round(2)
    filtered["Stat"] = stat

    dvp_outputs[stat] = filtered.sort_values(by="Avg DvP")

# === Dash App ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# === Gradient Styling Function ===
def get_dvp_style(stat):
    if stat == "Disposals":
        return [
            {"if": {"filter_query": "{Avg DvP} <= -4.0", "column_id": "Avg DvP"}, "backgroundColor": "#8B0000", "color": "white", "fontWeight": "bold"},
            {"if": {"filter_query": "{Avg DvP} <= -1.5 && {Avg DvP} > -4.0", "column_id": "Avg DvP"}, "backgroundColor": "#B22222", "color": "white"},
            {"if": {"filter_query": "{Avg DvP} <= -1.0 && {Avg DvP} > -3.0", "column_id": "Avg DvP"}, "backgroundColor": "#DC143C", "color": "white"},
            {"if": {"filter_query": "{Avg DvP} <= -0.5 && {Avg DvP} > -2.0", "column_id": "Avg DvP"}, "backgroundColor": "#F08080", "color": "black"},
        ]
    else:  # Marks and Tackles
        return [
            {"if": {"filter_query": "{Avg DvP} <= -1.5", "column_id": "Avg DvP"}, "backgroundColor": "#8B0000", "color": "white"},
            {"if": {"filter_query": "{Avg DvP} <= -1.0 && {Avg DvP} > -1.5", "column_id": "Avg DvP"}, "backgroundColor": "#B22222", "color": "white"},
            {"if": {"filter_query": "{Avg DvP} <= -0.75 && {Avg DvP} > -1.0", "column_id": "Avg DvP"}, "backgroundColor": "#DC143C", "color": "white"},
            {"if": {"filter_query": "{Avg DvP} <= -0.5 && {Avg DvP} > -0.75", "column_id": "Avg DvP"}, "backgroundColor": "#F08080", "color": "black"},
        ]

# === Layout ===
app.layout = dbc.Container(
    [
        html.H2("AFL No-Go Zones – DvP Unders Filters", className="text-center my-4 fw-bold"),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="team-filter",
                        options=[{"label": team, "value": team} for team in sorted(df["opponent"].unique())],
                        placeholder="Filter by Team...",
                        clearable=True,
                    ),
                    width=4,
                ),
            ],
            className="mb-4",
        ),
        html.Div(id="dvp-tables"),
    ],
    fluid=True,
)

# === Callback ===
@app.callback(
    Output("dvp-tables", "children"),
    Input("team-filter", "value"),
)
def update_tables(team_filter):
    tables = []
    for stat in ["Disposals", "Marks", "Tackles"]:
        table_df = dvp_outputs[stat]
        if team_filter:
            table_df = table_df[table_df["Team"] == team_filter]

        tables.append(
            dbc.Card(
                [
                    dbc.CardHeader(html.H5(f"{stat} – No-Go Zones", className="mb-0")),
                    dbc.CardBody(
                        dash_table.DataTable(
                            data=table_df.to_dict("records"),
                            columns=[{"name": col, "id": col} for col in table_df.columns],
                            style_table={"overflowX": "auto"},
                            style_header={
                                "backgroundColor": "#343a40",
                                "color": "white",
                                "fontWeight": "bold",
                            },
                            style_cell={
                                "textAlign": "center",
                                "padding": "8px",
                                "fontFamily": "Arial",
                                "fontSize": "14px",
                            },
                            style_data_conditional=get_dvp_style(stat),
                        )
                    )
                ],
                className="mb-4 shadow-sm"
            )
        )
    return tables

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True)
