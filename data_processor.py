import pandas as pd

# Map namedPosition to role groups
POSITION_MAP = {
    "KeyF": ["FF", "CHF"],
    "GenF": ["HFFR", "HFFL", "FPL", "FPR"],
    "Ruck": ["RK"],
    "InsM": ["C", "RR", "R"],
    "Wing": ["WL", "WR"],
    "GenD": ["HBFL", "HBFR", "BPL", "BPR"],
    "KeyD": ["CHB", "FB"]
}

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, skiprows=3)
    df["disposals"] = df["kicks"] + df["handballs"]

    df["assignedPosition"] = df["namedPosition"]
    int_players = df["namedPosition"] == "INT"
    
    # Backfill INT players with last known position
    df["assignedPosition"] = df.groupby("player")["assignedPosition"].transform(lambda x: x.ffill().bfill())

    
    df = df[df["assignedPosition"] != "INT"]

    # Map to broader role
    def map_position(pos):
        for role, tags in POSITION_MAP.items():
            if pos in tags:
                return role
        return None

    df["role"] = df["assignedPosition"].apply(map_position)
    df = df.dropna(subset=["role"])

    return df

def calculate_dvp(df, stat_col):
    group = df.groupby(["opponentTeam", "role"])[stat_col].agg(['mean', 'count']).reset_index()
    overall = df.groupby("role")[stat_col].mean().rename("overall_mean").reset_index()
    merged = group.merge(overall, on="role")
    merged["dvp"] = merged["mean"] - merged["overall_mean"]
    merged["pct"] = (merged["count"] / df.groupby("opponentTeam")["player"].count().reindex(merged["opponentTeam"].unique()).values) * 100
    filtered = merged[(merged["dvp"] <= -0.1) & (merged["pct"] >= 0.1)]

    return filtered[["opponentTeam", "role", "dvp", "pct"]].sort_values(["dvp"])
