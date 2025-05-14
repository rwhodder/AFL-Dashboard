def apply_sensitivity(rain, wind, humidity):
    marks = goals = tackles = 0

    # --- Rain ---
    if rain >= 2:
        marks -= 40
        goals -= 20
        tackles += 25
    elif rain >= 1:
        marks -= 25
        goals -= 15
        tackles += 15

    # --- Wind ---
    if wind >= 7:
        goals -= 15
    elif wind >= 4:
        goals -= 10

    # --- Humidity ---
    if humidity >= 80:
        marks -= 10
        tackles += 10

    rating = "✅ Good Unders" if (marks <= -30 or tackles >= 20 or goals <= -20) else "❌ Skip"

    return {
        "marks_adj": f"{marks}%",
        "goals_adj": f"{goals}%",
        "tackles_adj": f"{tackles}%",
        "rating": rating
    }
