import requests
from datetime import datetime, timedelta
import pytz
from collections import defaultdict

# ===== CONFIG =====
OPENWEATHER_API_KEY = "e76003c560c617b8ffb27f2dee7123f4"  # Replace with your actual API key
WEATHER_LOOKUP = {
#VIC
"MCG": (-37.8199, 144.9834),
"Marvel Stadium": (-37.8167, 144.9475),
"GMHBA Stadium": (-38.1625, 144.3555),
"Mars Stadium": (-37.5444, 143.8483),  
#SA
"Adelaide Oval": (-34.9158, 138.5965),
"Norwood Oval": (-34.5511, 144.9475),
"Barossa Park": (-34.6135, 138.9616),
#WA
"Optus Stadium": (-31.9505, 115.8890),
"Hands Oval Stadium": (-33.2046, 115.3834),
#NSW
"SCG": (-33.8914, 151.2245),
"ENGIE Stadium": (-33.8430, 151.0677),
#"Giants Stadium": (-33.8474, 151.0632), same as ENGIE
#ACT
"Manuka Oval": (-35.1950, 149.0850),
#QLD
"Gabba": (-27.4858, 153.0381),
"Heritage Bank Stadium": (-28.0055, 153.4153),
"People First Stadium": (-28.0023, 153.2220),
#TAS 
"UTAS Stadium": (-41.4298, 147.1571),
"Blundstone Arena": (-42.8881, 147.3766),
"University of Tasmania Stadium": (-41.4265, 147.1375),
#NT
"TIO Traeger Park": (-23.7090, 133.8751),
"TIO Stadium": (-12.2347, 130.5251)
}

# ===== FIXTURE SCRAPER =====
def scrape_next_round_fixture():
    try:
        url = "https://fixturedownload.com/feed/json/afl-2025"
        response = requests.get(url, verify=False)
        response.raise_for_status()
        data = response.json()

        melbourne_tz = pytz.timezone('Australia/Melbourne')
        now = datetime.now(melbourne_tz)

        rounds = defaultdict(list)

        for match in data:
            try:
                utc_dt = datetime.strptime(match['DateUtc'], "%Y-%m-%d %H:%M:%SZ")
                utc_dt = pytz.utc.localize(utc_dt)
                local_dt = utc_dt.astimezone(melbourne_tz)

                if local_dt > now:
                    rounds[match['RoundNumber']].append({
                        'match': f"{match['HomeTeam']} vs {match['AwayTeam']}",
                        'datetime': local_dt,
                        'stadium': match.get('Location', 'Unknown')
                    })

            except Exception as e:
                print(f"Skipping match due to error: {e}")
                continue

        if not rounds:
            return []

        next_round = sorted(rounds.items(), key=lambda x: min(m['datetime'] for m in x[1]))[0]
        return next_round[1]

    except requests.RequestException as e:
        print(f"Error fetching fixture data: {e}")
        return []

# ===== WEATHER FETCHING =====
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

# ===== STAT SENSITIVITY LOGIC =====
def apply_stat_sensitivity(weather):
    rain = weather.get('rain', 0)
    wind = weather.get('wind', 0)
    humidity = weather.get('humidity', 0)

    rating = "✅ Neutral"
    if rain > 2 or wind > 6 or humidity > 85:
        rating = "⚠️ Strong Unders Edge"

    marks = goals = tackles = disposals = 0
    if rain > 2:
        marks -= 25
        goals -= 20
        tackles += 15
        disposals -= 15
    if wind > 6:
        goals -= 10
        disposals -= 10
    if humidity > 85:
        marks -= 10
        tackles += 10
        disposals -= 8

    return {
        "Rain": categorize_rain(rain),
        "Wind": categorize_wind(wind),
        "Humid": categorize_humidity(humidity),
        "Marks": f"{marks}%",
        "Goals": f"{goals}%",
        "Tackles": f"{'+' if tackles >= 0 else ''}{tackles}%",
        "Disposals": f"{disposals}%",
        "Rating": rating
    }

def categorize_rain(mm):
    if mm == 0: return "Low"
    elif mm <= 2: return "Med"
    return "High"

def categorize_wind(kph):
    if kph <= 10: return "Low"
    elif kph <= 20: return "Med"
    return "High"

def categorize_humidity(h):
    if h <= 60: return "Low"
    elif h <= 85: return "Med"
    return "High"

# ===== MAIN PROCESS =====
def main():
    print("\U0001F504 Fetching AFL fixture and weather forecasts...\n")
    ROUND_FIXTURE = scrape_next_round_fixture()
    if not ROUND_FIXTURE:
        print("❌ Could not load fixture. Please check your internet connection or the fixture source.")
        return

    weather_data = {}
    for match in ROUND_FIXTURE:
        stadium = match["stadium"]
        latlon = WEATHER_LOOKUP.get(stadium)

        if not latlon:
            for key in WEATHER_LOOKUP:
                if key.lower() in stadium.lower():
                    latlon = WEATHER_LOOKUP[key]
                    break
            if not latlon:
                print(f"⚠️ Unknown venue: {stadium}")
                continue

        forecast_list = get_forecast(*latlon)
        weather = extract_weather_for_datetime(forecast_list, match["datetime"])

        match_key = match["match"].lower().strip()
        if weather:
            weather_data[match_key] = weather
        else:
            print(f"⚠️ No forecast found for {match['match']} at {stadium}")

    print(f"\n{'Match':<35} | Rain | Wind | Humid | Marks | Goals | Tackles | Disposals | Rating")
    print("-" * 105)
    for game in ROUND_FIXTURE:
        match_key = game["match"].lower().strip()
        forecast = weather_data.get(match_key)
        if forecast:
            effects = apply_stat_sensitivity(forecast)
            print(f"{game['match']:<35} | {effects['Rain']:>4} | {effects['Wind']:>4} | {effects['Humid']:>5} | "
                  f"{effects['Marks']:>6} | {effects['Goals']:>6} | {effects['Tackles']:>8} | {effects['Disposals']:>9} | {effects['Rating']}")
        else:
            print(f"{game['match']:<35} | No weather data available.")

if __name__ == "__main__":
    main()
