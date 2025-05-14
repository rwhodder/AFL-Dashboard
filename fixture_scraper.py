import requests
from datetime import datetime
import pytz

def scrape_next_round_fixture():
    """
    Fetches the AFL 2025 fixture from the correct JSON feed.
    Returns a list of match dictionaries for the next upcoming round.
    """
    try:
        url = "https://fixturedownload.com/feed/json/afl-2025"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        melbourne_tz = pytz.timezone('Australia/Melbourne')
        now = datetime.now(melbourne_tz)

        # Group upcoming matches by round number
        from collections import defaultdict
        rounds = defaultdict(list)

        for match in data:
            try:
                # Use the 'DateUtc' field directly
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

        # Return the matches from the next round
        if not rounds:
            return []

        next_round = sorted(rounds.items(), key=lambda x: min(m['datetime'] for m in x[1]))[0]
        return next_round[1]

    except requests.RequestException as e:
        print(f"Error fetching fixture data: {e}")
        return []

# Test block
if __name__ == "__main__":
    fixtures = scrape_next_round_fixture()
    if fixtures:
        print(f"\nNext Round: {len(fixtures)} matches")
        for match in fixtures:
            print(f"{match['match']} - {match['datetime'].strftime('%A %d %B, %I:%M %p')} at {match['stadium']}")
    else:
        print("No upcoming fixtures found.")
