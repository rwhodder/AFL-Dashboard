import requests
import urllib3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
AFL_COMP_ID = "ad4c78ec-e39d-45ee-8cec-ff5d485a3205"
FIXTURES_URL = f"https://api.dabble.com.au/competitions/{AFL_COMP_ID}/sport-fixtures"
HEADERS = {
    "User-Agent": "Dabble/1000030204 CFNetwork/3826.500.131 Darwin/24.5.0",
    "Accept": "application/json"
}

def get_afl_fixtures() -> List[Dict]:
    """Get all AFL fixtures for the current round"""
    response = requests.get(FIXTURES_URL, headers=HEADERS, verify=False)

    if response.status_code != 200:
        return []

    try:
        data = response.json()["data"]

        fixtures = []
        for match in data:
            match_name = match.get("name", "Unknown match")
            start_time = match.get("advertisedStart", "Unknown time")
            fixture_id = match.get("id", "No ID")

            fixtures.append({
                "name": match_name,
                "start_time": start_time,
                "fixture_id": fixture_id
            })

        return fixtures

    except Exception as e:
        return []

def get_pickem_lines(fixture_id: str) -> Optional[Dict]:
    """Get pickem lines for a specific fixture"""
    pickem_url = f"https://api.dabble.com.au/sportfixtures/details/{fixture_id}?filter=dfs-enabled"

    try:
        response = requests.get(pickem_url, headers=HEADERS, verify=False)

        if response.status_code != 200:
            return None

        data = response.json()
        fixture_data = data.get("data", {})
        markets = fixture_data.get("markets", [])

        pickem_markets = []
        for market in markets:
            market_type = market.get("resultingType", "")
            market_name = market.get("name", "")
            if ("pickem" in market_type.lower() or
                any(stat in market_name.lower() for stat in ["disposal", "mark", "tackle", "goal", "kick", "handball"])):
                pickem_markets.append(market)

        return {
            "fixture_id": fixture_id,
            "markets": pickem_markets
        }

    except Exception:
        return None

def extract_player_lines(pickem_data: Dict) -> List[Dict]:
    """Extract individual player lines from pickem data"""
    all_lines = []

    if not pickem_data or "markets" not in pickem_data:
        return all_lines

    fixture_id = pickem_data["fixture_id"]

    for market in pickem_data["markets"]:
        market_name = market.get("name", "Unknown Market")
        market_type = market.get("resultingType", "Unknown Type")

        if market_name and market_name != "Unknown Market":
            parts = market_name.split()

            if len(parts) >= 3:
                try:
                    line_value = float(parts[-1])
                    stat_type = parts[-2].lower()
                    player_name = " ".join(parts[:-2])

                    if market_type.startswith("pickem_"):
                        stat_name = market_type.replace("pickem_", "").title()
                    else:
                        stat_mapping = {
                            "disposals": "Disposals",
                            "marks": "Marks",
                            "tackles": "Tackles",
                            "goals": "Goals",
                            "kicks": "Kicks",
                            "handballs": "Handballs",
                            "fantasy": "Fantasy",
                            "supercoach": "SuperCoach"
                        }
                        stat_name = stat_mapping.get(stat_type, stat_type.title())

                    over_odds = None
                    under_odds = None
                    if "overPrice" in market:
                        over_odds = market.get("overPrice", {}).get("decimal")
                    if "underPrice" in market:
                        under_odds = market.get("underPrice", {}).get("decimal")

                    all_lines.append({
                        "fixture_id": fixture_id,
                        "player": player_name,
                        "stat": stat_name,
                        "line": line_value,
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                        "market_name": market_name,
                        "market_type": market_type
                    })

                except (ValueError, IndexError):
                    pass

    return all_lines

def get_all_pickem_lines() -> List[Dict]:
    """Get all pickem lines for all fixtures this round"""
    fixtures = get_afl_fixtures()

    if not fixtures:
        return []

    all_player_lines = []

    for fixture in fixtures:
        fixture_id = fixture["fixture_id"]
        match_name = fixture["name"]

        pickem_data = get_pickem_lines(fixture_id)

        if pickem_data:
            player_lines = extract_player_lines(pickem_data)
            for line in player_lines:
                line["match"] = match_name
                line["start_time"] = fixture["start_time"]
            all_player_lines.extend(player_lines)

    return all_player_lines

def get_pickem_data_for_dashboard(stat_type='disposals') -> Dict[str, float]:
    """
    Get pickem lines for dashboard integration - only returns disposals, marks, and tackles
    Returns a dictionary mapping player names to their line values for the specified stat type
    """
    try:
        all_lines = get_all_pickem_lines()

        if not all_lines:
            return {}

        stat_mapping = {
            'disposals': 'Disposals',
            'marks': 'Marks',
            'tackles': 'Tackles'
        }

        target_stat = stat_mapping.get(stat_type.lower(), stat_type.title())
        filtered_lines = [line for line in all_lines if line['stat'] == target_stat]

        player_lines = {}
        for line in filtered_lines:
            player_name = line['player']
            line_value = line['line']
            if player_name not in player_lines:
                player_lines[player_name] = line_value
            else:
                if line_value > player_lines[player_name]:
                    player_lines[player_name] = line_value

        return player_lines

    except Exception:
        return {}

def normalize_player_name(name: str) -> str:
    """
    Normalize player names for better matching between datasets
    """
    if not name:
        return ""
    
    # Basic normalization
    normalized = name.strip()
    
    # Common name variations that might need mapping
    # You can expand this based on discrepancies you find
    name_mappings = {
        # Add specific mappings if you find mismatches
        # "Dabble Name": "Dashboard Name"
    }
    
    return name_mappings.get(normalized, normalized)

# Keep the existing functions for standalone use
def display_results(all_lines: List[Dict]):
    """Display the results in a readable format"""
    if not all_lines:
        print("❌ No pickem lines found.")
        return
    
    print(f"📊 SUMMARY: Found {len(all_lines)} total pickem lines")
    print("=" * 80)
    
    # Group by match
    matches = {}
    for line in all_lines:
        match_name = line["match"]
        if match_name not in matches:
            matches[match_name] = []
        matches[match_name].append(line)
    
    # Display by match
    for match_name, lines in matches.items():
        print(f"\n🏉 {match_name}")
        print("-" * 60)
        
        # Group by stat type
        stats = {}
        for line in lines:
            stat = line["stat"]
            if stat not in stats:
                stats[stat] = []
            stats[stat].append(line)
        
        for stat_name, stat_lines in stats.items():
            print(f"\n  📈 {stat_name} ({len(stat_lines)} players):")
            
            # Sort by line value
            stat_lines.sort(key=lambda x: float(x["line"]) if x["line"] else 0)
            
            for line in stat_lines:
                player = line["player"]
                line_val = line["line"]
                over_odds = line["over_odds"]
                under_odds = line["under_odds"]
                
                odds_str = ""
                if over_odds and under_odds:
                    odds_str = f" (Over: {over_odds}, Under: {under_odds})"
                
                print(f"    • {player}: {line_val}{odds_str}")

def save_to_csv(all_lines: List[Dict], filename: str = None):
    """Save results to CSV file"""
    if not all_lines:
        print("❌ No data to save.")
        return
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"afl_pickem_lines_{timestamp}.csv"
    
    try:
        import pandas as pd
        df = pd.DataFrame(all_lines)
        
        # Reorder columns for better readability
        column_order = ["match", "player", "stat", "line", "over_odds", "under_odds", 
                       "start_time", "fixture_id", "market_name", "market_type"]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        df.to_csv(filename, index=False)
        print(f"💾 Results saved to: {filename}")
        
    except ImportError:
        print("⚠️ pandas not available. Saving as JSON instead.")
        json_filename = filename.replace('.csv', '.json')
        with open(json_filename, 'w') as f:
            json.dump(all_lines, f, indent=2)
        print(f"💾 Results saved to: {json_filename}")

def debug_single_fixture(fixture_id: str):
    """Debug function to examine the structure of a single fixture's API response"""
    pickem_url = f"https://api.dabble.com.au/sportfixtures/details/{fixture_id}?filter=dfs-enabled"
    
    print(f"🔍 DEBUGGING fixture {fixture_id}...")
    print(f"🔗 URL: {pickem_url}")
    
    try:
        response = requests.get(pickem_url, headers=HEADERS, verify=False)
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Show top-level structure
            print(f"📋 Top-level keys: {list(data.keys())}")
            
            if "data" in data:
                data_section = data["data"]
                print(f"📋 Data section keys: {list(data_section.keys())}")
                
                if "markets" in data_section:
                    markets = data_section["markets"]
                    print(f"📈 Found {len(markets)} markets")
                    
                    # Show first few markets in detail
                    for i, market in enumerate(markets[:3]):
                        print(f"\n📊 Market {i+1}:")
                        print(f"   Name: {market.get('name', 'N/A')}")
                        print(f"   Type: {market.get('resultingType', 'N/A')}")
                        print(f"   Keys: {list(market.keys())}")
                        
                        selections = market.get("selections", [])
                        print(f"   Selections: {len(selections)}")
                        
                        if selections:
                            print(f"   First selection keys: {list(selections[0].keys())}")
                            print(f"   First selection: {selections[0]}")
                    
                    # Show all market types
                    all_types = [m.get("resultingType", "Unknown") for m in markets]
                    unique_types = list(set(all_types))
                    print(f"\n📈 All market types found: {unique_types}")
                    
                    # Show all market names
                    all_names = [m.get("name", "Unknown") for m in markets]
                    print(f"📈 All market names: {all_names}")
        
    except Exception as e:
        print(f"❌ Error in debug: {e}")

def main():
    """Main execution function"""
    # Optional: Debug a single fixture first
    # Uncomment the next 3 lines to debug the structure of one fixture
    # print("🐛 DEBUG MODE - Examining single fixture structure")
    # debug_single_fixture("15e154fd-a1fa-4af2-80d8-2af3e4186c64")  # Use your example ID
    # return
    
    # Get all pickem lines
    all_lines = get_all_pickem_lines()
    
    # Display results
    display_results(all_lines)
    
    # Save to file
    if all_lines:
        save_to_csv(all_lines)
        print(f"\n✅ Complete! Found pickem lines for {len(set(line['match'] for line in all_lines))} matches.")
    else:
        print("\n❌ No pickem lines found.")

if __name__ == "__main__":
    main()