import requests
import urllib3
import json
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
    print(f"🔄 Requesting AFL fixture list...")
    response = requests.get(FIXTURES_URL, headers=HEADERS, verify=False)
    print(f"✅ Status Code: {response.status_code}")
    
    if response.status_code != 200:
        print("❌ Request failed.")
        return []
    
    try:
        data = response.json()["data"]
        print(f"📅 Found {len(data)} AFL fixtures:\n")
        
        fixtures = []
        for match in data:
            match_name = match.get("name", "Unknown match")
            start_time = match.get("advertisedStart", "Unknown time")
            fixture_id = match.get("id", "No ID")
            
            if start_time != "Unknown time":
                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                print(f"🏉 {match_name} at {dt.strftime('%Y-%m-%d %H:%M')} UTC")
            else:
                print(f"🏉 {match_name} at {start_time}")
            
            print(f"🔗 Fixture ID: {fixture_id}\n")
            
            fixtures.append({
                "name": match_name,
                "start_time": start_time,
                "fixture_id": fixture_id
            })
        
        return fixtures
        
    except Exception as e:
        print("❌ Error parsing fixtures response:", e)
        print("Raw response:", response.text[:300])
        return []

def get_pickem_lines(fixture_id: str) -> Optional[Dict]:
    """Get pickem lines for a specific fixture"""
    pickem_url = f"https://api.dabble.com.au/sportfixtures/details/{fixture_id}?filter=dfs-enabled"
    
    print(f"🔄 Requesting pickem lines for fixture {fixture_id}...")
    
    try:
        response = requests.get(pickem_url, headers=HEADERS, verify=False)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   ❌ Failed to get pickem data for {fixture_id}")
            return None
        
        data = response.json()
        
        # Extract relevant pickem data
        fixture_data = data.get("data", {})
        markets = fixture_data.get("markets", [])
        
        # Filter for pickem markets - look for any market with "pickem" in the resultingType
        pickem_markets = []
        
        for market in markets:
            market_type = market.get("resultingType", "")
            market_name = market.get("name", "")
            
            # Accept any market that has "pickem" in the type OR common stat names
            if ("pickem" in market_type.lower() or 
                any(stat in market_name.lower() for stat in ["disposal", "mark", "tackle", "goal", "kick", "handball"])):
                pickem_markets.append(market)
                print(f"   📊 Found market: {market_name} (type: {market_type})")
        
        print(f"   ✅ Found {len(pickem_markets)} pickem markets")
        return {
            "fixture_id": fixture_id,
            "markets": pickem_markets
        }
        
    except Exception as e:
        print(f"   ❌ Error getting pickem lines for {fixture_id}: {e}")
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
        selections = market.get("selections", [])
        
        print(f"   🔍 Processing market: {market_name} (type: {market_type}, {len(selections)} selections)")
        
        # Since selections are empty, extract player name and line from market name
        # Format: "Player Name stat_type line_value"
        # Example: "Jack Lukosius goals 2.5" 
        
        if market_name and market_name != "Unknown Market":
            # Parse the market name to extract player, stat, and line
            parts = market_name.split()
            
            if len(parts) >= 3:
                # Last part should be the line (number)
                try:
                    line_value = float(parts[-1])
                    
                    # Second to last part should be the stat type
                    stat_type = parts[-2].lower()
                    
                    # Everything before that should be the player name
                    player_name = " ".join(parts[:-2])
                    
                    # Determine stat name from market type or parsed stat
                    if market_type.startswith("pickem_"):
                        stat_name = market_type.replace("pickem_", "").title()
                    else:
                        # Map common stat types
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
                    
                    # Extract odds from market if available (though they seem to be in selections)
                    over_odds = None
                    under_odds = None
                    
                    # Check if there are any price fields in the market itself
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
                    
                    print(f"      ✅ Extracted: {player_name} | {stat_name} | {line_value}")
                
                except (ValueError, IndexError) as e:
                    print(f"      ⚠️ Could not parse market name: {market_name} - {e}")
            else:
                print(f"      ⚠️ Market name format unexpected: {market_name}")
    
    print(f"   📈 Extracted {len(all_lines)} total player lines from all markets")
    return all_lines

def get_all_pickem_lines() -> List[Dict]:
    """Get all pickem lines for all fixtures this round"""
    print("🚀 Starting AFL Pickem Lines Scraper\n")
    
    # Step 1: Get all fixtures
    fixtures = get_afl_fixtures()
    
    if not fixtures:
        print("❌ No fixtures found. Exiting.")
        return []
    
    print(f"📊 Processing {len(fixtures)} fixtures for pickem lines...\n")
    
    # Step 2: Get pickem lines for each fixture
    all_player_lines = []
    
    for fixture in fixtures:
        fixture_id = fixture["fixture_id"]
        match_name = fixture["name"]
        
        print(f"🔍 Processing: {match_name}")
        
        # Get pickem data for this fixture
        pickem_data = get_pickem_lines(fixture_id)
        
        if pickem_data:
            # Extract individual player lines
            player_lines = extract_player_lines(pickem_data)
            
            # Add match info to each line
            for line in player_lines:
                line["match"] = match_name
                line["start_time"] = fixture["start_time"]
            
            all_player_lines.extend(player_lines)
            print(f"   📈 Extracted {len(player_lines)} player lines")
        else:
            print(f"   ⚠️ No pickem data found for {match_name}")
        
        print()  # Empty line for readability
    
    return all_player_lines

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