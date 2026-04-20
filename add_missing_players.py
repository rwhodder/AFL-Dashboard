"""
add_missing_players.py
──────────────────────
Adds the 145 players found in the Bet Log but missing from player_positions.csv.
Run once, then re-run update_positions_in_sheet.py to push corrections to Sheets.
"""
import pandas as pd

PLAYER_POSITIONS_FILE = "player_positions.csv"

# (player, team, position, notes)
# REVIEW = needs manual check before trusting
MISSING = [
    # ADE
    ("Brandon Zerk-Thatcher", "ADE", "KeyD",   ""),
    ("Brodie Smith",           "ADE", "GenD",   ""),
    ("Harry Rowston",          "ADE", "SmF",    "REVIEW"),
    ("Jake Soligo",            "ADE", "FwdMid", ""),
    ("Max Michalanney",        "ADE", "KeyD",   ""),
    ("Mitchell Hinge",         "ADE", "GenD",   ""),
    ("Reilly O'Brien",         "ADE", "Ruck",   ""),
    # BRL
    ("Archer May",             "BRL", "KeyD",   ""),
    ("Brandon Starcevich",     "BRL", "GenD",   ""),
    ("Callum Ah Chee",         "BRL", "MedF",   ""),
    ("Eric Hipwood",           "BRL", "KeyF",   ""),
    ("Jed Walter",             "BRL", "KeyF",   ""),
    ("Max Kondogiannis",       "BRL", "MedF",   ""),
    ("Will Setterfield",       "BRL", "InsM",   ""),
    # CAR
    ("Adam Cerra",             "CAR", "InsM",   ""),
    ("George Hewett",          "CAR", "InsM",   ""),
    ("Jack Buller",            "CAR", "Ruck",   ""),
    ("Jesse Motlop",           "CAR", "SmF",    ""),
    ("Luke Trainor",           "CAR", "KeyD",   ""),
    ("Matthew Owies",          "CAR", "SmF",    ""),
    ("Sam Docherty",           "CAR", "GenD",   ""),
    ("Tim Kelly",              "CAR", "InsM",   ""),
    # COL
    ("Angus Hastie",           "COL", "Wing",   ""),
    ("Darcy Moore",            "COL", "KeyD",   ""),
    ("Jeremy Howe",            "COL", "KeyD",   ""),
    ("Lachie Shultz",          "COL", "SmF",    "Name variant of Lachie Schultz"),
    ("Mark Keane",             "COL", "KeyD",   ""),
    ("Scott Pendlebury",       "COL", "InsM",   ""),
    ("Will Hoskin-Elliott",    "COL", "SmF",    ""),
    # ESS
    ("Archie Perkins",         "ESS", "FwdMid", ""),
    ("Ben Hobbs",              "ESS", "InsM",   ""),
    ("Jordan Ridley",          "ESS", "KeyD",   ""),
    ("Luke Cleary",            "ESS", "GenD",   ""),
    ("Matt Guelfi",            "ESS", "SmF",    ""),
    ("Nic Martin",             "ESS", "InsM",   ""),
    # FRE
    ("Bailey Banfield",        "FRE", "MedF",   ""),
    ("Brennan Cox",            "FRE", "KeyD",   ""),
    ("Liam Henry",             "FRE", "Wing",   ""),
    ("Luamon Lual",            "FRE", "InsM",   ""),
    ("Mani Liddy",             "FRE", "InsM",   "REVIEW"),
    ("Nathan O'Driscoll",      "FRE", "InsM",   ""),
    ("Sean Darcy",             "FRE", "Ruck",   ""),
    # GCS
    ("Ben Long",               "GCS", "SmF",    ""),
    ("Calsher Dear",           "GCS", "InsM",   "REVIEW"),
    ("David Swallow",          "GCS", "InsM",   ""),
    ("Jack Bowes",             "GCS", "GenD",   ""),
    ("Jacob Newton",           "GCS", "InsM",   "REVIEW"),
    ("Jeremy Sharp",           "GCS", "InsM",   "REVIEW"),
    ("Jhye Clark",             "GCS", "InsM",   ""),
    ("Matt Rowell",            "GCS", "InsM",   ""),
    ("Milan Murdock",          "GCS", "GenD",   ""),
    ("Steely Green",           "GCS", "InsM",   "REVIEW"),
    ("Xavier Lindsay",         "GCS", "SmF",    "REVIEW"),
    # GEE
    ("Gryan Miers",            "GEE", "SmF",    ""),
    ("Lachie Fogarty",         "GEE", "SmF",    ""),
    ("Rhys Stanley",           "GEE", "Ruck",   ""),
    ("Riley Hardeman",         "GEE", "GenD",   "REVIEW"),
    ("Sandy Brock",            "GEE", "InsM",   "REVIEW"),
    ("Tyson Stengle",          "GEE", "SmF",    ""),
    # GWS
    ("Aaron Cadman",           "GWS", "KeyF",   "Tall forward developing as KeyF"),
    ("Alixzander Tauru",       "GWS", "InsM",   ""),
    ("Clay Hall",              "GWS", "InsM",   ""),
    ("Darcy Jones",            "GWS", "InsM",   ""),
    ("Elijah Hewett",          "GWS", "InsM",   ""),
    ("Harry Himmelberg",       "GWS", "KeyD",   "Converted to key defender"),
    ("Harry O'Farrell",        "GWS", "KeyD",   ""),
    ("James Peatling",         "GWS", "InsM",   ""),
    ("Josh Kelly",             "GWS", "InsM",   ""),
    ("Kieran Briggs",          "GWS", "Ruck",   ""),
    ("Nick Haynes",            "GWS", "KeyD",   ""),
    ("Riley Garcia",           "GWS", "InsM",   "REVIEW"),
    ("Sam Taylor",             "GWS", "KeyD",   ""),
    ("Tom Brown",              "GWS", "InsM",   ""),
    ("Tom Green",              "GWS", "InsM",   ""),
    # HAW
    ("Conor Nash",             "HAW", "InsM",   ""),
    ("Hunter Clark",           "HAW", "InsM",   ""),
    ("Jack Scrimshaw",         "HAW", "KeyD",   ""),
    ("Jaeger O'Meara",         "HAW", "InsM",   ""),
    ("James Worpel",           "HAW", "InsM",   ""),
    ("Latrelle Pickett",       "HAW", "SmF",    ""),
    ("Michael Frederick",      "HAW", "SmF",    ""),
    ("Tom Mitchell",           "HAW", "InsM",   ""),
    ("Tyler Brockman",         "HAW", "SmF",    ""),
    # MEL
    ("Angus Clarke",           "MEL", "Ruck",   ""),
    ("Charlie Spargo",         "MEL", "SmF",    ""),
    ("Jack Viney",             "MEL", "InsM",   ""),
    ("Jake Bowey",             "MEL", "GenD",   ""),
    ("Jake Melksham",          "MEL", "MedF",   ""),
    ("Jayden Hunt",            "MEL", "Wing",   ""),
    ("Steven May",             "MEL", "KeyD",   ""),
    ("Toby Bedford",           "MEL", "SmF",    ""),
    ("Tom McDonald",           "MEL", "KeyF",   ""),
    # NTH
    ("Caiden Cleary",          "NTH", "GenD",   ""),
    ("Colby McKercher",        "NTH", "InsM",   ""),
    ("Cooper Harvey",          "NTH", "InsM",   ""),
    ("George Wardlaw",         "NTH", "InsM",   ""),
    ("Harry Sharp",            "NTH", "InsM",   "REVIEW"),
    ("Jed McEntee",            "NTH", "MedF",   "Tall-ish forward but not classic KeyF"),
    ("Luke McDonald",          "NTH", "KeyD",   ""),
    ("Michael Sellwood",       "NTH", "MedF",   ""),
    ("Todd Goldstein",         "NTH", "Ruck",   ""),
    ("Tyrell Dewar",           "NTH", "InsM",   ""),
    ("Will Phillips",          "NTH", "InsM",   ""),
    ("Zeke Uwland",            "NTH", "InsM",   ""),
    # PTA
    ("Angus Sheldrick",        "PTA", "InsM",   ""),
    ("Lachlan Sullivan",       "PTA", "InsM",   ""),
    ("Ollie Wines",            "PTA", "InsM",   ""),
    ("Orazio Fantasia",        "PTA", "SmF",    ""),
    ("Paul Curtin",            "PTA", "InsM",   "REVIEW: verify not same as Paul Curtis NTH"),
    ("Sam Powell-Pepper",      "PTA", "InsM",   ""),
    ("Travis Boak",            "PTA", "InsM",   ""),
    # RIC
    ("Kamdyn McIntosh",        "RIC", "GenD",   ""),
    ("Ryan Maric",             "RIC", "Ruck",   "REVIEW"),
    ("Taj Hotton",             "RIC", "SmF",    ""),
    # STK
    ("Dan Butler",             "STK", "SmF",    ""),
    ("Jimmy Webster",          "STK", "KeyD",   ""),
    ("Ryan Byrnes",            "STK", "SmF",    ""),
    # SYD
    ("Braeden Campbell",       "SYD", "InsM",   ""),
    ("Harry Cunningham",       "SYD", "GenD",   ""),
    ("Hayden McLean",          "SYD", "KeyF",   ""),
    ("Leek Aleer",             "SYD", "InsM",   ""),
    # WBD
    ("Jack Macrae",            "WBD", "InsM",   ""),
    ("Jack Watkins",           "WBD", "SmF",    "REVIEW"),
    ("Jason Johannisen",       "WBD", "Wing",   ""),
    ("Joel Hamling",           "WBD", "KeyD",   ""),
    ("Lachlan McNeil",         "WBD", "SmF",    ""),
    ("Laitham Vandermeer",     "WBD", "Wing",   ""),
    ("Matthew Carroll",        "WBD", "SmF",    ""),
    # WCE
    ("Bailey J. Williams",     "WCE", "Ruck",   ""),
    ("Blake Acres",            "WCE", "FwdMid", ""),
    ("Daniel Curtin",          "WCE", "KeyD",   ""),
    ("Hugh Jackson",           "WCE", "Ruck",   ""),
    ("Jack Darling",           "WCE", "KeyF",   ""),
    ("Jack Williams",          "WCE", "KeyD",   ""),
    ("Tom Cole",               "WCE", "Wing",   ""),
    ("Willie Rioli",           "WCE", "SmF",    ""),
    ("Zane Duursma",           "WCE", "SmF",    ""),
    # Unconfirmed team / need review
    ("Alex Davies",            "???", "GenD",   "REVIEW: confirm team + position"),
    ("Braeden Campbell",       "SYD", "InsM",   ""),
    ("Jade Gresham",           "???", "FwdMid", "REVIEW: confirm current team"),
    ("Josh Sinn",              "PTA", "SmF",    ""),
    ("Lachlan Cowan",          "???", "GenD",   "REVIEW: confirm team + position"),
    ("Orazio Fantasia",        "PTA", "SmF",    ""),
    ("Sam Marshall",           "???", "InsM",   "REVIEW: confirm team + position"),
    ("Thomas Sims",            "???", "InsM",   "REVIEW: confirm team + position"),
]

df = pd.read_csv(PLAYER_POSITIONS_FILE)
existing = set(df["player"].str.lower().str.strip())

new_rows = []
skipped  = []

for player, team, position, notes in MISSING:
    key = player.lower().strip()
    if key in existing:
        skipped.append(player)
        continue
    new_rows.append({
        "player":          player,
        "team":            team,
        "position":        position,
        "most_common_pos": "",
        "notes":           notes,
    })
    existing.add(key)   # prevent duplicates within this list

if new_rows:
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(PLAYER_POSITIONS_FILE, index=False)
    print(f"✅  Added {len(new_rows)} player(s) to {PLAYER_POSITIONS_FILE}")
else:
    print("ℹ️  No new players to add — all already present.")

if skipped:
    print(f"\nℹ️  Skipped {len(skipped)} already-present player(s):")
    for p in skipped:
        print(f"   • {p}")

review = [r for r in new_rows if "REVIEW" in r["notes"]]
if review:
    print(f"\n⚠️  {len(review)} player(s) need manual position review:")
    for r in review:
        print(f"   • {r['player']} ({r['team']}) → {r['position']}  [{r['notes']}]")
