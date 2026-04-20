# AFL Dabble Dashboard — Agent Handover Context

_Last updated: 2026-04-17 (post Round 6 session)_

---

## Project Overview

A Dash/Plotly Python dashboard (`app.py`) for AFL unders betting on the Dabble Pick'Em platform. The user bets on players going **under** their tackle/disposal/marks lines in a Pick'Em hedge format (not singles).

**Run:** `python main.py` or `python app.py`  
**Platform:** Windows 11, Python 3.13  
**Key file:** `app.py` (~4800 lines — everything lives here)

---

## Tab Structure

```python
dbc.Tabs([
    dbc.Tab(label="📊 Performance", tab_id="tab-performance"),
    dbc.Tab(label="🎯 Multi Builder", tab_id="tab-multi"),
    dbc.Tab(label="🧠 Analysis",     tab_id="tab-analysis"),
], id="stat-tabs", active_tab="tab-performance")
```

---

## Key Data Sources

| Source | Purpose |
|---|---|
| Google Sheets (`GOOGLE_SHEET_ID`) | Bet log (Tab: "Bet Log"), Pair Log (Tab: "Pair Log") |
| `afl-2026-fixture.csv` | Fixture dates for round matching |
| `afl_player_stats.csv` | Historical player stats |
| `player_positions.csv` | Position lookup |
| Dabble scraper | Live Pick'Em lines |

**Bet Log columns:** Type, Year, Round, Player, Team, Opponent, Position, Travel Fatigue, Weather, DvP, Line, Avg vs Line, Line Consistency, Strategy, Actual, W/L  
**W/L encoding:** 1 = win (went under), -1 = loss (went over)

---

## Betting Criteria — `calculate_bet_flag()`

Tackles only. Rebuilt 2026-04-17 from 3,055+ bets across 2025+2026.

### Hard Avoids (applied in order, no exceptions)

| Filter | Threshold | Evidence |
|---|---|---|
| Line < 3 | always skip | ~50% WR — too tight for edge |
| Short Break travel | always skip | 0% WR — universal |
| Opponent GCS or GEE | always skip | ~42% WR — structural |
| Position = MedF | always skip | 43% WR — consistently over-tackles line |
| FwdMid without Mod/Strong Unders DvP | skip | 50% WR — too variable |
| Narrow ground (GMHBA/Kardinia Park) | skip **InsM/FwdMid/SmF only** | ground dimensions suppress tackle counts |
| AvL ≥ 10% | skip (post T1-bypass check) | 45% WR |
| Cons 61–80% + Moderate Unders DvP | skip | 42% WR — Dabble has this line right |

**Important:** Narrow ground filter now exempts Wing/Ruck/GenD — those positions tackle from role, not ground dimensions. This was fixed 2026-04-17 (previously Mark Blicavs at GMHBA was incorrectly excluded; he has 70% (20) historical WR).

### T1 / T2 Tiers (informational only — NOT used for hedge picks since 2026-04-17)

| Tier | Condition | Notes |
|---|---|---|
| T1 | Wing/Ruck/GenD position | Structural tacklers, 67% WR |
| T1 | AvL < -20% | Dabble line set too high, 65% WR |
| T2 | Strong Unders DvP | Monitoring — 50% WR in 2026 |
| T2 | Everything else that passes | 65% WR |

**T1/T2 are now informational context columns only.** Hedge picks are driven by the Hist WR + Confidence system (see below). The Tier column still appears in the legs table.

### Watch List (do not implement yet)

| Signal | Status | Sample |
|---|---|---|
| InsM + Moderate Unders | 52% WR (44 bets) — below 56% threshold, borderline | Re-evaluate at ~50 bets |
| NTH as opponent in 2026 | 40% WR (10 bets) — small sample | Re-evaluate at 15–20 bets |
| ADE as opponent in 2026 | 42% WR (7 bets) — too small | Re-evaluate at 15 bets |
| Ruck + Strong Unders DvP | 50% WR (10 bets) — small sample | Watch |
| Neutral DvP in 2026 | 33% WR (6 bets) — tiny sample | Watch |

### Key Constants

```python
TACKLE_BAD_OPPONENTS    = {'GCS', 'GEE'}   # SYD removed — was coincidental overlap
NARROW_GROUNDS_NO_TACKLE = {"GMHBA Stadium", "Kardinia Park"}
```

---

## Historical WR System (NEW — 2026-04-17)

The legs table now shows historical win rates fetched live from Google Sheets. This drives visual dimming AND hedge pick selection.

### Module-level cache

```python
_BET_HISTORY: dict = {'df': None}   # lazy-loaded once per process
```

### `get_leg_historical_wr(position, dvp, opponent) → str`

Returns `"67% (23)"` format. Cascade (most specific → least):
1. Position + DvP + Opponent (n ≥ 5) — e.g. "Wing vs SYD on Mod Unders"
2. Position + DvP (n ≥ 8)
3. Position only (n ≥ 5)
4. `""` — not enough data

### Confidence levels

| Label | Sample size |
|---|---|
| High | n ≥ 30 |
| Med | n ≥ 15 |
| Low | n ≥ 5 |
| — | < 5 |

### WR Range

±1 SE band. Example: 57% (58) → `"50–64%"`. Shows how wide the uncertainty is.  
Large range + Low confidence = don't act on it. Small range + High confidence = reliable.

### Inclusion threshold: **56% with Med or High confidence**

56% clears the highest breakeven across all formats (55.9% for 2-leg all-in). This replaced an arbitrary 58% threshold that was excluding valid bets (e.g. Harvey Thomas at 57% High was incorrectly dimmed).

---

## Legs Table (Multi Builder)

Now uses `dash_table.DataTable` with native sortable columns.

### What's shown

- **All tackle legs** with Line ≥ 3 and no Short Break travel (hard avoids pre-filtered — don't show at all)
- Disposals and marks are excluded from the display table (tackle strategy only)
- Default sort: active (full opacity) first, then WR% desc, then n desc

### Visual dimming

| Opacity | Condition |
|---|---|
| Full | Med/High confidence AND WR ≥ 56% |
| 25% (dimmed) | Everything else — low confidence, or WR below threshold |

The dim rule is placed **last** in `style_data_conditional` so it overrides all column-specific colour rules for inactive rows.

### Hidden columns (used by filter queries)

`_active` (0/1), `_wr_pct` (float), `_avl_num` (float), `_cons_num` (float)

These must be **declared in `dt_cols`** (with `name: ''`) for DataTable filter queries to work. They are listed in `hidden_columns` to suppress display.

### Heat colour bands

| Column | Teal | Blue | Amber | Red |
|---|---|---|---|---|
| Hist WR | ≥65% | 56–65% | 53–56% | <53% |
| Confidence | High | Med | Low | — |
| Line | ≥5 | 4–5 | 3.5–4 | <3.5 |
| AvL | ≤-10% | -10–0% | 0–5% | >5% |
| Consistency | ≥65% | 55–65% | 45–55% | <45% |

---

## Multi Builder — Hedge Pick System

### Core function: `build_hedge_picks(legs_by_quality)`

Returns: `(fmt_name, payout_desc, combo_picks, combo_meta, jackpot_info)`

**Tiered logic by number of available legs (n):**

| n | Combo plays | Jackpot entry |
|---|---|---|
| <3 | SKIP | None |
| 3 | 1× 3-leg hedge | None |
| 4 | 1× 4-leg hedge | None |
| 5 | 1× 5-leg hedge | None |
| 6 | 1× 6-leg hedge | None |
| 7+ | 7 balanced picks (see below) | 1× n-leg entry (capped at 12-leg) |

### Balanced picks algorithm (replaced C(7,6) from 2026-04-17)

Previously used C(7,6): took the best-7 legs, generated all 7 combinations of 6. This gave the best-7 each an exposure of 6-7 and everyone else an exposure of 1 (jackpot only).

**Now uses `_balanced_picks(legs, n_picks=7, pick_size=6, team_cap=2)`:**

Each round:
1. Sort candidates by `(current_exposure ASC, WR_rank ASC)`
2. Greedily pick 6 legs respecting max-2-per-team
3. If short (team cap too restrictive), relax cap and fill
4. Update exposure counts

Result: with 11 legs and 7 picks × 6 slots = 42 appearances, each leg gets **3–4 appearances** (vs 7 vs 1 before). Exposure chart is now balanced.

### Sort order for hedge picks

```python
def _wr_rank_key(row):
    # -WR% desc, then -n desc as tiebreak
    return (-wr_pct, -n_sample)
```

**No longer uses `leg_quality_score()`.** Picks are ranked purely by historical WR evidence.

### Team-capped best-7 pre-selection (for n ≥ 7)

Before passing to `build_hedge_picks`, a greedy selection enforces max-2-per-team:
1. Walk down WR-ranked list
2. Skip teams already at 2 legs
3. If still short after one pass, relax cap and fill from overflow
4. Jackpot gets all legs: team-balanced 7 first, then rest in WR order

### Active pool (what goes into hedge picks)

**`active_upcoming`** = rows from `table_upcoming` where `_wr_qualifies()` is True:
```python
def _wr_qualifies(row):
    return conf in ('High', 'Med') AND wr_pct >= 56
```
Rows with no Hist WR data or Low confidence are NOT included in picks.

### Dabble Pick'Em Payout Tables

**Breakeven WR per leg:**

| Format | BE WR/leg |
|---|---|
| 2-leg all-in | 55.9% ← highest across all formats |
| 3-leg hedge | 55.3% |
| 4-leg hedge | 53.9% |
| 5-leg hedge | 53.7% |
| 6-leg hedge | 53.2% ← primary format |
| 7–12 leg jackpot | ~53.0–53.5% |

**Working inclusion threshold: 56%** — clears every format with meaningful margin.

---

## Correlation & Team Concentration

### Same-team concentration indicator (per pick card)

| Signal | Condition |
|---|---|
| Teal `✓ max 2 per team` | max same-team ≤ 2 |
| Amber `~ 3 from same team — watch` | max same-team = 3 |
| Red `⚠ 4 from same team — historically -EV` | max same-team ≥ 4 |

### Why team diversity matters

- With 4 teams: if one game goes bad (high-possession, no contested ball), 2-3 legs from that team fail simultaneously. Historical same-team n=4 WR is 51.5% (below 53.9% BE).
- With 6 teams: failures are more independent, real-world WR tracks historical WR closely.
- Same outlay regardless of team count — but 6-team slates are structurally better bets.
- The balanced algorithm + max-2-per-team naturally handles this without hard rules.

---

## Key Architecture Notes

### Data flow in `build_multi_builder_layout()`

```
processed_data_by_stat['tackles']
    → all tackle rows with Line ≥ 3, no Short Break  →  table_upcoming
    → Hard avoids pre-filtered
    → get_leg_historical_wr() computed per row
    → WR Range + Confidence computed
    → Sort: active first, then WR desc
    → _wr_qualifies() filter  →  active_upcoming  →  hedge picks
    → _balanced_picks()  →  build_hedge_picks()
    → DataTable rendered with cond_styles dimming
```

### `_load_bet_history()`

Lazy-loads Google Sheets bet log once per process. Caches in `_BET_HISTORY['df']`. Filters to Tackle rows with a W/L result. Returns DataFrame with columns: Position, DvP, Opponent, win (0/1).

To **refresh** (e.g. after new results added), restart the process — no in-app refresh button yet.

---

## Known Issues / Edge Cases

- `_BET_HISTORY` cache is process-lifetime only. New bet results added to Sheets mid-session won't appear until restart.
- Legs with no Hist WR data (new players, rare matchups) are always dimmed and excluded from picks. If you want to include them, you need to override manually.
- `_infer_strategy()` in the Performance tab does not apply the narrow ground exemption for Wing/Ruck/GenD (stadium column not stored in Sheets). Tiny number of edge cases (GMHBA games) may be misclassified — acceptable for historical analysis.
- `upcoming` variable (from `combined`) is computed but never used after `active_upcoming` was removed from the T1/T2 path. Dead code — harmless.
- Several local heat functions (`_tier_heat`, `_line_heat`, etc.) remain defined but are unused since the DataTable replaced the HTML table. Dead code — harmless.

---

## Key Constants

```python
GOOGLE_SHEET_ID         = "10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"
GOOGLE_SHEET_TAB        = "Bet Log"
PAIR_LOG_TAB            = "Pair Log"
TACKLE_BAD_OPPONENTS    = {'GCS', 'GEE'}
NARROW_GROUNDS_NO_TACKLE = {"GMHBA Stadium", "Kardinia Park"}
ACTIVE_CODES            = ['T1', 'T2']   # still used for flagging; not for pick selection
```

---

## Stores in Layout

```python
dcc.Store(id='pairings-store', data={})
dcc.Store(id='rr-top-n-store', storage_type='local', data=4)
dcc.Store(id='placed-bets-store', storage_type='local', data=[])
dcc.Store(id='excluded-teams-store', storage_type='local', data=[])
```

`pairings-store` schema:
```json
{
  "hedge_picks": [[{Player, Team, Opponent, Line, Position, Strategy}]],
  "hedge_format": "6-leg Hedge  .  7 balanced picks from 11 available",
  "hedge_meta": {"min_wins": 4, "tiers": [[4, 0.5], [5, 2.5], [6, 25.0]], "breakeven": "53.2%"},
  "jackpot_pick": [{Player, Team, Opponent, Line, Position, Strategy}],
  "jackpot_format": "11-leg Jackpot  (1 entry)",
  "jackpot_meta": {"min_wins": 8, "tiers": [...], "n": 11}
}
```

---

## Pending / Possible Next Steps

- **InsM + Moderate Unders avoid rule**: at 52% (44 bets) — add rule when sample reaches ~50 bets or if WR drops further
- **GCS/GEE blanket avoid**: currently applied to all positions — watch whether Wing/Ruck/GenD vs GCS/GEE surface in the dimmed rows with good WR as sample grows
- **_BET_HISTORY refresh button**: currently requires restart to pick up new results; add a "Refresh WR cache" button
- **`leg_quality_score()`**: now dead code for picks (replaced by WR rank) — can be removed
- **Dead local heat functions**: `_tier_heat`, `_line_heat`, `_avl_heat` etc. — can be cleaned up
- **Performance tab**: `_infer_strategy()` — add narrow ground exemption for Wing/Ruck/GenD for historical accuracy
- **Team-level pricing bias to track**: ESS 88.2% (17), WCE 80.0% (25), WBD 72.7% (33) — Dabble oversetting. ADE 50.0% (14), STK 50.0% (24) — Dabble undersetting. Re-evaluate at 30+ bets each.
- **Round 11–15 weakness**: 52.9% WR (34 bets) — possible bye round effect or line tightening; investigate
