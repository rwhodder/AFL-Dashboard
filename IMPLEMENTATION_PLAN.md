# Criteria Implementation Plan

Staged changes to `calculate_bet_flag()` in `app.py`.  
Analysis date: 2026-04-17. Do not implement until analysis phase is complete.

---

## Status: IMPLEMENTED 2026-04-17

---

## Confirmed Changes (evidence is clear)

### 1. Remove SYD from `TACKLE_BAD_OPPONENTS`
- **File:** `app.py` — `TACKLE_BAD_OPPONENTS` constant
- **Change:** `{'SYD', 'GCS', 'GEE'}` → `{'GCS', 'GEE'}`
- **Evidence:** SYD is 71.4% WR after all other filters applied (14 bets). The bad historical SYD numbers were entirely driven by coincidental overlap with other bad signals. No independent opponent effect.

### 2. Add GenD to T1
- **File:** `app.py` — `calculate_bet_flag()`, T1 position check
- **Change:** `if position in ('Wing', 'Ruck'):` → `if position in ('Wing', 'Ruck', 'GenD'):`
- **Evidence:** GenD is 91.7% all-time (12 bets), 83.3% in 2026 (6 bets). Strongest positional signal in the dataset. Consistent across both years. Currently being treated as standard T2.

### 3. Add MedF to avoid filters
- **File:** `app.py` — `calculate_bet_flag()`, base avoid section
- **Change:** Add `if position == 'MedF': return {"priority": "", "description": ""}` after the line < 3 check
- **Evidence:** MedF is 43.2% all-time (37 bets), 45.5% in 2025, 40.0% in 2026. Consistently terrible and getting worse. Currently being included as T2.

### 4. ~~Expand Strong Easy avoid to all positions~~ — CANCELLED
- **Reason:** After checking what survives all other filters, the 12 non-FwdMid Strong Easy bets are performing at 66.7% WR. The AvL filter is already eliminating the bad Strong Easy bets. Adding a blanket Strong Easy avoid would cut good bets.

### 5. Replace FwdMid rules with single coherent rule
- **File:** `app.py` — `calculate_bet_flag()`, FwdMid section
- **Logic:** FwdMid is a high-variance position that needs a *meaningful* matchup signal to qualify. Slight Unders, Neutral, and Easy DvPs don't provide that — only Moderate Unders or Strong Unders do.
- **Change:** Replace the two existing FwdMid checks with one:
  ```python
  # FwdMid requires Moderate Unders or Strong Unders — everything else is noise
  if position == 'FwdMid' and dvp not in ('Moderate Unders', 'Strong Unders'):
      # more precisely:
      if not ('Moderate Unders' in dvp or 'Strong Unders' in dvp):
          return {"priority": "", "description": ""}
  ```
- **Replaces:**
  - Old rule 1: `FwdMid + any Easy DvP → avoid` (still covered — Easy is not Mod/Strong Unders)
  - Old rule 2: `FwdMid + Slight Unders → avoid` (new addition — Slight Unders is not Mod/Strong Unders)
- **Evidence:**
  - FwdMid × Moderate Unders: 83.3% WR (12 bets) ✓ keep
  - FwdMid × Slight Unders: 43.8% WR (16 bets) ✗ avoid
  - FwdMid × any Easy: ~40.7% WR ✗ avoid (existing)
- **Note:** SmF + Slight Unders was considered but dropped — only 15 bets and 2026 is 57.1% (above BE). Unstable signal.
- **Note:** InsM + Moderate Unders was considered but moved to watch list — too close to BE to call.

### ~~SmF + Slight Unders~~ — DROPPED (sample too small, 2026 at 57.1%)

### ~~InsM + Moderate Unders~~ — WATCH LIST (see below)

### 6. Team-level line pricing bias — track only, do not implement yet
- ESS players: 88.2% WR (17 bets) — Dabble consistently oversets
- WCE players: 80.0% WR (25 bets)
- WBD players: 72.7% WR (33 bets)
- ADE players: 50.0% WR (14 bets) — Dabble undersets
- STK players: 50.0% WR (24 bets)
- BRL players: 51.9% WR (27 bets)
- **Action:** Use as a soft deprioritisation signal in leg quality scoring. Do not hard-avoid. Re-evaluate at ~30+ bets per team. ESS InsM specifically (92.9%, 14 bets) is the strongest sub-signal.

---

## Watch List (more data needed before acting)

### NTH — 2026 weak signal
- All-time: 56.5% (23 bets) — fine
- 2026: 40.0% (10 bets) — concerning
- **Decision:** Do not add to avoid list yet. Need ~15-20 more 2026 bets to confirm. Re-evaluate mid-season.

### ADE — flipped in 2026
- All-time: 65.2% (23 bets) — strong
- 2026: 42.9% (7 bets) — terrible
- **Decision:** Do not add to avoid list yet. 7 bets is too small. Re-evaluate at 15+ 2026 bets.

### InsM + Moderate Unders — borderline, watch only
- Combined: 31 bets at 51.6% — below safe threshold
- 2025: 16 bets at 50.0% / 2026: 15 bets at 53.3% — improving, now right at 6-leg BE
- Clear outlier within InsM DvP profile (10pp below next-worst DvP type)
- **Decision:** Do not implement yet. Re-evaluate at ~50 total bets or if 2026 drops back below 52%.

### GCS/GEE blanket avoid — confirmed appropriate, no position exceptions needed
- Almost all flagged bets against GCS/GEE are InsM (10/12 GCS, 14/18 GEE)
- T1 positions vs GCS/GEE: 4 combined bets, all losses — no evidence to split
- Blanket avoid is effectively an InsM filter in practice anyway

### BRL — position-specific issue
- Overall after filters: 55.6% (27 bets) — marginal pass
- InsM vs BRL in 2026: 40.0% (5 bets) — fail
- GenD/Wing/Ruck vs BRL: no meaningful sample
- **Decision:** Do not blanket-avoid BRL. The problem appears to be InsM specifically. Monitor InsM vs BRL sample — consider a position+opponent interaction rule if it holds.

### Neutral DvP — 2026 sample too small
- All-time: 58.3% (24 bets) — fine
- 2026: 33.3% (6 bets) — alarming but tiny sample
- **Decision:** Watch only. Not actionable yet.

### Moderate Easy DvP — marginal
- All-time: 54.8%, 2026: 55.6%
- Passes 6-leg breakeven (53.2%) but fails 3-leg (55.3%)
- **Decision:** Keep including. Only becomes an issue if we are in a 3-leg scenario (which is already a minimal-legs situation).

---

## Breakeven Reference

| Format | BE WR/leg | Notes |
|---|---|---|
| 3-leg hedge | 55.3% | Rarely used — only when <4 legs |
| 4-leg hedge | 53.9% | |
| 5-leg hedge | 53.7% | |
| 6-leg hedge (C(7,6)) | 53.2% | Primary format |
| 7–12 leg jackpot | ~53.0–53.5% | Additive EV on top |
| All-in 2–10 leg | 52.5–55.9% | Not primary strategy |

**Working inclusion threshold: 54%** (clears all formats safely)

---

## Analysis Still To Do

- [ ] Check if the Slight Easy anomaly (70.1% WR) holds in 2026 specifically
- [ ] Investigate NTH structural change — new coach? Player turnover?
- [ ] Check disposals and marks for same criteria patterns (or are avoids tackle-specific?)
- [ ] Re-run opponent analysis once more 2026 data comes in (~Round 8-10)
- [ ] Validate GenD sample further — positions can be miscoded in early data
- [ ] Deep dive on ESS/WCE/WBD team pricing bias — confirm it holds in 2026 and by position
- [ ] Investigate Round 11-15 weakness (52.9%, 34 bets) — bye round effect or line tightening?
- [ ] High consistency (61-80%) + Mod/Slight Unders = 48.3% FAIL (29 bets) — is this worth adding as an avoid?
- [ ] Check ADE InsM specifically (37.5%, 8 bets) vs ADE opponent effect — are these independent?
- [ ] Validate the Slight Easy DvP anomaly — why does "Easy" produce 69.8% for InsM and 71.4% for SmF?
