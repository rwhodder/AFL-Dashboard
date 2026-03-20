# Google Sheets Setup — one time only

## What you need to install
```
pip install gspread google-auth pandas
```

---

## Step 1 — Create a Google Cloud project (5 min)

1. Go to https://console.cloud.google.com
2. Click **New Project** → give it any name (e.g. "AFL Dashboard") → Create
3. In the search bar type **"Google Sheets API"** → click it → **Enable**
4. In the search bar type **"Google Drive API"** → click it → **Enable**

---

## Step 2 — Create a Service Account (3 min)

1. Go to **APIs & Services → Credentials**
2. Click **Create Credentials → Service Account**
3. Give it any name → click **Done**
4. Click the service account you just created
5. Go to the **Keys** tab → **Add Key → Create new key → JSON**
6. A JSON file downloads — **rename it `google_credentials.json`**
7. **Move it into the same folder as `app.py`**

---

## Step 3 — Share your Google Sheet with the service account

1. Open `google_credentials.json` in a text editor
2. Find the `"client_email"` field — copy that email address
   (looks like `something@your-project.iam.gserviceaccount.com`)
3. Open your Google Sheet
4. Click **Share** (top right)
5. Paste the service account email → set role to **Editor** → Share

---

## Step 4 — Confirm your Sheet ID

Your sheet URL looks like:
```
https://docs.google.com/spreadsheets/d/10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w/edit
```
The ID is the long string between `/d/` and `/edit`.

This is already set in both `app.py` and `update_results.py`:
```python
GOOGLE_SHEET_ID = "10GNqW9nE2fmacbdRQZtga3u8nGQlXch066GYQ9JgX7w"
```
No change needed unless you move to a different sheet.

---

## Weekly workflow

### Thursday–Sunday (each session)
1. Open dashboard in VS Code → `python app.py`
2. Check the flagged bets
3. Place any bets
4. Click **📤 Push Bets to Google Sheets**
   - All three stat types push at once
   - Already-pushed rows are skipped automatically
   - No CSV exports, no manual copy-paste

### Mid-week (after downloading stats)
1. Download latest `afl_player_stats.csv` into your project folder
2. Run: `python update_results.py`
   - Finds every row in your sheet with an empty "Actual" column
   - Fills in the real stat and calculates W/L automatically
   - Takes ~10 seconds

---

## Column order in your sheet
```
Type | Year | Round | Player | Team | Opponent | Position |
Travel Fatigue | Weather | DvP | Line | Avg vs Line |
Line Consistency | Bet Priority | Bet Flag | Actual | W/L
```
The scripts write exactly these columns in this order.
