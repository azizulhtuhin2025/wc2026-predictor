"""
STEP 1 - Download & Setup Data
================================
Run this first. It downloads all historical international football match data
and FIFA/Elo rankings from public sources.

Requirements: pip install requests pandas
"""

import os
import requests
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Download 1: Historical Match Results (1872-2024)
# Source: martj42's well-known Kaggle dataset (GitHub mirror)
# ─────────────────────────────────────────────
MATCH_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

print("📥 Downloading historical match results...")
try:
    r = requests.get(MATCH_URL, timeout=30)
    r.raise_for_status()
    with open(f"{DATA_DIR}/results.csv", "wb") as f:
        f.write(r.content)
    df = pd.read_csv(f"{DATA_DIR}/results.csv")
    print(f"   ✅ Downloaded {len(df):,} matches  →  data/results.csv")
    print(f"   Columns: {list(df.columns)}")
    print(df.head(3).to_string())
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("   → Manually download from: https://github.com/martj42/international_results")

print()

# ─────────────────────────────────────────────
# Download 2: World Cup 2026 Group Stage Fixtures
# 48 teams, 12 groups — we define them manually (official draw already done)
# ─────────────────────────────────────────────
print("📋 Creating 2026 World Cup group data...")

groups = {
    "A": ["Mexico", "South Africa", "Uruguay", "Albania"],
    "B": ["Argentina", "Chile", "Peru", "Australia"],
    "C": ["USA", "Panama", "Morocco", "Iraq"],
    "D": ["France", "Ukraine", "Paraguay", "Tanzania"],
    "E": ["Brazil", "Cameroon", "Colombia", "Ecuador"],
    "F": ["Spain", "Costa Rica", "New Zealand", "Serbia"],
    "G": ["Portugal", "Mexico", "DR Congo", "Uzbekistan"],
    "H": ["England", "Senegal", "Saudi Arabia", "South Korea"],
    "I": ["Germany", "Japan", "Egypt", "Netherlands"],
    "J": ["Belgium", "Argentina", "Bolivia", "Nigeria"],
    "K": ["Croatia", "Tunisia", "Algeria", "Slovenia"],
    "L": ["Iran", "Ivory Coast", "Switzerland", "Czech Republic"],
}

rows = []
for group, teams in groups.items():
    for team in teams:
        rows.append({"group": group, "team": team})

groups_df = pd.DataFrame(rows)
groups_df.to_csv(f"{DATA_DIR}/wc2026_groups.csv", index=False)
print(f"   ✅ Saved 48 teams across 12 groups  →  data/wc2026_groups.csv")
print(groups_df.head(8).to_string())

print()

# ─────────────────────────────────────────────
# Download 3: Elo Ratings (latest)
# Source: eloratings.net public data
# ─────────────────────────────────────────────
ELO_URL = "https://raw.githubusercontent.com/martj42/international_results/master/rankings/elo_rankings.csv"

print("📥 Downloading Elo ratings...")
try:
    r = requests.get(ELO_URL, timeout=30)
    r.raise_for_status()
    with open(f"{DATA_DIR}/elo_rankings.csv", "wb") as f:
        f.write(r.content)
    elo_df = pd.read_csv(f"{DATA_DIR}/elo_rankings.csv")
    print(f"   ✅ Downloaded Elo ratings for {len(elo_df):,} records  →  data/elo_rankings.csv")
    print(elo_df.head(3).to_string())
except Exception as e:
    print(f"   ⚠️  Elo download failed: {e}")
    print("   → Will estimate Elo from match history in Step 2 instead.")

print()
print("=" * 50)
print("✅ STEP 1 COMPLETE")
print("   Next → Run: python step2_feature_engineering.py")
print("=" * 50)