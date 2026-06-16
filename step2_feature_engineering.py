"""
STEP 2 - Feature Engineering
==============================
Transforms raw match data into ML-ready features for every historical match.

Features built:
  - Rolling form (goals scored/conceded, win rate over last 10 matches)
  - Elo rating for each team at the time of the match
  - Match context (neutral ground, tournament weight)
  - Head-to-head record

Output: data/features.csv

Requirements: pip install pandas numpy
"""

import pandas as pd
import numpy as np

print("🔧 Loading raw match data...")
df = pd.read_csv("data/results.csv", parse_dates=["date"])
print(f"   Loaded {len(df):,} matches from {df['date'].min().year} to {df['date'].max().year}")

# ─────────────────────────────────────────────
# 1. Filter to meaningful matches only
# ─────────────────────────────────────────────
# Drop matches with missing scores
df = df.dropna(subset=["home_score", "away_score"])

# Tournament weights — World Cup matches matter more
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 4,
    "FIFA World Cup qualification": 2,
    "UEFA Euro": 3,
    "Copa América": 3,
    "AFC Asian Cup": 3,
    "Africa Cup of Nations": 3,
    "UEFA Nations League": 2,
    "Friendly": 0.5,
}

def get_tournament_weight(t):
    for key, w in TOURNAMENT_WEIGHTS.items():
        if key in str(t):
            return w
    return 1.0

df["tournament_weight"] = df["tournament"].apply(get_tournament_weight)

print(f"   After cleaning: {len(df):,} matches")

# ─────────────────────────────────────────────
# 2. Build Elo ratings dynamically from match history
# ─────────────────────────────────────────────
print("⚡ Computing Elo ratings across all matches...")

ELO_BASE = 1500
ELO_K = 32

elo_ratings = {}

def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def update_elo(ra, rb, result, k=ELO_K):
    """result: 1=win, 0.5=draw, 0=loss from perspective of team A"""
    ea = expected_score(ra, rb)
    new_ra = ra + k * (result - ea)
    new_rb = rb + k * ((1 - result) - (1 - ea))
    return new_ra, new_rb

home_elo_before = []
away_elo_before = []

for _, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    ra = elo_ratings.get(home, ELO_BASE)
    rb = elo_ratings.get(away, ELO_BASE)

    home_elo_before.append(ra)
    away_elo_before.append(rb)

    # Determine result
    if row["home_score"] > row["away_score"]:
        result = 1
    elif row["home_score"] == row["away_score"]:
        result = 0.5
    else:
        result = 0

    new_ra, new_rb = update_elo(ra, rb, result)
    elo_ratings[home] = new_ra
    elo_ratings[away] = new_rb

df["home_elo"] = home_elo_before
df["away_elo"] = away_elo_before
df["elo_diff"] = df["home_elo"] - df["away_elo"]

print(f"   ✅ Elo computed for {len(elo_ratings)} teams")

# Save final Elo ratings for use in Step 4 (tournament simulation)
elo_df = pd.DataFrame(
    [(team, rating) for team, rating in elo_ratings.items()],
    columns=["team", "elo"]
).sort_values("elo", ascending=False)
elo_df.to_csv("data/current_elo.csv", index=False)
print(f"   Top 5 teams by Elo: {list(elo_df.head(5)['team'])}")

# ─────────────────────────────────────────────
# 3. Rolling form features (last 10 matches per team)
# ─────────────────────────────────────────────
print("📊 Building rolling form features...")

df = df.sort_values("date").reset_index(drop=True)

team_history = {}  # team -> list of (goals_scored, goals_conceded, result)

home_form_score = []
home_form_concede = []
home_form_winrate = []
away_form_score = []
away_form_concede = []
away_form_winrate = []

WINDOW = 10

def get_form(history, window=WINDOW):
    recent = history[-window:] if len(history) >= window else history
    if not recent:
        return 1.2, 1.2, 0.4  # neutral defaults
    goals_scored = np.mean([h[0] for h in recent])
    goals_conceded = np.mean([h[1] for h in recent])
    win_rate = np.mean([h[2] for h in recent])
    return goals_scored, goals_conceded, win_rate

for _, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    h_scored, h_conceded, h_winrate = get_form(team_history.get(home, []))
    a_scored, a_conceded, a_winrate = get_form(team_history.get(away, []))

    home_form_score.append(h_scored)
    home_form_concede.append(h_conceded)
    home_form_winrate.append(h_winrate)
    away_form_score.append(a_scored)
    away_form_concede.append(a_conceded)
    away_form_winrate.append(a_winrate)

    # Update history after recording (no lookahead bias)
    if home not in team_history:
        team_history[home] = []
    if away not in team_history:
        team_history[away] = []

    h_res = 1 if row["home_score"] > row["away_score"] else (0.5 if row["home_score"] == row["away_score"] else 0)
    a_res = 1 - h_res if h_res != 0.5 else 0.5

    team_history[home].append((row["home_score"], row["away_score"], h_res))
    team_history[away].append((row["away_score"], row["home_score"], a_res))

df["home_form_scored"] = home_form_score
df["home_form_conceded"] = home_form_concede
df["home_form_winrate"] = home_form_winrate
df["away_form_scored"] = away_form_score
df["away_form_conceded"] = away_form_concede
df["away_form_winrate"] = away_form_winrate

# Derived diff features
df["form_scored_diff"] = df["home_form_scored"] - df["away_form_scored"]
df["form_conceded_diff"] = df["home_form_conceded"] - df["away_form_conceded"]
df["form_winrate_diff"] = df["home_form_winrate"] - df["away_form_winrate"]

# ─────────────────────────────────────────────
# 4. Match context features
# ─────────────────────────────────────────────
df["neutral"] = df["neutral"].astype(int)

# ─────────────────────────────────────────────
# 5. Target variable: outcome
# ─────────────────────────────────────────────
# 1 = home/first team wins, 0 = draw, -1 = loss
def get_outcome(row):
    if row["home_score"] > row["away_score"]:
        return 1
    elif row["home_score"] == row["away_score"]:
        return 0
    else:
        return -1

df["outcome"] = df.apply(get_outcome, axis=1)

# ─────────────────────────────────────────────
# 6. Save feature dataset
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "date", "home_team", "away_team",
    "home_score", "away_score", "outcome",
    "neutral", "tournament_weight",
    "home_elo", "away_elo", "elo_diff",
    "home_form_scored", "home_form_conceded", "home_form_winrate",
    "away_form_scored", "away_form_conceded", "away_form_winrate",
    "form_scored_diff", "form_conceded_diff", "form_winrate_diff",
]

features_df = df[FEATURE_COLS].copy()
features_df.to_csv("data/features.csv", index=False)

print(f"   ✅ Feature dataset saved: {len(features_df):,} rows  →  data/features.csv")
print(f"   Feature columns: {FEATURE_COLS[6:]}")
print()
print("=" * 50)
print("✅ STEP 2 COMPLETE")
print("   Next → Run: python step3_train_model.py")
print("=" * 50)
