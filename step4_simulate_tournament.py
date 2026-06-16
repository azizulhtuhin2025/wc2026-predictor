"""
STEP 4 - Monte Carlo Tournament Simulation
============================================
Simulates the full FIFA World Cup 2026 bracket 10,000 times
using the trained models to estimate each team's probability of:
  - Advancing from group stage
  - Reaching Round of 32 / QF / SF / Final / Winning

New 2026 format:
  - 48 teams, 12 groups of 4
  - Top 2 from each group + 8 best 3rd-place teams advance (32 total)
  - Then knockout rounds: R32 → R16 → QF → SF → Final

Output: outputs/simulation_results.csv

Requirements: pip install pandas numpy scikit-learn xgboost joblib tqdm
"""

import pandas as pd
import numpy as np
import joblib
import os
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Create outputs directory
os.makedirs("outputs", exist_ok=True)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

N_SIMULATIONS = 10_000

# ─────────────────────────────────────────────
# Feature columns (must match step3 exactly)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "neutral", "tournament_weight",
    "elo_diff", "home_elo", "away_elo",
    "home_form_scored", "home_form_conceded", "home_form_winrate",
    "away_form_scored", "away_form_conceded", "away_form_winrate",
    "form_scored_diff", "form_conceded_diff", "form_winrate_diff",
]

# ─────────────────────────────────────────────
# Load models and data
# ─────────────────────────────────────────────
print("📂 Loading models and data...")
outcome_model = joblib.load("models/outcome_model.pkl")
poisson_home = joblib.load("models/poisson_home.pkl")
poisson_away = joblib.load("models/poisson_away.pkl")

elo_df = pd.read_csv("data/current_elo.csv")
elo_lookup = dict(zip(elo_df["team"], elo_df["elo"]))

groups_df = pd.read_csv("data/wc2026_groups.csv")
features_df = pd.read_csv("data/features.csv", parse_dates=["date"])

print("   Building team form profiles...")
team_form = {}

for team in elo_lookup:
    tm = features_df[
        (features_df["home_team"] == team) | (features_df["away_team"] == team)
    ].tail(10)

    scored = []
    conceded = []
    wins = []

    for _, r in tm.iterrows():
        if r["home_team"] == team:
            scored.append(r["home_score"])
            conceded.append(r["away_score"])
            wins.append(1 if r["home_score"] > r["away_score"] else (0.5 if r["home_score"] == r["away_score"] else 0))
        else:
            scored.append(r["away_score"])
            conceded.append(r["home_score"])
            wins.append(1 if r["away_score"] > r["home_score"] else (0.5 if r["away_score"] == r["home_score"] else 0))

    team_form[team] = {
        "form_scored": np.mean(scored) if scored else 1.2,
        "form_conceded": np.mean(conceded) if conceded else 1.2,
        "form_winrate": np.mean(wins) if wins else 0.4,
    }

DEFAULT_FORM = {"form_scored": 1.2, "form_conceded": 1.2, "form_winrate": 0.4}
DEFAULT_ELO = 1400

def get_team_elo(team):
    return elo_lookup.get(team, DEFAULT_ELO)

def get_team_form(team):
    return team_form.get(team, DEFAULT_FORM)

# ─────────────────────────────────────────────
# Match prediction function
# ─────────────────────────────────────────────
def predict_match(home, away, neutral=True):
    """
    Returns (home_goals, away_goals) sampled from Poisson model.
    Uses outcome model probabilities to adjust.
    """
    h_elo = get_team_elo(home)
    a_elo = get_team_elo(away)
    h_form = get_team_form(home)
    a_form = get_team_form(away)

    features = {
        "neutral": int(neutral),
        "tournament_weight": 4.0,  # World Cup weight
        "elo_diff": h_elo - a_elo,
        "home_elo": h_elo,
        "away_elo": a_elo,
        "home_form_scored": h_form["form_scored"],
        "home_form_conceded": h_form["form_conceded"],
        "home_form_winrate": h_form["form_winrate"],
        "away_form_scored": a_form["form_scored"],
        "away_form_conceded": a_form["form_conceded"],
        "away_form_winrate": a_form["form_winrate"],
        "form_scored_diff": h_form["form_scored"] - a_form["form_scored"],
        "form_conceded_diff": h_form["form_conceded"] - a_form["form_conceded"],
        "form_winrate_diff": h_form["form_winrate"] - a_form["form_winrate"],
    }

    X = pd.DataFrame([features])[FEATURE_COLS]

    # Get expected goals from Poisson model
    mu_home = max(0.1, poisson_home.predict(X)[0])
    mu_away = max(0.1, poisson_away.predict(X)[0])

    # Sample actual goals
    home_goals = np.random.poisson(mu_home)
    away_goals = np.random.poisson(mu_away)

    return home_goals, away_goals

def simulate_penalty(home, away):
    """50/50 penalty shootout (slight edge to Elo favorite)"""
    h_elo = get_team_elo(home)
    a_elo = get_team_elo(away)
    prob_home = 1 / (1 + 10 ** ((a_elo - h_elo) / 800))  # softer than Elo
    return home if np.random.random() < prob_home else away

# ─────────────────────────────────────────────
# Group stage simulation
# ─────────────────────────────────────────────
def simulate_group(teams):
    """
    Round robin among 4 teams.
    Returns standings sorted by points, then GD, then GF.
    """
    standings = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            home, away = teams[i], teams[j]
            hg, ag = predict_match(home, away, neutral=True)
            standings[home]["gf"] += hg
            standings[home]["ga"] += ag
            standings[away]["gf"] += ag
            standings[away]["ga"] += hg

            if hg > ag:
                standings[home]["pts"] += 3
            elif hg == ag:
                standings[home]["pts"] += 1
                standings[away]["pts"] += 1
            else:
                standings[away]["pts"] += 3

    # Sort: points → GD → GF → random tiebreak
    sorted_teams = sorted(
        teams,
        key=lambda t: (
            standings[t]["pts"],
            standings[t]["gf"] - standings[t]["ga"],
            standings[t]["gf"],
            np.random.random()
        ),
        reverse=True,
    )

    return sorted_teams, standings

# ─────────────────────────────────────────────
# Knockout match
# ─────────────────────────────────────────────
def simulate_knockout(home, away):
    """Single match, with extra time/penalties if needed."""
    hg, ag = predict_match(home, away, neutral=True)
    if hg != ag:
        return (home, hg, ag) if hg > ag else (away, hg, ag)
    else:
        # Penalties
        winner = simulate_penalty(home, away)
        return winner, hg, ag

# ─────────────────────────────────────────────
# Full tournament simulation
# ─────────────────────────────────────────────
print(f"\n🎲 Running {N_SIMULATIONS:,} tournament simulations...")

group_map = groups_df.groupby("group")["team"].apply(list).to_dict()

stage_counts = defaultdict(lambda: defaultdict(int))
# stages: group_advance, r32, r16, qf, sf, final, winner

iterator = range(N_SIMULATIONS)
if HAS_TQDM:
    iterator = tqdm(iterator, desc="Simulating", ncols=70)

for sim in iterator:
    # ── Group Stage ──
    group_winners = []
    group_runners = []
    all_third = []  # (team, pts, gd, gf)

    for g, teams in group_map.items():
        sorted_teams, standings = simulate_group(teams)

        group_winners.append(sorted_teams[0])
        group_runners.append(sorted_teams[1])

        third = sorted_teams[2]
        s = standings[third]
        all_third.append((third, s["pts"], s["gf"] - s["ga"], s["gf"]))

        for team in sorted_teams[:2]:
            stage_counts[team]["group_advance"] += 1
        stage_counts[sorted_teams[2]]["group_third"] += 1

    # Best 8 third-place teams advance
    all_third.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    best_thirds = [t[0] for t in all_third[:8]]
    for t in best_thirds:
        stage_counts[t]["group_advance"] += 1

    # ── Round of 32 ──
    r32_teams = group_winners + group_runners + best_thirds
    np.random.shuffle(r32_teams)
    r16_teams = []
    for i in range(0, len(r32_teams), 2):
        if i + 1 < len(r32_teams):
            winner, _, _ = simulate_knockout(r32_teams[i], r32_teams[i + 1])
        else:
            winner = r32_teams[i]
        r16_teams.append(winner)
        stage_counts[winner]["r32"] += 1

    # ── Round of 16 ──
    qf_teams = []
    for i in range(0, len(r16_teams), 2):
        winner, _, _ = simulate_knockout(r16_teams[i], r16_teams[i + 1])
        qf_teams.append(winner)
        stage_counts[winner]["r16"] += 1

    # ── Quarter-Finals ──
    sf_teams = []
    for i in range(0, len(qf_teams), 2):
        winner, _, _ = simulate_knockout(qf_teams[i], qf_teams[i + 1])
        sf_teams.append(winner)
        stage_counts[winner]["qf"] += 1

    # ── Semi-Finals ──
    final_teams = []
    for i in range(0, len(sf_teams), 2):
        winner, _, _ = simulate_knockout(sf_teams[i], sf_teams[i + 1])
        final_teams.append(winner)
        stage_counts[winner]["sf"] += 1

    # ── Final ──
    if len(final_teams) >= 2:
        champion, _, _ = simulate_knockout(final_teams[0], final_teams[1])
        stage_counts[final_teams[0]]["final"] += 1
        stage_counts[final_teams[1]]["final"] += 1
        stage_counts[champion]["winner"] += 1

# ─────────────────────────────────────────────
# Compile results
# ─────────────────────────────────────────────
print("\n📊 Compiling results...")

all_teams = groups_df["team"].tolist()
rows = []
for team in all_teams:
    sc = stage_counts[team]
    rows.append({
        "team": team,
        "group": groups_df[groups_df["team"] == team]["group"].values[0],
        "elo": round(get_team_elo(team), 0),
        "win_pct": round(sc.get("winner", 0) / N_SIMULATIONS * 100, 2),
        "final_pct": round(sc.get("final", 0) / N_SIMULATIONS * 100, 2),
        "sf_pct": round(sc.get("sf", 0) / N_SIMULATIONS * 100, 2),
        "qf_pct": round(sc.get("qf", 0) / N_SIMULATIONS * 100, 2),
        "r16_pct": round(sc.get("r16", 0) / N_SIMULATIONS * 100, 2),
        "advance_pct": round(sc.get("group_advance", 0) / N_SIMULATIONS * 100, 2),
    })

results_df = pd.DataFrame(rows).sort_values("win_pct", ascending=False).reset_index(drop=True)
results_df.to_csv("outputs/simulation_results.csv", index=False)

# ─────────────────────────────────────────────
# Print top 15 teams
# ─────────────────────────────────────────────
print("\n🏆 TOP 15 WORLD CUP 2026 PREDICTIONS")
print("=" * 75)
print(f"{'Rank':<5} {'Team':<22} {'Group':<7} {'Win%':<8} {'Final%':<8} {'SF%':<8} {'QF%':<8} {'Adv%'}")
print("-" * 75)
for i, row in results_df.head(15).iterrows():
    print(
        f"{i+1:<5} {row['team']:<22} {row['group']:<7} "
        f"{row['win_pct']:<8.1f} {row['final_pct']:<8.1f} "
        f"{row['sf_pct']:<8.1f} {row['qf_pct']:<8.1f} {row['advance_pct']:.1f}"
    )
print("=" * 75)

print("\n💾 Full results saved → outputs/simulation_results.csv")
print()
print("=" * 50)
print("✅ STEP 4 COMPLETE")
print("   Next → Run: python step5_visualize.py")
print("=" * 50)