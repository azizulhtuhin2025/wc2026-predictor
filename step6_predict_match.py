"""
STEP 6 - Interactive Match Predictor (Bonus)
=============================================
Type any two team names and get a detailed match prediction:
  - Win/Draw/Loss probabilities
  - Predicted scoreline
  - Most likely scorelines
  - Head-to-head record

Requirements: pip install pandas numpy scikit-learn xgboost joblib
"""

import pandas as pd
import numpy as np
import joblib
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Load everything
# ─────────────────────────────────────────────
print("🔧 Loading models...")
outcome_model = joblib.load("models/outcome_model.pkl")
poisson_home = joblib.load("models/poisson_home.pkl")
poisson_away = joblib.load("models/poisson_away.pkl")
FEATURE_COLS = pd.read_csv("models/feature_cols.csv", header=None)[0].tolist()

elo_df = pd.read_csv("data/current_elo.csv")
elo_lookup = dict(zip(elo_df["team"], elo_df["elo"]))

features_df = pd.read_csv("data/features.csv", parse_dates=["date"])
results_df = pd.read_csv("data/results.csv", parse_dates=["date"])

DEFAULT_ELO = 1400

def get_team_elo(team):
    return elo_lookup.get(team, DEFAULT_ELO)

def get_team_form(team):
    tm = features_df[
        (features_df["home_team"] == team) | (features_df["away_team"] == team)
    ].tail(10)

    scored, conceded, wins = [], [], []
    for _, r in tm.iterrows():
        if r["home_team"] == team:
            scored.append(r["home_score"])
            conceded.append(r["away_score"])
            wins.append(1 if r["home_score"] > r["away_score"] else (0.5 if r["home_score"] == r["away_score"] else 0))
        else:
            scored.append(r["away_score"])
            conceded.append(r["home_score"])
            wins.append(1 if r["away_score"] > r["home_score"] else (0.5 if r["away_score"] == r["home_score"] else 0))

    return {
        "form_scored": np.mean(scored) if scored else 1.2,
        "form_conceded": np.mean(conceded) if conceded else 1.2,
        "form_winrate": np.mean(wins) if wins else 0.4,
    }

def build_features(home, away):
    h_elo = get_team_elo(home)
    a_elo = get_team_elo(away)
    h_form = get_team_form(home)
    a_form = get_team_form(away)

    return pd.DataFrame([{
        "neutral": 1,
        "tournament_weight": 4.0,
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
    }])[FEATURE_COLS]

def get_head_to_head(t1, t2, last_n=10):
    h2h = results_df[
        ((results_df["home_team"] == t1) & (results_df["away_team"] == t2)) |
        ((results_df["home_team"] == t2) & (results_df["away_team"] == t1))
    ].tail(last_n)

    t1_wins = t2_wins = draws = 0
    for _, r in h2h.iterrows():
        if r["home_team"] == t1:
            if r["home_score"] > r["away_score"]: t1_wins += 1
            elif r["home_score"] == r["away_score"]: draws += 1
            else: t2_wins += 1
        else:
            if r["away_score"] > r["home_score"]: t1_wins += 1
            elif r["home_score"] == r["away_score"]: draws += 1
            else: t2_wins += 1

    return t1_wins, draws, t2_wins, len(h2h)

def predict_match(home, away, n_sim=5000):
    X = build_features(home, away)

    # Outcome probabilities
    probs = outcome_model.predict_proba(X)[0]
    # Classes: 0=Away Win, 1=Draw, 2=Home Win
    classes = outcome_model.classes_
    prob_map = dict(zip(classes, probs))
    p_home_win = prob_map.get(2, 0)
    p_draw = prob_map.get(1, 0)
    p_away_win = prob_map.get(0, 0)

    # Poisson goal sampling
    mu_home = max(0.1, poisson_home.predict(X)[0])
    mu_away = max(0.1, poisson_away.predict(X)[0])

    scorelines = Counter()
    for _ in range(n_sim):
        hg = np.random.poisson(mu_home)
        ag = np.random.poisson(mu_away)
        scorelines[(hg, ag)] += 1

    top_scorelines = scorelines.most_common(5)

    return {
        "p_home_win": p_home_win,
        "p_draw": p_draw,
        "p_away_win": p_away_win,
        "mu_home": mu_home,
        "mu_away": mu_away,
        "top_scorelines": top_scorelines,
        "n_sim": n_sim,
    }

def display_prediction(home, away):
    print("\n" + "═" * 55)
    print(f"  ⚽  {home}  vs  {away}")
    print("═" * 55)

    # Check if teams exist in data
    known_teams = set(results_df["home_team"]).union(set(results_df["away_team"]))
    if home not in known_teams:
        print(f"  ⚠️  '{home}' not found in dataset. Using default stats.")
    if away not in known_teams:
        print(f"  ⚠️  '{away}' not found in dataset. Using default stats.")

    result = predict_match(home, away)

    print(f"\n  Elo Ratings:")
    print(f"    {home:<22}  {get_team_elo(home):.0f}")
    print(f"    {away:<22}  {get_team_elo(away):.0f}")

    print(f"\n  Win Probabilities:")
    pw = result["p_home_win"] * 100
    pd_ = result["p_draw"] * 100
    pa = result["p_away_win"] * 100

    bar_len = 30
    h_bar = "█" * int(pw / 100 * bar_len)
    d_bar = "░" * int(pd_ / 100 * bar_len)
    a_bar = "▒" * int(pa / 100 * bar_len)

    print(f"    {home:<22}  {h_bar}  {pw:.1f}%")
    print(f"    {'Draw':<22}  {d_bar}  {pd_:.1f}%")
    print(f"    {away:<22}  {a_bar}  {pa:.1f}%")

    print(f"\n  Expected Goals:")
    print(f"    {home:<22}  {result['mu_home']:.2f}")
    print(f"    {away:<22}  {result['mu_away']:.2f}")

    print(f"\n  Most Likely Scorelines:")
    for (hg, ag), count in result["top_scorelines"]:
        pct = count / result["n_sim"] * 100
        label = f"  {home} {hg} - {ag} {away}"
        print(f"    {label:<40}  {pct:.1f}%")

    # Head to head
    t1w, draws, t2w, total = get_head_to_head(home, away)
    if total > 0:
        print(f"\n  Head-to-Head (last {total} meetings):")
        print(f"    {home} wins: {t1w}  |  Draws: {draws}  |  {away} wins: {t2w}")

    # Verdict
    if pw > pa + 5:
        verdict = f"🏅 {home} is FAVORED"
    elif pa > pw + 5:
        verdict = f"🏅 {away} is FAVORED"
    else:
        verdict = "⚖️  This is a CLOSE MATCH"
    print(f"\n  Verdict: {verdict}")
    print("═" * 55)

# ─────────────────────────────────────────────
# Main interactive loop
# ─────────────────────────────────────────────
print("\n🌍 FIFA World Cup 2026 — Interactive Match Predictor")
print("   Type team names exactly as they appear in the dataset.")
print("   Examples: Spain, Argentina, Brazil, England, France")
print("   Type 'quit' to exit.\n")

# Show all available teams
known_teams = sorted(set(results_df["home_team"]).union(set(results_df["away_team"])))
print(f"   ({len(known_teams)} teams in dataset)\n")

while True:
    print()
    home = input("  Team 1 (or 'quit'): ").strip()
    if home.lower() in ("quit", "exit", "q"):
        print("\n  👋 Goodbye!\n")
        break

    away = input("  Team 2           : ").strip()
    if away.lower() in ("quit", "exit", "q"):
        print("\n  👋 Goodbye!\n")
        break

    if not home or not away:
        print("  ⚠️  Please enter both team names.")
        continue

    if home == away:
        print("  ⚠️  Teams must be different.")
        continue

    display_prediction(home, away)
