# 🏆 FIFA World Cup 2026 Prediction Model

A complete ML pipeline to simulate and predict the 2026 FIFA World Cup.

---

## 📁 Project Structure

```
wc2026_predictor/
├── step1_download_data.py       ← Download match history & setup groups
├── step2_feature_engineering.py ← Build Elo ratings + rolling form features
├── step3_train_model.py         ← Train XGBoost + Poisson models
├── step4_simulate_tournament.py ← 10,000 Monte Carlo simulations
├── step5_visualize.py           ← Generate charts
├── step6_predict_match.py       ← Interactive: predict any match
├── requirements.txt
├── data/                        ← Downloaded & processed data (auto-created)
├── models/                      ← Saved trained models (auto-created)
└── outputs/                     ← Charts and CSV results (auto-created)
```

---

## ⚙️ Setup

### 1. Install Python packages
```bash
pip install -r requirements.txt
```

---

## 🚀 Run Step by Step

### Step 1 — Download Data
```bash
python step1_download_data.py
```
Downloads ~49,000 historical international matches from GitHub.
Creates `data/results.csv` and `data/wc2026_groups.csv`.

---

### Step 2 — Feature Engineering
```bash
python step2_feature_engineering.py
```
Builds:
- **Elo ratings** for every team (dynamically from match history)
- **Rolling form** (goals scored/conceded, win rate — last 10 matches)
- **Match context** (neutral ground, tournament weight)

Output: `data/features.csv`, `data/current_elo.csv`

---

### Step 3 — Train Model
```bash
python step3_train_model.py
```
Trains and compares:
- Logistic Regression
- Random Forest
- **XGBoost** (best performer)

Also trains a **Poisson goal model** to predict scorelines.
Validates on WC 2018 + WC 2022.

Output: `models/outcome_model.pkl`, `models/poisson_home.pkl`, `models/poisson_away.pkl`

---

### Step 4 — Simulate Tournament
```bash
python step4_simulate_tournament.py
```
Runs **10,000 full tournament simulations** using the new 2026 format (48 teams, 12 groups).
Calculates probability for each team at every stage.

Output: `outputs/simulation_results.csv`

---

### Step 5 — Visualize
```bash
python step5_visualize.py
```
Creates 4 charts:
1. Top 20 teams by championship probability
2. Heatmap of all 48 teams across all stages
3. Stage-by-stage funnel for top 10
4. Elo vs Win probability scatter

Output: `outputs/chart1_*.png` through `chart4_*.png`

---

### Step 6 — Predict Any Match (Interactive)
```bash
python step6_predict_match.py
```
Type any two team names and get:
- Win/Draw/Loss probabilities
- Expected goals
- Most likely scorelines
- Head-to-head record

---

## 🧠 How It Works

| Component | Method |
|---|---|
| Team strength | Elo rating (computed from 1872–2026 match history) |
| Form features | Rolling averages over last 10 matches |
| Outcome model | XGBoost classifier (Win/Draw/Loss) |
| Goal model | Poisson regression |
| Tournament | Monte Carlo simulation × 10,000 |

---

## 📊 Sample Output

```
🏆 TOP 15 WORLD CUP 2026 PREDICTIONS
Rank  Team                   Group   Win%     Final%   SF%      QF%
1     Spain                  F       18.4     32.1     51.2     68.3
2     Argentina              B       15.2     27.8     46.5     63.1
3     France                 D       12.7     24.3     42.1     60.8
...
```
