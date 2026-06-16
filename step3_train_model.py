"""
STEP 3 - Train Prediction Model
=================================
Trains an XGBoost classifier to predict match outcomes (Win/Draw/Loss).
Also trains a Poisson regression model to predict goals.

Validates on World Cup 2018 and 2022 matches.
Saves trained models to models/ folder.

Requirements: pip install pandas numpy scikit-learn xgboost joblib
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not found. Install with: pip install xgboost")
    print("   Falling back to Random Forest.\n")

os.makedirs("models", exist_ok=True)

print("📂 Loading feature data...")
df = pd.read_csv("data/features.csv", parse_dates=["date"])
print(f"   Loaded {len(df):,} matches")

# ─────────────────────────────────────────────
# 1. Train / Validation Split
# ─────────────────────────────────────────────
# Train on everything before 2018, validate on WC 2018 + WC 2022
TRAIN_CUTOFF = "2018-01-01"
VAL1_START = "2018-06-14"
VAL1_END = "2018-07-15"
VAL2_START = "2022-11-20"
VAL2_END = "2022-12-18"

train_df = df[df["date"] < TRAIN_CUTOFF].copy()
val1_df = df[(df["date"] >= VAL1_START) & (df["date"] <= VAL1_END)].copy()
val2_df = df[(df["date"] >= VAL2_START) & (df["date"] <= VAL2_END)].copy()
val_df = pd.concat([val1_df, val2_df])

print(f"   Train set   : {len(train_df):,} matches (before {TRAIN_CUTOFF})")
print(f"   Val WC 2018 : {len(val1_df):,} matches")
print(f"   Val WC 2022 : {len(val2_df):,} matches")

# ─────────────────────────────────────────────
# 2. Prepare Features
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "neutral", "tournament_weight",
    "elo_diff", "home_elo", "away_elo",
    "home_form_scored", "home_form_conceded", "home_form_winrate",
    "away_form_scored", "away_form_conceded", "away_form_winrate",
    "form_scored_diff", "form_conceded_diff", "form_winrate_diff",
]

# Encode outcome: 1→2 (win), 0→1 (draw), -1→0 (loss) for classifier
def encode_outcome(o):
    return {1: 2, 0: 1, -1: 0}[o]

train_df["label"] = train_df["outcome"].apply(encode_outcome)
val_df = val_df.copy()
val_df["label"] = val_df["outcome"].apply(encode_outcome)

X_train = train_df[FEATURE_COLS]
y_train = train_df["label"]
X_val = val_df[FEATURE_COLS]
y_val = val_df["label"]

# ─────────────────────────────────────────────
# 3. Train Models & Compare
# ─────────────────────────────────────────────
print("\n🤖 Training models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42, verbosity=0
    )

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    results[name] = (model, acc)
    print(f"   {name:25s}  →  Val Accuracy: {acc:.3f}")

# Pick best model
best_name = max(results, key=lambda k: results[k][1])
best_model, best_acc = results[best_name]
print(f"\n   🏆 Best model: {best_name} (accuracy: {best_acc:.3f})")

# Detailed report on WC 2022 specifically
print(f"\n📋 Classification report on WC 2022 matches:")
val2_df["label"] = val2_df["outcome"].apply(encode_outcome)
preds_2022 = best_model.predict(val2_df[FEATURE_COLS])
print(classification_report(
    val2_df["label"], preds_2022,
    target_names=["Away Win", "Draw", "Home Win"]
))

# ─────────────────────────────────────────────
# 4. Feature Importance
# ─────────────────────────────────────────────
if HAS_XGB and best_name == "XGBoost":
    importances = best_model.feature_importances_
    fi_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)
    print("📊 Feature Importances (XGBoost):")
    for _, row in fi_df.iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"   {row['feature']:30s}  {bar}  {row['importance']:.3f}")

# ─────────────────────────────────────────────
# 5. Train Poisson Goal Model
# ─────────────────────────────────────────────
print("\n⚽ Training Poisson goal model...")

poisson_home = PoissonRegressor(max_iter=1000)
poisson_away = PoissonRegressor(max_iter=1000)

poisson_home.fit(X_train, train_df["home_score"])
poisson_away.fit(X_train, train_df["away_score"])

# Check fit
pred_home_goals = poisson_home.predict(X_val)
pred_away_goals = poisson_away.predict(X_val)
actual_total = (val_df["home_score"] + val_df["away_score"]).mean()
pred_total = (pred_home_goals + pred_away_goals).mean()
print(f"   Avg actual goals/game : {actual_total:.2f}")
print(f"   Avg predicted goals   : {pred_total:.2f}")

# ─────────────────────────────────────────────
# 6. Save models
# ─────────────────────────────────────────────
joblib.dump(best_model, "models/outcome_model.pkl")
joblib.dump(poisson_home, "models/poisson_home.pkl")
joblib.dump(poisson_away, "models/poisson_away.pkl")

# Save feature column order (important for inference)
pd.Series(FEATURE_COLS).to_csv("models/feature_cols.csv", index=False)

print("\n💾 Saved:")
print("   models/outcome_model.pkl")
print("   models/poisson_home.pkl")
print("   models/poisson_away.pkl")
print("   models/feature_cols.csv")
print()
print("=" * 50)
print("✅ STEP 3 COMPLETE")
print("   Next → Run: python step4_simulate_tournament.py")
print("=" * 50)