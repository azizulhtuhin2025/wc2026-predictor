"""
STEP 5 - Visualize Results
============================
Creates charts from the simulation results:
  1. Top 20 teams by win probability (bar chart)
  2. Group-by-group advancement chances (heatmap)
  3. Tournament bracket probability chart

Saves all charts to outputs/ folder.

Requirements: pip install pandas matplotlib seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)

print("📂 Loading simulation results...")
df = pd.read_csv("outputs/simulation_results.csv")
groups_df = pd.read_csv("data/wc2026_groups.csv")

# ─────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────
plt.style.use("dark_background")
COLORS = {
    "primary": "#00D4AA",
    "secondary": "#FF6B35",
    "accent": "#FFD700",
    "bg": "#0D1117",
    "card": "#161B22",
    "text": "#E6EDF3",
    "muted": "#8B949E",
}

# ─────────────────────────────────────────────
# Chart 1: Top 20 Teams by Win Probability
# ─────────────────────────────────────────────
print("📊 Chart 1: Top 20 teams by win probability...")

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["card"])

top20 = df.head(20).copy()
top20 = top20.sort_values("win_pct")

bars = ax.barh(top20["team"], top20["win_pct"], color=COLORS["primary"], alpha=0.85, height=0.7)

# Color top 3 differently
for i, bar in enumerate(bars):
    team_rank = len(top20) - 1 - i
    if team_rank == 0:
        bar.set_color(COLORS["accent"])
    elif team_rank in [1, 2]:
        bar.set_color(COLORS["secondary"])

for bar, (_, row) in zip(bars, top20.iterrows()):
    ax.text(
        bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
        f"{row['win_pct']:.1f}%", va="center", ha="left",
        color=COLORS["text"], fontsize=9, fontweight="bold"
    )

ax.set_xlabel("Championship Probability (%)", color=COLORS["text"], fontsize=11)
ax.set_title("🏆 FIFA World Cup 2026 — Win Probability\n(10,000 Monte Carlo Simulations)",
             color=COLORS["text"], fontsize=14, fontweight="bold", pad=15)
ax.tick_params(colors=COLORS["text"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(COLORS["muted"])
ax.spines["left"].set_color(COLORS["muted"])
ax.xaxis.label.set_color(COLORS["text"])

legend_elements = [
    mpatches.Patch(color=COLORS["accent"], label="🥇 Favorite"),
    mpatches.Patch(color=COLORS["secondary"], label="🥈🥉 Top Contenders"),
    mpatches.Patch(color=COLORS["primary"], label="Top 20"),
]
ax.legend(handles=legend_elements, loc="lower right", facecolor=COLORS["bg"],
          labelcolor=COLORS["text"], framealpha=0.7)

plt.tight_layout()
plt.savefig("outputs/chart1_win_probability.png", dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"])
plt.close()
print("   ✅ Saved → outputs/chart1_win_probability.png")

# ─────────────────────────────────────────────
# Chart 2: Group Stage Advancement Heat Map
# ─────────────────────────────────────────────
print("📊 Chart 2: Group advancement heatmap...")

group_order = sorted(df["group"].unique())
heatmap_data = []

for group in group_order:
    group_teams = df[df["group"] == group].sort_values("win_pct", ascending=False)
    for _, row in group_teams.iterrows():
        heatmap_data.append({
            "team": f"[{group}] {row['team']}",
            "Group Advance": row["advance_pct"],
            "Round of 16": row["r16_pct"],
            "Quarter-Final": row["qf_pct"],
            "Semi-Final": row["sf_pct"],
            "Final": row["final_pct"],
            "Champion": row["win_pct"],
        })

hm_df = pd.DataFrame(heatmap_data).set_index("team")

fig, ax = plt.subplots(figsize=(10, 18))
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

sns.heatmap(
    hm_df, annot=True, fmt=".0f", cmap="YlOrRd",
    linewidths=0.3, linecolor="#333333",
    cbar_kws={"label": "Probability (%)", "shrink": 0.5},
    ax=ax, annot_kws={"size": 7},
)

ax.set_title("FIFA World Cup 2026 — Stage Advancement Probabilities (%)",
             color=COLORS["text"], fontsize=12, fontweight="bold", pad=15)
ax.tick_params(colors=COLORS["text"], labelsize=8)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
plt.xticks(rotation=20, ha="left")

plt.tight_layout()
plt.savefig("outputs/chart2_group_heatmap.png", dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"])
plt.close()
print("   ✅ Saved → outputs/chart2_group_heatmap.png")

# ─────────────────────────────────────────────
# Chart 3: Stage-by-stage funnel for top 10
# ─────────────────────────────────────────────
print("📊 Chart 3: Stage funnel for top 10 teams...")

top10 = df.head(10).copy()
stages = ["advance_pct", "r16_pct", "qf_pct", "sf_pct", "final_pct", "win_pct"]
stage_labels = ["Advance", "R16", "QF", "SF", "Final", "🏆 Win"]

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["card"])

x = np.arange(len(stages))
n = len(top10)
width = 0.07

palette = plt.cm.tab10(np.linspace(0, 1, n))

for i, (_, row) in enumerate(top10.iterrows()):
    vals = [row[s] for s in stages]
    offset = (i - n / 2) * width + width / 2
    bars = ax.bar(x + offset, vals, width, label=row["team"],
                  color=palette[i], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(stage_labels, color=COLORS["text"], fontsize=11)
ax.set_ylabel("Probability (%)", color=COLORS["text"])
ax.set_title("Top 10 Teams — Stage-by-Stage Tournament Progression",
             color=COLORS["text"], fontsize=13, fontweight="bold")
ax.tick_params(colors=COLORS["text"])
ax.legend(loc="upper right", facecolor=COLORS["bg"], labelcolor=COLORS["text"],
          fontsize=8, ncol=2, framealpha=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(COLORS["muted"])
ax.spines["left"].set_color(COLORS["muted"])

plt.tight_layout()
plt.savefig("outputs/chart3_stage_funnel.png", dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"])
plt.close()
print("   ✅ Saved → outputs/chart3_stage_funnel.png")

# ─────────────────────────────────────────────
# Chart 4: Elo vs Win Probability scatter
# ─────────────────────────────────────────────
print("📊 Chart 4: Elo rating vs Win probability...")

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["card"])

scatter = ax.scatter(df["elo"], df["win_pct"],
                     c=df["win_pct"], cmap="YlOrRd",
                     s=80, alpha=0.85, edgecolors="#333", linewidth=0.5)

# Label top 10
for _, row in df.head(10).iterrows():
    ax.annotate(row["team"], (row["elo"], row["win_pct"]),
                textcoords="offset points", xytext=(5, 4),
                color=COLORS["text"], fontsize=7.5)

plt.colorbar(scatter, ax=ax, label="Win Probability (%)")
ax.set_xlabel("Elo Rating", color=COLORS["text"], fontsize=11)
ax.set_ylabel("Championship Win Probability (%)", color=COLORS["text"], fontsize=11)
ax.set_title("Elo Rating vs Championship Probability",
             color=COLORS["text"], fontsize=13, fontweight="bold")
ax.tick_params(colors=COLORS["text"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(COLORS["muted"])
ax.spines["left"].set_color(COLORS["muted"])

plt.tight_layout()
plt.savefig("outputs/chart4_elo_vs_winpct.png", dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"])
plt.close()
print("   ✅ Saved → outputs/chart4_elo_vs_winpct.png")

print()
print("=" * 50)
print("✅ STEP 5 COMPLETE — All charts saved to outputs/")
print()
print("   chart1_win_probability.png   — Top 20 win chances")
print("   chart2_group_heatmap.png     — All teams, all stages")
print("   chart3_stage_funnel.png      — Top 10 progression")
print("   chart4_elo_vs_winpct.png     — Elo vs probability")
print()
print("   Optional: Run python step6_predict_match.py")
print("             to predict a specific match interactively")
print("=" * 50)
