"""
Generate point-spread submission for Logistic Brier competition.
Uses Level 12's trained LOSO XGBoost models to output raw point differentials
(before spline calibration).
"""
# Just re-run level12 but save raw margins instead of probabilities
# We exec the level12 script up to the prediction step, then output margins

print("Running Level 12 to generate spread predictions...")
print("(This re-runs the full pipeline — takes ~2 minutes)")
print()

# We need to modify the output section of level12
# Easiest: run level12 as-is but intercept the margin predictions

import subprocess, sys

# Add a flag to level12 to output margins
# Actually, let's just inline the critical part

exec(open("level12_torvik_features.py").read())

# At this point, models{}, features, and X with all features are loaded
# Re-generate predictions as raw margins (not probabilities)

from xgboost import DMatrix
import numpy as np
import pandas as pd

dtest = DMatrix(X[features].values, feature_names=features)
all_margins = []
for oof_season in seasons:
    margin_preds = models[oof_season].predict(dtest)
    all_margins.append(margin_preds)

X["Spread"] = np.mean(all_margins, axis=0)

# Also do the 80/20 blend with Level 9 margins
# Level 9 margins aren't available, so just use Level 12 margins directly
# The blend was for probability space — for spreads, Level 12 alone is cleaner

spread_sub = X[["ID"]].copy()
spread_sub["Pred"] = X["Spread"]

print(f"\nSpread predictions:")
print(f"  Range: {spread_sub['Pred'].min():.1f} to {spread_sub['Pred'].max():.1f}")
print(f"  Mean:  {spread_sub['Pred'].mean():.1f}")
print(f"  Std:   {spread_sub['Pred'].std():.1f}")
print()

# Show some examples
m_seeds_2026 = pd.read_csv("data/MNCAATourneySeeds.csv")
m_seeds_2026 = m_seeds_2026[m_seeds_2026["Season"] == 2026]
seed_map = dict(zip(m_seeds_2026["TeamID"], m_seeds_2026["Seed"].apply(lambda x: int(x[1:3]))))

print("Sample tournament matchup spreads:")
tourney_spreads = spread_sub.copy()
tourney_spreads["T1"] = tourney_spreads["ID"].apply(lambda x: int(x.split("_")[1]))
tourney_spreads["T2"] = tourney_spreads["ID"].apply(lambda x: int(x.split("_")[2]))
tourney_spreads["T1_seed"] = tourney_spreads["T1"].map(seed_map)
tourney_spreads["T2_seed"] = tourney_spreads["T2"].map(seed_map)

ones = tourney_spreads[(tourney_spreads["T1_seed"] == 1) & (tourney_spreads["T2_seed"] == 1)]
for _, row in ones.iterrows():
    n1 = team_name_map.get(row["T1"], "?")
    n2 = team_name_map.get(row["T2"], "?")
    spread = row["Pred"]
    favored = n1 if spread > 0 else n2
    print(f"  {n1} vs {n2}: {n1} {spread:+.1f} (favored: {favored} by {abs(spread):.1f})")

print("\n1-seed vs 16-seed:")
s116 = tourney_spreads[(tourney_spreads["T1_seed"] == 1) & (tourney_spreads["T2_seed"] == 16)]
for _, row in s116.head(4).iterrows():
    n1 = team_name_map.get(row["T1"], "?")
    n2 = team_name_map.get(row["T2"], "?")
    print(f"  {n1} vs {n2}: {n1} {row['Pred']:+.1f}")

spread_sub[["ID", "Pred"]].to_csv("submission_logistic_brier.csv", index=False)
print(f"\nSaved to: submission_logistic_brier.csv")
print("Done!")
