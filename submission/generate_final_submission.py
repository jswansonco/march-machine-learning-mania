"""
Generate Final Submission
==========================
Produces both competition submissions:
  1. submission_level12_blend80.csv — 80% Level 12 + 20% Level 9 (Brier + Semi-Spherical)
  2. submission_logistic_brier.csv  — Raw Level 12 point spreads (Logistic Brier)

Prerequisites:
  - Run level12_torvik_features.py first (produces submission_level12_torvik_features.csv)
  - Run level9_special_sauce.py first (produces submission_level9_special_sauce.csv)

Or run this script with --full to execute both models from scratch.
"""

import pandas as pd
import numpy as np
import sys
import os

FULL_RUN = "--full" in sys.argv

if FULL_RUN:
    print("=" * 60)
    print("Running Level 9 (special sauce)...")
    print("=" * 60)
    os.system(f"{sys.executable} level9_special_sauce.py")

    print()
    print("=" * 60)
    print("Running Level 12 (Torvik features)...")
    print("=" * 60)
    os.system(f"{sys.executable} level12_torvik_features.py")
    print()

# ============================================================
# Load both model outputs
# ============================================================
print("=" * 60)
print("Generating final submissions...")
print("=" * 60)

l9_path = "submission_level9_special_sauce.csv"
l12_path = "submission_level12_torvik_features.csv"

if not os.path.exists(l9_path):
    print(f"ERROR: {l9_path} not found. Run level9_special_sauce.py first, or use --full flag.")
    sys.exit(1)
if not os.path.exists(l12_path):
    print(f"ERROR: {l12_path} not found. Run level12_torvik_features.py first, or use --full flag.")
    sys.exit(1)

l9 = pd.read_csv(l9_path)
l12 = pd.read_csv(l12_path)

print(f"Level 9:  {len(l9):,} rows, pred range [{l9['Pred'].min():.4f}, {l9['Pred'].max():.4f}]")
print(f"Level 12: {len(l12):,} rows, pred range [{l12['Pred'].min():.4f}, {l12['Pred'].max():.4f}]")
print()

# ============================================================
# 1. Brier / Semi-Spherical submission (80/20 blend)
# ============================================================
BLEND_WEIGHT = 0.8  # 80% Level 12, 20% Level 9

both = l9.merge(l12, on="ID", suffixes=("_l9", "_l12"))
both["Pred"] = BLEND_WEIGHT * both["Pred_l12"] + (1 - BLEND_WEIGHT) * both["Pred_l9"]
both["Pred"] = both["Pred"].clip(0.01, 0.99)

brier_sub = both[["ID", "Pred"]]
brier_path = "submission_level12_blend80.csv"
brier_sub.to_csv(brier_path, index=False)

print(f"Brier/Semi-Spherical submission:")
print(f"  File: {brier_path}")
print(f"  Blend: {BLEND_WEIGHT:.0%} Level 12 / {1-BLEND_WEIGHT:.0%} Level 9")
print(f"  Rows: {len(brier_sub):,}")
print(f"  Range: [{brier_sub['Pred'].min():.4f}, {brier_sub['Pred'].max():.4f}]")
print(f"  Mean: {brier_sub['Pred'].mean():.4f}")
print()

# ============================================================
# 2. Logistic Brier submission (raw point spreads from Level 12)
# ============================================================
# Level 12 outputs probabilities via spline calibration.
# For logistic brier we need raw point-diff margins.
# We need to re-run Level 12's LOSO models on the submission data.
#
# Since the models aren't saved to disk, we invert the spline
# to approximate the original margins. The spline maps margin -> prob,
# so we find the margin that produces each probability.

from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq

# Reconstruct spline from Level 12's probability output
# We know the spline maps [-25, 25] -> [0.01, 0.99] monotonically
# We can invert it numerically

# First, check if the logistic brier file already exists from generate_spread_submission.py
logistic_path = "submission_logistic_brier.csv"
if os.path.exists(logistic_path):
    logistic_sub = pd.read_csv(logistic_path)
    print(f"Logistic Brier submission (pre-existing):")
    print(f"  File: {logistic_path}")
    print(f"  Rows: {len(logistic_sub):,}")
    print(f"  Range: [{logistic_sub['Pred'].min():.1f}, {logistic_sub['Pred'].max():.1f}]")
    print(f"  Mean: {logistic_sub['Pred'].mean():.1f}")
else:
    # Approximate: convert probability back to spread using logistic inverse
    # P = 1/(1+exp(-spread/c)) → spread = -c * ln(1/P - 1)
    # Use c=11 (our spline's approximate scale, not the competition's c=7)
    C_APPROX = 11.0
    spreads = -C_APPROX * np.log(1.0 / l12["Pred"].clip(0.001, 0.999) - 1.0)

    logistic_sub = l12[["ID"]].copy()
    logistic_sub["Pred"] = spreads
    logistic_sub.to_csv(logistic_path, index=False)

    print(f"Logistic Brier submission (approximated from probabilities):")
    print(f"  File: {logistic_path}")
    print(f"  Rows: {len(logistic_sub):,}")
    print(f"  Range: [{logistic_sub['Pred'].min():.1f}, {logistic_sub['Pred'].max():.1f}]")
    print(f"  Mean: {logistic_sub['Pred'].mean():.1f}")
    print(f"  NOTE: For best results, run generate_spread_submission.py which uses actual XGBoost margins.")

print()
print("=" * 60)
print("SUBMISSION FILES READY")
print("=" * 60)
print()
print(f"  1. {brier_path}")
print(f"     → Upload to: March Machine Learning Mania 2026 (Brier)")
print(f"     → Upload to: March Machine Learning Mania 2026 (Semi-Spherical)")
print()
print(f"  2. {logistic_path}")
print(f"     → Upload to: March Mania 2026 - Logistic Brier")
print()
print("Remember to MANUALLY SELECT your final submission on each competition page!")
print()
print("Done!")
