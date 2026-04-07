"""
Blend Weight Analysis: XGBoost vs Torvik
==========================================
Tests different blend ratios and evaluates using:
1. Pinnacle odds as market benchmark (men's only)
2. Prediction distribution analysis
3. Self-consistency checks (are blended predictions well-behaved?)
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

DATA = "data/"

# ============================================================
# Load predictions
# ============================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

# Level 9 (pure XGBoost) submission
xgb_sub = pd.read_csv("submission_level9_special_sauce.csv")

# Level 11 submission (already blended at 70/30)
# We need the raw XGBoost and Torvik preds separately
# Let's recompute Torvik predictions standalone

# Load Torvik data
tv_m = pd.read_csv(f"{DATA}torvik_mens_2026.csv")
tv_w = pd.read_csv(f"{DATA}torvik_womens_2026.csv")

# Team name matching
m_teams = pd.read_csv(f"{DATA}MTeams.csv")
w_teams = pd.read_csv(f"{DATA}WTeams.csv")
m_spell = pd.read_csv(f"{DATA}MTeamSpellings.csv", encoding="latin1")
w_spell = pd.read_csv(f"{DATA}WTeamSpellings.csv", encoding="latin1")

m_name_to_id = dict(zip(m_teams["TeamName"].str.lower(), m_teams["TeamID"]))
w_name_to_id = dict(zip(w_teams["TeamName"].str.lower(), w_teams["TeamID"]))
for _, row in m_spell.iterrows():
    m_name_to_id[str(row["TeamNameSpelling"]).lower().strip()] = row["TeamID"]
for _, row in w_spell.iterrows():
    w_name_to_id[str(row["TeamNameSpelling"]).lower().strip()] = row["TeamID"]

manual_map = {
    "tarleton st.": "tarleton st", "ut rio grande valley": "utrgv",
    "illinois chicago": "uic", "texas a&m corpus chris": "texas a&m cc",
    "southeast missouri st.": "se missouri st", "tennessee martin": "ut martin",
    "queens": "queens nc", "bethune cookman": "bethune-cookman",
    "arkansas pine bluff": "ark pine bluff", "cal st. bakersfield": "cal st bakersfield",
    "louisiana monroe": "ul monroe", "saint francis": "st francis pa",
    "mississippi valley st.": "ms valley st",
}

def match_torvik(tv_df, name_to_id):
    tv_df = tv_df.copy()
    team_ids = []
    for _, row in tv_df.iterrows():
        name = row["team"].lower().strip()
        tid = name_to_id.get(name) or name_to_id.get(manual_map.get(name, ""), None)
        team_ids.append(tid)
    tv_df["TeamID"] = team_ids
    return tv_df

tv_m = match_torvik(tv_m, m_name_to_id)
tv_w = match_torvik(tv_w, w_name_to_id)

torvik_lookup = {}
for _, row in pd.concat([tv_m, tv_w]).iterrows():
    if pd.notna(row["TeamID"]):
        torvik_lookup[int(row["TeamID"])] = {
            "adjoe": row["adjoe"], "adjde": row["adjde"], "adjt": row["adjt"],
        }

# Compute Torvik predictions for all matchups
def torvik_prob(t1, t2, std_dev=11.0):
    tv1, tv2 = torvik_lookup.get(t1), torvik_lookup.get(t2)
    if tv1 is None or tv2 is None:
        return None
    eff_margin = (tv1["adjoe"] - tv2["adjde"]) - (tv2["adjoe"] - tv1["adjde"])
    avg_tempo = (tv1["adjt"] + tv2["adjt"]) / 2
    spread = (eff_margin / 100) * avg_tempo
    return np.clip(norm.cdf(spread / std_dev), 0.01, 0.99)

sub = xgb_sub.copy()
sub["T1"] = sub["ID"].apply(lambda x: int(x.split("_")[1]))
sub["T2"] = sub["ID"].apply(lambda x: int(x.split("_")[2]))
sub["xgb"] = sub["Pred"]
sub["torvik"] = sub.apply(lambda r: torvik_prob(r["T1"], r["T2"]), axis=1)

has_torvik = sub["torvik"].notna()
print(f"Matchups with both predictions: {has_torvik.sum():,} / {len(sub):,}")
print()

# ============================================================
# Load Pinnacle odds for market benchmark
# ============================================================
print("=" * 60)
print("Loading Pinnacle odds as market benchmark...")
print("=" * 60)

odds = pd.read_csv("odds_data/ncaa_main_lines.csv")

# Get the LATEST spread per game (closest to tip-off = sharpest line)
odds["timestamp"] = pd.to_datetime(odds["timestamp"])
latest_odds = odds.sort_values("timestamp").groupby("game_link").last().reset_index()
latest_odds = latest_odds[latest_odds["team1_spread"].notna()]

print(f"Games with spreads: {len(latest_odds)}")

# Match Pinnacle team names to TeamIDs
def match_pinnacle_name(name, name_to_id):
    name_lower = name.lower().strip()
    tid = name_to_id.get(name_lower)
    if tid is None:
        mapped = manual_map.get(name_lower, name_lower)
        tid = name_to_id.get(mapped)
    return tid

latest_odds["T1_ID"] = latest_odds["team1"].apply(lambda x: match_pinnacle_name(x, m_name_to_id))
latest_odds["T2_ID"] = latest_odds["team2"].apply(lambda x: match_pinnacle_name(x, m_name_to_id))

matched_odds = latest_odds[latest_odds["T1_ID"].notna() & latest_odds["T2_ID"].notna()].copy()
matched_odds["T1_ID"] = matched_odds["T1_ID"].astype(int)
matched_odds["T2_ID"] = matched_odds["T2_ID"].astype(int)

# Convert Pinnacle spread to implied probability
# Spread is from team1's perspective: negative = team1 favored
# We need to orient to lower ID (matching submission format)
def pinnacle_to_prob(row, std_dev=11.0):
    t1_id, t2_id = row["T1_ID"], row["T2_ID"]
    spread = row["team1_spread"]  # team1's spread (negative = favored)
    # Convert to implied margin for team1
    margin = -spread  # if spread is -5, team1 favored by 5
    prob_t1 = norm.cdf(margin / std_dev)

    # Orient to lower ID
    lower_id = min(t1_id, t2_id)
    if lower_id == t1_id:
        return lower_id, max(t1_id, t2_id), prob_t1
    else:
        return lower_id, max(t1_id, t2_id), 1 - prob_t1

pinnacle_preds = []
for _, row in matched_odds.iterrows():
    lower, higher, prob = pinnacle_to_prob(row)
    pinnacle_preds.append({"T1": lower, "T2": higher, "pinnacle_prob": prob})

pin_df = pd.DataFrame(pinnacle_preds)
# Average across multiple games between same teams
pin_df = pin_df.groupby(["T1", "T2"])["pinnacle_prob"].mean().reset_index()

print(f"Pinnacle matchups matched: {len(pin_df)}")

# Merge with our predictions
sub = sub.merge(pin_df, on=["T1", "T2"], how="left")
has_pinnacle = sub["pinnacle_prob"].notna()
print(f"Matchups with Pinnacle benchmark: {has_pinnacle.sum()}")
print()

# ============================================================
# Test blend weights
# ============================================================
print("=" * 60)
print("Testing blend weights...")
print("=" * 60)

# Load seeds for tournament team filtering
m_seeds = pd.read_csv(f"{DATA}MNCAATourneySeeds.csv")
w_seeds = pd.read_csv(f"{DATA}WNCAATourneySeeds.csv")
all_seeds = pd.concat([m_seeds, w_seeds])
tourney_teams_2026 = set(all_seeds[all_seeds["Season"] == 2026]["TeamID"])

# Filter to tournament matchups with Torvik available
tourney_mask = sub["T1"].isin(tourney_teams_2026) & sub["T2"].isin(tourney_teams_2026) & has_torvik
tourney_sub = sub[tourney_mask].copy()
print(f"Tournament matchups with Torvik: {len(tourney_sub):,}")

# Also filter to those with Pinnacle
pin_tourney = tourney_sub[tourney_sub["pinnacle_prob"].notna()].copy()
print(f"Tournament matchups with Pinnacle: {len(pin_tourney):,}")
print()

weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print(f"{'Weight (XGB/TV)':<18s} {'MSE vs Pinnacle':>16s} {'Avg |diff| vs Pin':>18s} {'Pred StdDev':>12s} {'Mean Pred':>10s}")
print("-" * 78)

best_mse = 1.0
best_weight = 0.7

for w in weights:
    blend = w * tourney_sub["xgb"] + (1 - w) * tourney_sub["torvik"]

    # MSE vs Pinnacle (where available)
    pin_mask = tourney_sub["pinnacle_prob"].notna()
    if pin_mask.sum() > 0:
        blend_pin = w * tourney_sub.loc[pin_mask, "xgb"] + (1 - w) * tourney_sub.loc[pin_mask, "torvik"]
        mse_vs_pin = np.mean((blend_pin - tourney_sub.loc[pin_mask, "pinnacle_prob"]) ** 2)
        mad_vs_pin = np.mean(np.abs(blend_pin - tourney_sub.loc[pin_mask, "pinnacle_prob"]))
    else:
        mse_vs_pin = float("nan")
        mad_vs_pin = float("nan")

    std = blend.std()
    mean = blend.mean()

    marker = " ←" if w == 0.7 else ""
    print(f"  {w:.0%} XGB / {1-w:.0%} TV    {mse_vs_pin:>14.6f}   {mad_vs_pin:>16.4f}   {std:>10.4f}   {mean:>8.4f}{marker}")

    if mse_vs_pin < best_mse:
        best_mse = mse_vs_pin
        best_weight = w

print()
print(f"Best weight by MSE vs Pinnacle: {best_weight:.0%} XGB / {1-best_weight:.0%} Torvik")
print()

# ============================================================
# Also test Torvik std_dev parameter
# ============================================================
print("=" * 60)
print("Testing Torvik std_dev (spread → probability conversion)...")
print("=" * 60)

print(f"{'StdDev':<10s} {'MSE vs Pinnacle':>16s}")
print("-" * 30)

best_std_mse = 1.0
best_std = 11.0

for std_dev in [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]:
    # Recompute Torvik with different std_dev
    tv_preds = tourney_sub.apply(
        lambda r: torvik_prob(r["T1"], r["T2"], std_dev=std_dev), axis=1
    )
    pin_mask = tourney_sub["pinnacle_prob"].notna()
    blend_pin = best_weight * tourney_sub.loc[pin_mask, "xgb"] + (1 - best_weight) * tv_preds[pin_mask]
    mse = np.mean((blend_pin - tourney_sub.loc[pin_mask, "pinnacle_prob"]) ** 2)
    print(f"  {std_dev:>5.1f}    {mse:>14.6f}")
    if mse < best_std_mse:
        best_std_mse = mse
        best_std = std_dev

print()
print(f"Best std_dev: {best_std}")
print()

# ============================================================
# Show some calibration examples
# ============================================================
print("=" * 60)
print("Calibration check: XGBoost vs Torvik vs Pinnacle (sample matchups)...")
print("=" * 60)

team_name_map = dict(zip(
    pd.concat([m_teams[["TeamID","TeamName"]], w_teams[["TeamID","TeamName"]]])["TeamID"],
    pd.concat([m_teams[["TeamID","TeamName"]], w_teams[["TeamID","TeamName"]]])["TeamName"]
))

sample = pin_tourney.nlargest(15, "pinnacle_prob").head(15)
print(f"{'Matchup':<35s} {'XGBoost':>8s} {'Torvik':>8s} {'Pinnacle':>9s} {'Blend70':>8s}")
print("-" * 72)
for _, row in sample.iterrows():
    n1 = team_name_map.get(row["T1"], str(int(row["T1"])))
    n2 = team_name_map.get(row["T2"], str(int(row["T2"])))
    matchup = f"{n1} vs {n2}"[:34]
    blend = 0.7 * row["xgb"] + 0.3 * row["torvik"]
    print(f"  {matchup:<33s} {row['xgb']:>7.1%} {row['torvik']:>7.1%} {row['pinnacle_prob']:>8.1%} {blend:>7.1%}")

print()

# Bottom (closest games)
sample2 = pin_tourney.iloc[(pin_tourney["pinnacle_prob"] - 0.5).abs().argsort()[:10]]
print("Closest to 50/50 (most uncertain):")
print(f"{'Matchup':<35s} {'XGBoost':>8s} {'Torvik':>8s} {'Pinnacle':>9s} {'Blend70':>8s}")
print("-" * 72)
for _, row in sample2.iterrows():
    n1 = team_name_map.get(row["T1"], str(int(row["T1"])))
    n2 = team_name_map.get(row["T2"], str(int(row["T2"])))
    matchup = f"{n1} vs {n2}"[:34]
    blend = 0.7 * row["xgb"] + 0.3 * row["torvik"]
    print(f"  {matchup:<33s} {row['xgb']:>7.1%} {row['torvik']:>7.1%} {row['pinnacle_prob']:>8.1%} {blend:>7.1%}")

print()
print("Done!")
