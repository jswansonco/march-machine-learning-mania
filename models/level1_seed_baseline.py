"""
Level 1: Seed Baseline Model
=============================
The simplest useful March Madness predictor.

IDEA: In the NCAA tournament, every team gets a "seed" from 1 (best) to 16 (worst).
Historically, lower seeds (better teams) beat higher seeds at very predictable rates.
We simply look at all past tournament games and ask:
  "When a seed X played a seed Y, what % of the time did X win?"
Then we use that historical win rate as our prediction.

For teams WITHOUT a seed (not in the tournament), we assign a default seed of 16
(assuming they're roughly as weak as the weakest tournament teams).
"""

import pandas as pd
import numpy as np

# ============================================================
# STEP 1: Load the data we need
# ============================================================
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

DATA = "data/"

# Men's data
m_seeds = pd.read_csv(f"{DATA}MNCAATourneySeeds.csv")
m_tourney = pd.read_csv(f"{DATA}MNCAATourneyCompactResults.csv")

# Women's data
w_seeds = pd.read_csv(f"{DATA}WNCAATourneySeeds.csv")
w_tourney = pd.read_csv(f"{DATA}WNCAATourneyCompactResults.csv")

# The submission template (tells us what matchups to predict)
submission = pd.read_csv(f"{DATA}SampleSubmissionStage2.csv")

print(f"Men's tournament games:   {len(m_tourney)}")
print(f"Women's tournament games: {len(w_tourney)}")
print(f"Matchups to predict:      {len(submission)}")
print()

# ============================================================
# STEP 2: Extract the seed NUMBER from the seed string
# ============================================================
# Seeds look like "W01", "X16", "Y11a" etc.
# The letter = region (W/X/Y/Z), the digits = seed number (1-16)
# We only care about the number.

print("=" * 60)
print("STEP 2: Parsing seed numbers...")
print("=" * 60)

def parse_seed_number(seed_str):
    """Extract the numeric seed (1-16) from a string like 'W01' or 'X16a'."""
    return int(seed_str[1:3])

# Combine men's and women's seeds into one table
all_seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)
all_seeds["SeedNum"] = all_seeds["Seed"].apply(parse_seed_number)

print("Sample seeds:")
print(all_seeds.head(10).to_string(index=False))
print(f"\nTotal seed records: {len(all_seeds)}")
print(f"Seed number range: {all_seeds['SeedNum'].min()} to {all_seeds['SeedNum'].max()}")
print()

# ============================================================
# STEP 3: For every past tournament game, find each team's seed
# ============================================================
print("=" * 60)
print("STEP 3: Matching tournament games to seeds...")
print("=" * 60)

# Combine men's and women's tournament results
all_tourney = pd.concat([m_tourney, w_tourney], ignore_index=True)

# Join winner's seed
all_tourney = all_tourney.merge(
    all_seeds[["Season", "TeamID", "SeedNum"]],
    left_on=["Season", "WTeamID"],
    right_on=["Season", "TeamID"],
    how="left"
).rename(columns={"SeedNum": "WSeed"}).drop(columns=["TeamID"])

# Join loser's seed
all_tourney = all_tourney.merge(
    all_seeds[["Season", "TeamID", "SeedNum"]],
    left_on=["Season", "LTeamID"],
    right_on=["Season", "TeamID"],
    how="left"
).rename(columns={"SeedNum": "LSeed"}).drop(columns=["TeamID"])

print(f"Tournament games with seeds: {all_tourney.dropna(subset=['WSeed','LSeed']).shape[0]}")
print(f"Tournament games missing seeds: {all_tourney[['WSeed','LSeed']].isna().any(axis=1).sum()}")
print()

# Drop any games where we couldn't find a seed (shouldn't happen, but just in case)
all_tourney = all_tourney.dropna(subset=["WSeed", "LSeed"])
all_tourney["WSeed"] = all_tourney["WSeed"].astype(int)
all_tourney["LSeed"] = all_tourney["LSeed"].astype(int)

# ============================================================
# STEP 4: Build the seed matchup win-rate table
# ============================================================
# For each game, the WINNER had seed WSeed and LOSER had seed LSeed.
# We want to know: when seed A plays seed B, how often does A win?
# We'll organize it so "better seed" (lower number) is always first.

print("=" * 60)
print("STEP 4: Building seed vs seed win-rate table...")
print("=" * 60)

# For each game, record it from the perspective of the better seed
records = []
for _, row in all_tourney.iterrows():
    wseed, lseed = row["WSeed"], row["LSeed"]

    # "Better" seed = lower number
    better_seed = min(wseed, lseed)
    worse_seed = max(wseed, lseed)

    # Did the better seed win? (winner's seed == the better seed)
    better_seed_won = 1 if wseed <= lseed else 0

    records.append({
        "BetterSeed": better_seed,
        "WorseSeed": worse_seed,
        "BetterSeedWon": better_seed_won
    })

matchups_df = pd.DataFrame(records)

# Calculate win rate for each seed matchup
seed_winrates = (
    matchups_df
    .groupby(["BetterSeed", "WorseSeed"])["BetterSeedWon"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "BetterSeedWinRate", "count": "NumGames"})
    .reset_index()
)

# Show the most common matchups (these are the Round 1 games)
print("\nMost common seed matchups (Round 1 games):")
round1 = seed_winrates[
    seed_winrates["BetterSeed"] + seed_winrates["WorseSeed"] == 17
].sort_values("BetterSeed")

for _, row in round1.iterrows():
    s1, s2 = int(row["BetterSeed"]), int(row["WorseSeed"])
    wr = row["BetterSeedWinRate"]
    n = int(row["NumGames"])
    bar = "#" * int(wr * 40)
    print(f"  {s1:2d} vs {s2:2d}: {wr:.1%} win rate ({n:3d} games)  {bar}")

print()

# ============================================================
# STEP 5: Create a lookup + fallback for ANY seed matchup
# ============================================================
# Not all seed matchups have happened (e.g., 1 vs 2 is rare).
# For missing matchups, we'll estimate based on seed difference.

print("=" * 60)
print("STEP 5: Building prediction lookup...")
print("=" * 60)

# Build a quick lookup dictionary: (better_seed, worse_seed) -> win_rate
seed_lookup = {}
for _, row in seed_winrates.iterrows():
    key = (int(row["BetterSeed"]), int(row["WorseSeed"]))
    seed_lookup[key] = row["BetterSeedWinRate"]

# For missing matchups, fit a simple model based on seed difference.
# Logistic-ish curve: the bigger the seed gap, the more likely the better seed wins.
matchups_df["SeedDiff"] = matchups_df["WorseSeed"] - matchups_df["BetterSeed"]

# Group by seed difference and get average win rate
diff_rates = (
    matchups_df
    .groupby("SeedDiff")["BetterSeedWon"]
    .mean()
    .to_dict()
)

print("Win rate by seed difference:")
for diff in sorted(diff_rates.keys()):
    wr = diff_rates[diff]
    print(f"  Seed gap {diff:2d}: {wr:.1%}")

# Fallback: use seed difference if exact matchup not seen
# Default to 0.5 if somehow both are same seed
DEFAULT_SEED = 16  # for teams not in tournament

print()

# ============================================================
# STEP 6: Generate predictions for the submission file
# ============================================================
print("=" * 60)
print("STEP 6: Generating predictions...")
print("=" * 60)

# Get 2026 seeds (we may not have them yet since it's before Selection Sunday)
seeds_2026 = all_seeds[all_seeds["Season"] == 2026]
seed_map_2026 = dict(zip(seeds_2026["TeamID"], seeds_2026["SeedNum"]))
print(f"Teams with 2026 seeds: {len(seed_map_2026)}")
print(f"(If 0, that's expected — seeds aren't assigned until Selection Sunday)")
print()

# Parse the submission file and make predictions
def predict_matchup(row_id):
    """Given an ID like '2026_1101_1104', predict P(team 1101 wins)."""
    parts = row_id.split("_")
    season = int(parts[0])
    team1 = int(parts[1])  # lower ID
    team2 = int(parts[2])  # higher ID

    # Look up seeds (default to 16 if unknown)
    seed1 = seed_map_2026.get(team1, DEFAULT_SEED)
    seed2 = seed_map_2026.get(team2, DEFAULT_SEED)

    if seed1 == seed2:
        return 0.5  # same seed = coin flip

    # Figure out who has the better (lower) seed
    better_seed = min(seed1, seed2)
    worse_seed = max(seed1, seed2)

    # Look up historical win rate
    if (better_seed, worse_seed) in seed_lookup:
        better_win_prob = seed_lookup[(better_seed, worse_seed)]
    else:
        # Fallback: use seed difference
        diff = worse_seed - better_seed
        if diff in diff_rates:
            better_win_prob = diff_rates[diff]
        else:
            better_win_prob = 0.5

    # Return probability that TEAM1 (lower ID) wins
    # If team1 has the better seed, return the win rate directly
    # If team2 has the better seed, return (1 - win rate)
    if seed1 < seed2:
        return better_win_prob
    elif seed1 > seed2:
        return 1 - better_win_prob
    else:
        return 0.5

submission["Pred"] = submission["ID"].apply(predict_matchup)

print(f"Predictions generated: {len(submission)}")
print(f"Prediction range: {submission['Pred'].min():.4f} to {submission['Pred'].max():.4f}")
print(f"Mean prediction:  {submission['Pred'].mean():.4f}")
print()

# Show some sample predictions
print("Sample predictions:")
print(submission.head(10).to_string(index=False))
print()

# ============================================================
# STEP 7: Save the submission
# ============================================================
output_path = "submission_level1_seed_baseline.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to: {output_path}")
print(f"Total rows: {len(submission)}")
print()

# ============================================================
# STEP 8: Evaluate on historical data (2022-2025 tournaments)
# ============================================================
print("=" * 60)
print("STEP 8: Evaluating on historical tournaments (2022-2025)...")
print("=" * 60)

# Load Stage 1 (historical matchups for validation)
stage1 = pd.read_csv(f"{DATA}SampleSubmissionStage1.csv")

# Get actual tournament results for 2022-2025
eval_tourney = all_tourney[all_tourney["Season"].isin([2022, 2023, 2024, 2025])].copy()

# Build a set of actual results: for each game, record (season, lower_id, higher_id, did_lower_win)
actual_results = {}
for _, row in eval_tourney.iterrows():
    season = row["Season"]
    wteam = row["WTeamID"]
    lteam = row["LTeamID"]
    lower_id = min(wteam, lteam)
    higher_id = max(wteam, lteam)
    lower_won = 1 if wteam == lower_id else 0
    key = f"{season}_{lower_id}_{higher_id}"
    actual_results[key] = lower_won

print(f"Actual tournament games (2022-2025): {len(actual_results)}")

# For each historical season, build seed map and predict
brier_scores = []
for season in [2022, 2023, 2024, 2025]:
    season_seeds = all_seeds[all_seeds["Season"] == season]
    season_seed_map = dict(zip(season_seeds["TeamID"], season_seeds["SeedNum"]))

    season_results = {k: v for k, v in actual_results.items() if k.startswith(str(season))}

    for game_id, actual_outcome in season_results.items():
        parts = game_id.split("_")
        team1, team2 = int(parts[1]), int(parts[2])

        seed1 = season_seed_map.get(team1, DEFAULT_SEED)
        seed2 = season_seed_map.get(team2, DEFAULT_SEED)

        if seed1 == seed2:
            pred = 0.5
        else:
            better_seed = min(seed1, seed2)
            worse_seed = max(seed1, seed2)

            if (better_seed, worse_seed) in seed_lookup:
                better_win_prob = seed_lookup[(better_seed, worse_seed)]
            else:
                diff = worse_seed - better_seed
                better_win_prob = diff_rates.get(diff, 0.5)

            if seed1 < seed2:
                pred = better_win_prob
            elif seed1 > seed2:
                pred = 1 - better_win_prob
            else:
                pred = 0.5

        brier = (pred - actual_outcome) ** 2
        brier_scores.append({"Season": season, "GameID": game_id, "Pred": pred,
                            "Actual": actual_outcome, "Brier": brier})

brier_df = pd.DataFrame(brier_scores)

print(f"\nBrier Score by Season:")
for season in [2022, 2023, 2024, 2025]:
    season_brier = brier_df[brier_df["Season"] == season]["Brier"].mean()
    n_games = len(brier_df[brier_df["Season"] == season])
    print(f"  {season}: {season_brier:.4f} ({n_games} games)")

overall_brier = brier_df["Brier"].mean()
print(f"\n  OVERALL: {overall_brier:.4f}")
print()

# For reference: a "coin flip" model (always predicting 0.5) scores 0.25
print("For reference:")
print("  Always predicting 0.5 (coin flip): Brier = 0.2500")
print(f"  Our seed baseline:                  Brier = {overall_brier:.4f}")
print(f"  Improvement over coin flip:         {((0.25 - overall_brier) / 0.25) * 100:.1f}%")
print()
print("Done! 🏀")
