"""
Level 2: Win Percentage Model
==============================
Instead of relying on tournament seeds (which aren't known until Selection Sunday),
we use each team's REGULAR SEASON record to estimate how good they are.

IDEA:
  1. For each team in a season, count wins and losses → win percentage
  2. To predict a matchup, convert the gap in win percentages into a probability
  3. We use logistic regression to learn exactly HOW to convert that gap

WHY THIS IS BETTER THAN LEVEL 1:
  - Works for ALL teams (not just tournament teams with seeds)
  - Can generate real 2026 predictions right now
  - Distinguishes between teams with the same seed

WHAT IS LOGISTIC REGRESSION? (simple explanation)
  Imagine you have a number (like the gap in win percentages between two teams).
  You want to turn that into a probability between 0 and 1.
  Logistic regression learns a formula:  probability = 1 / (1 + e^(-something))
  This creates an S-shaped curve that naturally maps any number to a 0-1 range.
  The "learning" part figures out the best shape of that S-curve from historical data.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

DATA = "data/"

# ============================================================
# STEP 1: Load regular season results
# ============================================================
print("=" * 60)
print("STEP 1: Loading regular season data...")
print("=" * 60)

m_regular = pd.read_csv(f"{DATA}MRegularSeasonCompactResults.csv")
w_regular = pd.read_csv(f"{DATA}WRegularSeasonCompactResults.csv")
all_regular = pd.concat([m_regular, w_regular], ignore_index=True)

print(f"Total regular season games: {len(all_regular):,}")
print(f"Seasons: {all_regular['Season'].min()} to {all_regular['Season'].max()}")
print()

# ============================================================
# STEP 2: Calculate win percentage for every team in every season
# ============================================================
print("=" * 60)
print("STEP 2: Calculating win percentages...")
print("=" * 60)

# Count wins: each row has a WTeamID (winner) and LTeamID (loser)
# So team X's wins = number of times they appear as WTeamID
wins = (
    all_regular
    .groupby(["Season", "WTeamID"])
    .size()
    .reset_index(name="Wins")
    .rename(columns={"WTeamID": "TeamID"})
)

# Count losses: number of times they appear as LTeamID
losses = (
    all_regular
    .groupby(["Season", "LTeamID"])
    .size()
    .reset_index(name="Losses")
    .rename(columns={"LTeamID": "TeamID"})
)

# Merge wins and losses
team_records = wins.merge(losses, on=["Season", "TeamID"], how="outer").fillna(0)
team_records["Wins"] = team_records["Wins"].astype(int)
team_records["Losses"] = team_records["Losses"].astype(int)
team_records["Games"] = team_records["Wins"] + team_records["Losses"]
team_records["WinPct"] = team_records["Wins"] / team_records["Games"]

print(f"Team-season records: {len(team_records):,}")
print(f"\nSample (2025 season, top 10 by win %):")
sample = (
    team_records[team_records["Season"] == 2025]
    .sort_values("WinPct", ascending=False)
    .head(10)
)

# Load team names for display
m_teams = pd.read_csv(f"{DATA}MTeams.csv")
w_teams = pd.read_csv(f"{DATA}WTeams.csv")
all_teams = pd.concat([m_teams[["TeamID", "TeamName"]], w_teams[["TeamID", "TeamName"]]])
team_name_map = dict(zip(all_teams["TeamID"], all_teams["TeamName"]))

for _, row in sample.iterrows():
    name = team_name_map.get(row["TeamID"], "???")
    print(f"  {name:20s}  {int(row['Wins']):2d}-{int(row['Losses']):2d}  ({row['WinPct']:.1%})")
print()

# ============================================================
# STEP 3: Build training data from historical tournament games
# ============================================================
# For each past tournament game, we know:
#   - Who won and who lost
#   - Each team's regular season win percentage going into the tournament
# We'll use the GAP in win percentages as our feature.

print("=" * 60)
print("STEP 3: Building training data from tournament history...")
print("=" * 60)

m_tourney = pd.read_csv(f"{DATA}MNCAATourneyCompactResults.csv")
w_tourney = pd.read_csv(f"{DATA}WNCAATourneyCompactResults.csv")
all_tourney = pd.concat([m_tourney, w_tourney], ignore_index=True)

# Create a lookup: (season, team) -> win_pct
winpct_lookup = {}
for _, row in team_records.iterrows():
    winpct_lookup[(row["Season"], row["TeamID"])] = row["WinPct"]

# For each tournament game, compute the win% gap from team1's perspective
# where team1 = lower TeamID (matching submission format)
training_rows = []
for _, row in all_tourney.iterrows():
    season = row["Season"]
    wteam = row["WTeamID"]
    lteam = row["LTeamID"]

    wpct_w = winpct_lookup.get((season, wteam))
    wpct_l = winpct_lookup.get((season, lteam))

    if wpct_w is None or wpct_l is None:
        continue  # skip if we don't have records

    # Orient to "lower ID" perspective (matching submission format)
    team1 = min(wteam, lteam)  # lower ID
    team2 = max(wteam, lteam)  # higher ID
    team1_won = 1 if wteam == team1 else 0

    wpct_1 = winpct_lookup[(season, team1)]
    wpct_2 = winpct_lookup[(season, team2)]

    training_rows.append({
        "Season": season,
        "WinPctDiff": wpct_1 - wpct_2,  # positive = team1 has better record
        "Team1Won": team1_won
    })

train_df = pd.DataFrame(training_rows)
print(f"Training examples: {len(train_df)}")
print(f"Team1 (lower ID) win rate: {train_df['Team1Won'].mean():.1%}")
print(f"WinPctDiff range: {train_df['WinPctDiff'].min():.3f} to {train_df['WinPctDiff'].max():.3f}")
print()

# ============================================================
# STEP 4: Train logistic regression
# ============================================================
print("=" * 60)
print("STEP 4: Training logistic regression...")
print("=" * 60)

X = train_df[["WinPctDiff"]].values
y = train_df["Team1Won"].values

model = LogisticRegression()
model.fit(X, y)

# Show what the model learned
coef = model.coef_[0][0]
intercept = model.intercept_[0]
print(f"Model learned:")
print(f"  Coefficient (weight on WinPctDiff): {coef:.3f}")
print(f"  Intercept (bias):                   {intercept:.3f}")
print()
print("What this means:")
print("  If two teams have equal win%, prediction ≈ 50%")
print(f"  For every 10% gap in win%, prediction shifts by ~{abs(coef) * 0.1 / 4 * 100:.1f} percentage points")
print()

# Show predictions at various gaps
print("Example predictions:")
for diff in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
    prob = model.predict_proba([[diff]])[0][1]
    if diff < 0:
        print(f"  Team1 win% is {abs(diff)*100:.0f}% WORSE  → P(Team1 wins) = {prob:.1%}")
    elif diff > 0:
        print(f"  Team1 win% is {diff*100:.0f}% BETTER → P(Team1 wins) = {prob:.1%}")
    else:
        print(f"  Teams have EQUAL win%     → P(Team1 wins) = {prob:.1%}")
print()

# ============================================================
# STEP 5: Evaluate on 2022-2025 tournaments
# ============================================================
print("=" * 60)
print("STEP 5: Evaluating on 2022-2025 tournaments...")
print("=" * 60)

# Train on pre-2022 data, test on 2022-2025 (proper time-based split)
eval_years = [2022, 2023, 2024, 2025]

brier_scores = []
for season in eval_years:
    # Re-train on everything BEFORE this season (to avoid data leakage)
    train_mask = train_df["Season"] < season
    X_train = train_df.loc[train_mask, ["WinPctDiff"]].values
    y_train = train_df.loc[train_mask, "Team1Won"].values

    test_mask = train_df["Season"] == season
    X_test = train_df.loc[test_mask, ["WinPctDiff"]].values
    y_test = train_df.loc[test_mask, "Team1Won"].values

    if len(X_train) == 0 or len(X_test) == 0:
        continue

    m = LogisticRegression()
    m.fit(X_train, y_train)
    preds = m.predict_proba(X_test)[:, 1]

    season_brier = np.mean((preds - y_test) ** 2)
    brier_scores.append({"Season": season, "Brier": season_brier, "Games": len(y_test)})
    print(f"  {season}: Brier = {season_brier:.4f} ({len(y_test)} games)")

overall_brier = np.mean([b["Brier"] for b in brier_scores])
print(f"\n  OVERALL: {overall_brier:.4f}")
print()
print("Comparison:")
print(f"  Coin flip (0.5):     Brier = 0.2500")
print(f"  Level 1 (seeds):     Brier = 0.1671")
print(f"  Level 2 (win pct):   Brier = {overall_brier:.4f}")

if overall_brier < 0.1671:
    print(f"  → Level 2 is BETTER by {(0.1671 - overall_brier):.4f}")
else:
    print(f"  → Level 2 is WORSE by {(overall_brier - 0.1671):.4f}")
    print(f"  (This is expected! Seeds contain info about team quality that raw win% misses.)")
    print(f"  (Seeds are curated by a committee who watches every team all season.)")
print()

# ============================================================
# STEP 6: Generate 2026 predictions
# ============================================================
print("=" * 60)
print("STEP 6: Generating 2026 predictions...")
print("=" * 60)

# Train on ALL historical data for final model
X_all = train_df[["WinPctDiff"]].values
y_all = train_df["Team1Won"].values
final_model = LogisticRegression()
final_model.fit(X_all, y_all)

# Load submission template
submission = pd.read_csv(f"{DATA}SampleSubmissionStage2.csv")

# Build 2026 win% lookup
wpct_2026 = {}
records_2026 = team_records[team_records["Season"] == 2026]
for _, row in records_2026.iterrows():
    wpct_2026[row["TeamID"]] = row["WinPct"]

print(f"Teams with 2026 records: {len(wpct_2026)}")

# Default win% for teams with no data (shouldn't happen, but just in case)
DEFAULT_WPCT = 0.5

def predict_matchup(row_id):
    parts = row_id.split("_")
    team1 = int(parts[1])
    team2 = int(parts[2])

    wpct1 = wpct_2026.get(team1, DEFAULT_WPCT)
    wpct2 = wpct_2026.get(team2, DEFAULT_WPCT)

    diff = wpct1 - wpct2
    prob = final_model.predict_proba([[diff]])[0][1]
    return prob

submission["Pred"] = submission["ID"].apply(predict_matchup)

print(f"Predictions generated: {len(submission):,}")
print(f"Prediction range: {submission['Pred'].min():.4f} to {submission['Pred'].max():.4f}")
print(f"Mean prediction:  {submission['Pred'].mean():.4f}")
print()

# Show some interesting matchups (find teams with best/worst records)
print("Some notable 2026 predictions:")
best_teams = records_2026.nlargest(3, "WinPct")
worst_teams = records_2026.nsmallest(3, "WinPct")

for _, good in best_teams.iterrows():
    for _, bad in worst_teams.iterrows():
        t1 = min(int(good["TeamID"]), int(bad["TeamID"]))
        t2 = max(int(good["TeamID"]), int(bad["TeamID"]))
        row_id = f"2026_{t1}_{t2}"
        match = submission[submission["ID"] == row_id]
        if len(match) > 0:
            pred = match.iloc[0]["Pred"]
            name1 = team_name_map.get(t1, str(t1))
            name2 = team_name_map.get(t2, str(t2))
            print(f"  {name1} vs {name2}: P({name1} wins) = {pred:.1%}")
print()

# ============================================================
# STEP 7: Save submission
# ============================================================
output_path = "submission_level2_winpct.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to: {output_path}")
print(f"Total rows: {len(submission):,}")
print()
print("Done! 🏀")
