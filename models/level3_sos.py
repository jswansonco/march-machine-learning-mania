"""
Level 3: Win Percentage + Strength of Schedule
================================================
Level 2's problem: a 28-3 team in a weak conference looks the same as 28-3 in a
powerhouse conference. We fix this by measuring the QUALITY of opponents.

NEW CONCEPTS:

1. STRENGTH OF SCHEDULE (SOS):
   "How hard was your schedule?"
   We measure this as the average win percentage of all teams you played against.
   If you mostly played teams with 70%+ win rates, your schedule was tough.
   If you mostly played teams with 40% win rates, your schedule was easy.

2. ADJUSTED WIN PERCENTAGE:
   Raw win% + SOS together paint a much better picture.
   Going 25-5 against hard opponents is more impressive than 30-0 against easy ones.

3. MULTIPLE FEATURES in logistic regression:
   Level 2 used ONE number (win% gap) to predict outcomes.
   Now we'll use MULTIPLE numbers (win% gap AND SOS gap AND scoring margin).
   Logistic regression handles multiple features — it just learns a weight for each.

4. SCORING MARGIN:
   Average "points scored minus points allowed" per game.
   This captures HOW MUCH a team wins/loses by, not just whether they won.
   A team that wins by 20 every game is probably better than one that squeaks by.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

DATA = "data/"

# ============================================================
# STEP 1: Load data
# ============================================================
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

m_regular = pd.read_csv(f"{DATA}MRegularSeasonCompactResults.csv")
w_regular = pd.read_csv(f"{DATA}WRegularSeasonCompactResults.csv")
all_regular = pd.concat([m_regular, w_regular], ignore_index=True)

m_tourney = pd.read_csv(f"{DATA}MNCAATourneyCompactResults.csv")
w_tourney = pd.read_csv(f"{DATA}WNCAATourneyCompactResults.csv")
all_tourney = pd.concat([m_tourney, w_tourney], ignore_index=True)

m_seeds = pd.read_csv(f"{DATA}MNCAATourneySeeds.csv")
w_seeds = pd.read_csv(f"{DATA}WNCAATourneySeeds.csv")
all_seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)
all_seeds["SeedNum"] = all_seeds["Seed"].apply(lambda s: int(s[1:3]))

m_teams = pd.read_csv(f"{DATA}MTeams.csv")
w_teams = pd.read_csv(f"{DATA}WTeams.csv")
team_name_map = dict(zip(
    pd.concat([m_teams[["TeamID","TeamName"]], w_teams[["TeamID","TeamName"]]])["TeamID"],
    pd.concat([m_teams[["TeamID","TeamName"]], w_teams[["TeamID","TeamName"]]])["TeamName"]
))

print(f"Regular season games: {len(all_regular):,}")
print(f"Tournament games:     {len(all_tourney):,}")
print()

# ============================================================
# STEP 2: Calculate team stats per season
# ============================================================
print("=" * 60)
print("STEP 2: Calculating team stats per season...")
print("=" * 60)

def compute_team_stats(games_df):
    """
    For every (season, team), compute:
      - Wins, Losses, WinPct
      - PointsFor (avg points scored per game)
      - PointsAgainst (avg points allowed per game)
      - ScoringMargin (avg margin = PointsFor - PointsAgainst)
      - Opponents list (for SOS calculation later)
    """
    records = {}  # (season, team) -> {wins, losses, pts_for, pts_against, opponents}

    for _, row in games_df.iterrows():
        season = row["Season"]
        winner = row["WTeamID"]
        loser = row["LTeamID"]
        wscore = row["WScore"]
        lscore = row["LScore"]

        # Winner's record
        key_w = (season, winner)
        if key_w not in records:
            records[key_w] = {"Wins": 0, "Losses": 0, "PtsFor": 0, "PtsAgainst": 0,
                              "Games": 0, "Opponents": []}
        records[key_w]["Wins"] += 1
        records[key_w]["PtsFor"] += wscore
        records[key_w]["PtsAgainst"] += lscore
        records[key_w]["Games"] += 1
        records[key_w]["Opponents"].append(loser)

        # Loser's record
        key_l = (season, loser)
        if key_l not in records:
            records[key_l] = {"Wins": 0, "Losses": 0, "PtsFor": 0, "PtsAgainst": 0,
                              "Games": 0, "Opponents": []}
        records[key_l]["Losses"] += 1
        records[key_l]["PtsFor"] += lscore
        records[key_l]["PtsAgainst"] += wscore
        records[key_l]["Games"] += 1
        records[key_l]["Opponents"].append(winner)

    return records

records = compute_team_stats(all_regular)

# Convert to a flat dataframe (without opponents list, we'll use that separately)
stats_rows = []
for (season, team), r in records.items():
    stats_rows.append({
        "Season": season,
        "TeamID": team,
        "Wins": r["Wins"],
        "Losses": r["Losses"],
        "Games": r["Games"],
        "WinPct": r["Wins"] / r["Games"],
        "AvgPtsFor": r["PtsFor"] / r["Games"],
        "AvgPtsAgainst": r["PtsAgainst"] / r["Games"],
        "ScoringMargin": (r["PtsFor"] - r["PtsAgainst"]) / r["Games"],
    })

team_stats = pd.DataFrame(stats_rows)

# Step 2b: Now compute Strength of Schedule (SOS)
# SOS = average win percentage of all opponents you played
print("Computing Strength of Schedule...")

# First, build a quick lookup for win%
winpct_lookup = {}
for _, row in team_stats.iterrows():
    winpct_lookup[(row["Season"], row["TeamID"])] = row["WinPct"]

# Now compute SOS for each team
sos_values = []
for (season, team), r in records.items():
    opp_winpcts = []
    for opp in r["Opponents"]:
        opp_wp = winpct_lookup.get((season, opp))
        if opp_wp is not None:
            opp_winpcts.append(opp_wp)
    sos = np.mean(opp_winpcts) if opp_winpcts else 0.5
    sos_values.append({"Season": season, "TeamID": team, "SOS": sos})

sos_df = pd.DataFrame(sos_values)
team_stats = team_stats.merge(sos_df, on=["Season", "TeamID"])

print(f"Team-season records with SOS: {len(team_stats):,}")
print()

# Show some examples — compare teams with similar win% but different SOS
print("2025 season: Teams with ~85-90% win rate, sorted by SOS:")
mask = (team_stats["Season"] == 2025) & (team_stats["WinPct"] >= 0.80) & (team_stats["WinPct"] <= 0.92)
example = team_stats[mask].sort_values("SOS", ascending=False).head(10)
for _, row in example.iterrows():
    name = team_name_map.get(row["TeamID"], "???")
    print(f"  {name:20s}  {int(row['Wins']):2d}-{int(row['Losses']):2d} "
          f"(Win%={row['WinPct']:.1%})  SOS={row['SOS']:.3f}  "
          f"Margin={row['ScoringMargin']:+.1f}")
print("  ↑ Higher SOS = tougher opponents = more impressive record")
print()

# ============================================================
# STEP 3: Build training data with multiple features
# ============================================================
print("=" * 60)
print("STEP 3: Building training data with multiple features...")
print("=" * 60)

# Build a full lookup for all stats
stats_lookup = {}
for _, row in team_stats.iterrows():
    stats_lookup[(row["Season"], row["TeamID"])] = row

# Also build seed lookup
seed_lookup = {}
for _, row in all_seeds.iterrows():
    seed_lookup[(row["Season"], row["TeamID"])] = row["SeedNum"]

training_rows = []
for _, row in all_tourney.iterrows():
    season = row["Season"]
    wteam = row["WTeamID"]
    lteam = row["LTeamID"]

    stats_w = stats_lookup.get((season, wteam))
    stats_l = stats_lookup.get((season, lteam))
    if stats_w is None or stats_l is None:
        continue

    # Orient to lower ID perspective
    team1 = min(wteam, lteam)
    team2 = max(wteam, lteam)
    team1_won = 1 if wteam == team1 else 0

    s1 = stats_lookup[(season, team1)]
    s2 = stats_lookup[(season, team2)]

    seed1 = seed_lookup.get((season, team1), 16)
    seed2 = seed_lookup.get((season, team2), 16)

    training_rows.append({
        "Season": season,
        "Team1Won": team1_won,
        # Feature: gap in win percentages
        "WinPctDiff": s1["WinPct"] - s2["WinPct"],
        # Feature: gap in strength of schedule
        "SOSDiff": s1["SOS"] - s2["SOS"],
        # Feature: gap in scoring margin
        "MarginDiff": s1["ScoringMargin"] - s2["ScoringMargin"],
        # Feature: gap in seeds (negative = team1 has better seed = lower number)
        "SeedDiff": seed2 - seed1,  # positive means team1 has better seed
    })

train_df = pd.DataFrame(training_rows)
print(f"Training examples: {len(train_df)}")
print()

# ============================================================
# STEP 4: Train and evaluate different feature combinations
# ============================================================
print("=" * 60)
print("STEP 4: Comparing feature combinations...")
print("=" * 60)

eval_years = [2022, 2023, 2024, 2025]

feature_sets = {
    "L2: WinPct only":              ["WinPctDiff"],
    "L3a: WinPct + SOS":            ["WinPctDiff", "SOSDiff"],
    "L3b: WinPct + SOS + Margin":   ["WinPctDiff", "SOSDiff", "MarginDiff"],
    "L3c: All (+ Seeds)":           ["WinPctDiff", "SOSDiff", "MarginDiff", "SeedDiff"],
    "L3d: Margin + Seeds":          ["MarginDiff", "SeedDiff"],
}

print(f"{'Model':<35s} {'2022':>7s} {'2023':>7s} {'2024':>7s} {'2025':>7s} {'Overall':>8s}")
print("-" * 75)

best_brier = 1.0
best_name = ""
best_features = []

for name, features in feature_sets.items():
    season_briers = []
    for season in eval_years:
        train_mask = train_df["Season"] < season
        test_mask = train_df["Season"] == season

        X_train = train_df.loc[train_mask, features].values
        y_train = train_df.loc[train_mask, "Team1Won"].values
        X_test = train_df.loc[test_mask, features].values
        y_test = train_df.loc[test_mask, "Team1Won"].values

        if len(X_train) == 0 or len(X_test) == 0:
            season_briers.append(0.25)
            continue

        m = LogisticRegression(max_iter=1000)
        m.fit(X_train, y_train)
        preds = m.predict_proba(X_test)[:, 1]
        brier = np.mean((preds - y_test) ** 2)
        season_briers.append(brier)

    overall = np.mean(season_briers)
    print(f"  {name:<33s} {season_briers[0]:>7.4f} {season_briers[1]:>7.4f} "
          f"{season_briers[2]:>7.4f} {season_briers[3]:>7.4f} {overall:>8.4f}")

    if overall < best_brier:
        best_brier = overall
        best_name = name
        best_features = features

print()
print(f"Best model: {best_name} → Brier = {best_brier:.4f}")
print()
print("Reference scores:")
print(f"  Coin flip:         0.2500")
print(f"  Level 1 (seeds):   0.1671")
print(f"  Level 2 (win%):    0.2273")
print(f"  Level 3 (best):    {best_brier:.4f}")
print()

# ============================================================
# STEP 5: Train final model and show what it learned
# ============================================================
print("=" * 60)
print("STEP 5: Final model details...")
print("=" * 60)

X_all = train_df[best_features].values
y_all = train_df["Team1Won"].values
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_all, y_all)

print(f"Features and their learned weights:")
for feat, weight in zip(best_features, final_model.coef_[0]):
    print(f"  {feat:20s}: {weight:+.3f}")
print(f"  {'Intercept':20s}: {final_model.intercept_[0]:+.3f}")
print()
print("Interpretation: larger positive weight = stronger predictor.")
print("A positive weight means 'when this value is higher for team1, team1 is more likely to win'.")
print()

# ============================================================
# STEP 6: Generate 2026 predictions
# ============================================================
print("=" * 60)
print("STEP 6: Generating 2026 predictions...")
print("=" * 60)

submission = pd.read_csv(f"{DATA}SampleSubmissionStage2.csv")

# 2026 stats lookup
stats_2026 = {row["TeamID"]: row for _, row in team_stats[team_stats["Season"] == 2026].iterrows()}
seeds_2026 = {tid: sn for (s, tid), sn in seed_lookup.items() if s == 2026}

print(f"Teams with 2026 stats: {len(stats_2026)}")
print(f"Teams with 2026 seeds: {len(seeds_2026)} (0 expected pre-Selection Sunday)")

# Default stats for unknown teams
default_stats = {"WinPct": 0.5, "SOS": 0.5, "ScoringMargin": 0.0}

def get_features(team1, team2):
    s1 = stats_2026.get(team1, default_stats)
    s2 = stats_2026.get(team2, default_stats)

    feat_dict = {
        "WinPctDiff": s1["WinPct"] - s2["WinPct"],
        "SOSDiff": s1["SOS"] - s2["SOS"],
        "MarginDiff": s1["ScoringMargin"] - s2["ScoringMargin"],
        "SeedDiff": seeds_2026.get(team2, 16) - seeds_2026.get(team1, 16),
    }
    return [feat_dict[f] for f in best_features]

def predict_matchup(row_id):
    parts = row_id.split("_")
    team1, team2 = int(parts[1]), int(parts[2])
    features = np.array(get_features(team1, team2)).reshape(1, -1)
    return final_model.predict_proba(features)[0][1]

# This is slow with .apply on 132K rows — let's vectorize
print("Generating predictions (this may take a moment)...")

ids = submission["ID"].values
preds = []
for rid in ids:
    preds.append(predict_matchup(rid))

submission["Pred"] = preds

print(f"Predictions generated: {len(submission):,}")
print(f"Range: {submission['Pred'].min():.4f} to {submission['Pred'].max():.4f}")
print(f"Mean:  {submission['Pred'].mean():.4f}")
print()

# Show top predicted blowouts and closest matchups
print("Most lopsided predicted matchups:")
submission_sorted = submission.sort_values("Pred")
for _, row in submission_sorted.head(3).iterrows():
    parts = row["ID"].split("_")
    t1, t2 = int(parts[1]), int(parts[2])
    n1, n2 = team_name_map.get(t1, str(t1)), team_name_map.get(t2, str(t2))
    print(f"  {n1} vs {n2}: P({n1} wins) = {row['Pred']:.1%}")

print()
print("Closest to 50/50:")
submission["DistFrom50"] = abs(submission["Pred"] - 0.5)
closest = submission.nsmallest(3, "DistFrom50")
for _, row in closest.iterrows():
    parts = row["ID"].split("_")
    t1, t2 = int(parts[1]), int(parts[2])
    n1, n2 = team_name_map.get(t1, str(t1)), team_name_map.get(t2, str(t2))
    print(f"  {n1} vs {n2}: P({n1} wins) = {row['Pred']:.1%}")

submission = submission.drop(columns=["DistFrom50"])
print()

# ============================================================
# STEP 7: Save submission
# ============================================================
output_path = "submission_level3_sos.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to: {output_path}")
print()
print("Done! 🏀")
