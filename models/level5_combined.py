"""
Level 5: Combined Model (Elo + Season Stats + Seeds)
=====================================================
Levels 1-4 each captured something different:
  - Level 1: Expert judgment (seeds)
  - Level 2: Win/loss record
  - Level 3: Opponent quality (SOS) and dominance (margin)
  - Level 4: Game-by-game strength tracking (Elo)

Now we combine them all. We feed every feature into logistic regression and let
it figure out the optimal weight for each.

WHY THIS SHOULD WORK:
  Elo captures things the season stats miss (hot streaks, recent form).
  Season stats capture things Elo misses (SOS is computed differently than Elo).
  Seeds capture expert judgment that neither stat fully replicates.
  Together they should cover each other's blind spots.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

DATA = "data/"

# ============================================================
# STEP 1: Load all data
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

# Combine all games for Elo
all_games = pd.concat([all_regular, all_tourney], ignore_index=True)
all_games = all_games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

print(f"Regular season games: {len(all_regular):,}")
print(f"Tournament games:     {len(all_tourney):,}")
print()

# ============================================================
# STEP 2: Compute Elo ratings (using best params from Level 4)
# ============================================================
print("=" * 60)
print("STEP 2: Computing Elo ratings...")
print("=" * 60)

def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

def run_elo(games, K=32, home_advantage=0, margin_factor=0.8, season_reversion=0.4,
            initial_rating=1500):
    """Run Elo and return end-of-regular-season ratings for each (season, team)."""
    ratings = {}
    current_season = None
    # Snapshot ratings at DayNum=132 (end of regular season, before tournament)
    season_snapshots = {}

    for _, game in games.iterrows():
        season = game["Season"]
        winner = game["WTeamID"]
        loser = game["LTeamID"]
        wloc = game["WLoc"]
        daynum = game["DayNum"]

        # Season reset
        if season != current_season:
            # Save snapshot of previous season's end-of-regular-season ratings
            if current_season is not None:
                for team, rating in ratings.items():
                    season_snapshots[(current_season, team)] = rating
                # Apply reversion
                for team in ratings:
                    ratings[team] = initial_rating + (ratings[team] - initial_rating) * (1 - season_reversion)
            current_season = season

        if winner not in ratings:
            ratings[winner] = initial_rating
        if loser not in ratings:
            ratings[loser] = initial_rating

        # Snapshot at transition from regular season to tournament
        if daynum > 132:
            for tid in [winner, loser]:
                if (season, tid) not in season_snapshots:
                    season_snapshots[(season, tid)] = ratings[tid]

        r_winner = ratings[winner]
        r_loser = ratings[loser]

        if wloc == "H":
            pred_winner = expected_score(r_winner + home_advantage, r_loser)
        elif wloc == "A":
            pred_winner = expected_score(r_winner, r_loser + home_advantage)
        else:
            pred_winner = expected_score(r_winner, r_loser)

        if margin_factor > 0:
            margin = game["WScore"] - game["LScore"]
            mov_mult = np.log(1 + margin) * margin_factor
        else:
            mov_mult = 1.0

        update = K * mov_mult * (1 - pred_winner)
        ratings[winner] += update
        ratings[loser] -= update

    # Save final season snapshot
    for team, rating in ratings.items():
        season_snapshots[(current_season, team)] = rating

    return ratings, season_snapshots

print("Running Elo (K=32, margin=0.8, reversion=0.4)...")
current_ratings, elo_snapshots = run_elo(all_games)
print(f"Elo snapshots: {len(elo_snapshots):,} (season, team) pairs")
print()

# ============================================================
# STEP 3: Compute season stats (from Level 3)
# ============================================================
print("=" * 60)
print("STEP 3: Computing season stats (win%, SOS, margin)...")
print("=" * 60)

# Win/loss/margin per team per season
records = {}
for _, row in all_regular.iterrows():
    season = row["Season"]
    winner, loser = row["WTeamID"], row["LTeamID"]
    wscore, lscore = row["WScore"], row["LScore"]

    for tid, pts_for, pts_against, won in [
        (winner, wscore, lscore, True), (loser, lscore, wscore, False)
    ]:
        key = (season, tid)
        if key not in records:
            records[key] = {"Wins": 0, "Losses": 0, "PtsFor": 0, "PtsAgainst": 0,
                            "Games": 0, "Opponents": []}
        records[key]["Wins"] += int(won)
        records[key]["Losses"] += int(not won)
        records[key]["PtsFor"] += pts_for
        records[key]["PtsAgainst"] += pts_against
        records[key]["Games"] += 1
        opp = loser if won else winner
        records[key]["Opponents"].append(opp)

# Compute win% first (needed for SOS)
winpct_lookup = {}
for (season, tid), r in records.items():
    winpct_lookup[(season, tid)] = r["Wins"] / r["Games"]

# Compute all stats including SOS
stats_lookup = {}
for (season, tid), r in records.items():
    opp_winpcts = [winpct_lookup.get((season, opp), 0.5) for opp in r["Opponents"]]
    stats_lookup[(season, tid)] = {
        "WinPct": r["Wins"] / r["Games"],
        "SOS": np.mean(opp_winpcts),
        "ScoringMargin": (r["PtsFor"] - r["PtsAgainst"]) / r["Games"],
    }

# Seed lookup
seed_lookup = {}
for _, row in all_seeds.iterrows():
    seed_lookup[(row["Season"], row["TeamID"])] = row["SeedNum"]

print(f"Season stats computed for {len(stats_lookup):,} (season, team) pairs")
print()

# ============================================================
# STEP 4: Build training data with ALL features
# ============================================================
print("=" * 60)
print("STEP 4: Building training data with all features...")
print("=" * 60)

training_rows = []
for _, row in all_tourney.iterrows():
    season = row["Season"]
    wteam, lteam = row["WTeamID"], row["LTeamID"]

    s_w = stats_lookup.get((season, wteam))
    s_l = stats_lookup.get((season, lteam))
    elo_w = elo_snapshots.get((season, wteam))
    elo_l = elo_snapshots.get((season, lteam))

    if s_w is None or s_l is None or elo_w is None or elo_l is None:
        continue

    # Orient to lower ID
    team1 = min(wteam, lteam)
    team2 = max(wteam, lteam)
    team1_won = 1 if wteam == team1 else 0

    s1 = stats_lookup[(season, team1)]
    s2 = stats_lookup[(season, team2)]
    elo1 = elo_snapshots[(season, team1)]
    elo2 = elo_snapshots[(season, team2)]
    seed1 = seed_lookup.get((season, team1), 16)
    seed2 = seed_lookup.get((season, team2), 16)

    training_rows.append({
        "Season": season,
        "Team1Won": team1_won,
        "EloDiff": elo1 - elo2,
        "WinPctDiff": s1["WinPct"] - s2["WinPct"],
        "SOSDiff": s1["SOS"] - s2["SOS"],
        "MarginDiff": s1["ScoringMargin"] - s2["ScoringMargin"],
        "SeedDiff": seed2 - seed1,  # positive = team1 has better seed
    })

train_df = pd.DataFrame(training_rows)
print(f"Training examples: {len(train_df)}")
print()

# ============================================================
# STEP 5: Compare feature combinations
# ============================================================
print("=" * 60)
print("STEP 5: Comparing feature combinations...")
print("=" * 60)

eval_years = [2022, 2023, 2024, 2025]

feature_sets = {
    "L1: Seeds only":                 ["SeedDiff"],
    "L3: WinPct+SOS+Margin+Seeds":   ["WinPctDiff", "SOSDiff", "MarginDiff", "SeedDiff"],
    "L4: Elo only":                   ["EloDiff"],
    "L5a: Elo + Seeds":              ["EloDiff", "SeedDiff"],
    "L5b: Elo + SOS + Margin":       ["EloDiff", "SOSDiff", "MarginDiff"],
    "L5c: Elo + SOS + Margin + Seeds":["EloDiff", "SOSDiff", "MarginDiff", "SeedDiff"],
    "L5d: ALL features":             ["EloDiff", "WinPctDiff", "SOSDiff", "MarginDiff", "SeedDiff"],
}

print(f"{'Model':<35s} {'2022':>7s} {'2023':>7s} {'2024':>7s} {'2025':>7s} {'Overall':>8s}")
print("-" * 75)

best_brier = 1.0
best_name = ""
best_features = []
all_results = {}

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
    all_results[name] = overall
    print(f"  {name:<33s} {season_briers[0]:>7.4f} {season_briers[1]:>7.4f} "
          f"{season_briers[2]:>7.4f} {season_briers[3]:>7.4f} {overall:>8.4f}")

    if overall < best_brier:
        best_brier = overall
        best_name = name
        best_features = features

print()
print(f"Best model: {best_name} → Brier = {best_brier:.4f}")
print()
print("Full comparison across all levels:")
print(f"  Coin flip:                0.2500")
print(f"  Level 2 (win%):           0.2273")
print(f"  Level 4 (Elo standalone): {all_results.get('L4: Elo only', 'N/A'):.4f}")
print(f"  Level 3 (SOS+margin):     {all_results.get('L3: WinPct+SOS+Margin+Seeds', 'N/A'):.4f}")
print(f"  Level 1 (seeds only):     {all_results.get('L1: Seeds only', 'N/A'):.4f}")
print(f"  Level 5 (best combo):     {best_brier:.4f}")
print()

# ============================================================
# STEP 6: Train final model and show weights
# ============================================================
print("=" * 60)
print("STEP 6: Final model details...")
print("=" * 60)

X_all = train_df[best_features].values
y_all = train_df["Team1Won"].values
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_all, y_all)

print(f"Features and learned weights:")
for feat, weight in zip(best_features, final_model.coef_[0]):
    print(f"  {feat:20s}: {weight:+.4f}")
print(f"  {'Intercept':20s}: {final_model.intercept_[0]:+.4f}")
print()

# ============================================================
# STEP 7: Generate 2026 predictions
# ============================================================
print("=" * 60)
print("STEP 7: Generating 2026 predictions...")
print("=" * 60)

submission = pd.read_csv(f"{DATA}SampleSubmissionStage2.csv")

stats_2026 = {tid: s for (season, tid), s in stats_lookup.items() if season == 2026}
seeds_2026 = {tid: sn for (season, tid), sn in seed_lookup.items() if season == 2026}
default_stats = {"WinPct": 0.5, "SOS": 0.5, "ScoringMargin": 0.0}

print(f"Teams with 2026 stats: {len(stats_2026)}")
print(f"Teams with 2026 seeds: {len(seeds_2026)}")
print(f"Teams with 2026 Elo:   {sum(1 for (s,t) in elo_snapshots if s == 2026)}")

def get_features(team1, team2):
    s1 = stats_2026.get(team1, default_stats)
    s2 = stats_2026.get(team2, default_stats)
    elo1 = current_ratings.get(team1, 1500)
    elo2 = current_ratings.get(team2, 1500)
    seed1 = seeds_2026.get(team1, 16)
    seed2 = seeds_2026.get(team2, 16)

    feat_dict = {
        "EloDiff": elo1 - elo2,
        "WinPctDiff": s1["WinPct"] - s2["WinPct"],
        "SOSDiff": s1["SOS"] - s2["SOS"],
        "MarginDiff": s1["ScoringMargin"] - s2["ScoringMargin"],
        "SeedDiff": seed2 - seed1,
    }
    return [feat_dict[f] for f in best_features]

print("Generating predictions...")
preds = []
for rid in submission["ID"].values:
    parts = rid.split("_")
    team1, team2 = int(parts[1]), int(parts[2])
    features = np.array(get_features(team1, team2)).reshape(1, -1)
    preds.append(final_model.predict_proba(features)[0][1])

submission["Pred"] = preds

print(f"Predictions generated: {len(submission):,}")
print(f"Range: {submission['Pred'].min():.4f} to {submission['Pred'].max():.4f}")
print(f"Mean:  {submission['Pred'].mean():.4f}")
print()

# Show top men's matchups
print("Top men's team matchup predictions:")
top_men = sorted(
    [(tid, current_ratings[tid]) for tid in stats_2026 if tid < 2000 and tid in current_ratings],
    key=lambda x: -x[1]
)[:5]
for i in range(len(top_men)):
    for j in range(i+1, len(top_men)):
        t1 = min(top_men[i][0], top_men[j][0])
        t2 = max(top_men[i][0], top_men[j][0])
        row_id = f"2026_{t1}_{t2}"
        match = submission[submission["ID"] == row_id]
        if len(match) > 0:
            pred = match.iloc[0]["Pred"]
            n1 = team_name_map.get(t1, str(t1))
            n2 = team_name_map.get(t2, str(t2))
            print(f"  {n1} vs {n2}: P({n1} wins) = {pred:.1%}")
print()

# ============================================================
# STEP 8: Save submission
# ============================================================
output_path = "submission_level5_combined.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to: {output_path}")
print()
print("Done! 🏀")
