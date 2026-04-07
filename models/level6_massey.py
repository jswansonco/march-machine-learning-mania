"""
Level 6: Massey Ordinals + Probability Clipping
=================================================
Two improvements:

1. MASSEY ORDINALS (men's only):
   The dataset includes rankings from ~200 different ranking systems (KenPom, Sagarin,
   RPI, AP Poll, etc.). These are expert/algorithmic rankings computed independently.
   We use the FINAL pre-tournament rankings (DayNum=133) as features.

   Rather than picking one system, we'll use an ENSEMBLE approach:
   - Take the average ranking across all available systems for each team
   - This "wisdom of crowds" approach smooths out quirks of individual systems
   - Also use the best individual systems (POM/KenPom is widely considered the gold standard)

   NOTE: Massey Ordinals only exist for men's teams. For women's, we'll rely on the
   features we already have (Elo, SOS, margin, seeds).

2. PROBABILITY CLIPPING:
   Level 5 produced predictions at exactly 0.0 and 1.0. This is dangerous because:
   - If you predict 1.0 and the team loses: Brier penalty = (1.0 - 0)² = 1.0 (maximum!)
   - No sports outcome is truly 100% certain
   We cap predictions to [0.02, 0.98] to protect against this.
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

# Massey Ordinals (men's only)
print("Loading Massey Ordinals (this is 122MB, may take a moment)...")
massey = pd.read_csv(f"{DATA}MMasseyOrdinals.csv")

all_games = pd.concat([all_regular, all_tourney], ignore_index=True)
all_games = all_games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

print(f"Regular season games: {len(all_regular):,}")
print(f"Tournament games:     {len(all_tourney):,}")
print(f"Massey ordinal rows:  {len(massey):,}")
print()

# ============================================================
# STEP 2: Process Massey Ordinals
# ============================================================
print("=" * 60)
print("STEP 2: Processing Massey Ordinals...")
print("=" * 60)

# Use final pre-tournament rankings (DayNum=133) for each season
# If 133 not available, use the latest available ranking
def get_final_rankings(massey_df):
    """For each (season, team), get the final available ranking from each system."""
    # Get the max RankingDayNum per season per system
    max_days = (
        massey_df
        .groupby(["Season", "SystemName"])["RankingDayNum"]
        .max()
        .reset_index()
        .rename(columns={"RankingDayNum": "MaxDay"})
    )

    # Prefer DayNum 133 (final pre-tourney), but fall back to latest available
    final = massey_df.merge(max_days, on=["Season", "SystemName"])
    final = final[final["RankingDayNum"] == final["MaxDay"]]
    final = final.drop(columns=["MaxDay"])

    return final

final_massey = get_final_rankings(massey)
print(f"Final rankings records: {len(final_massey):,}")

# Compute average ranking across all systems for each (season, team)
avg_rank = (
    final_massey
    .groupby(["Season", "TeamID"])["OrdinalRank"]
    .agg(["mean", "min", "count"])
    .rename(columns={"mean": "AvgRank", "min": "BestRank", "count": "NumSystems"})
    .reset_index()
)

print(f"Teams with average rankings: {len(avg_rank):,}")
print()

# Also extract specific top systems individually
# POM (KenPom) is the gold standard
top_systems = ["POM", "MOR", "SAG", "RPI"]
system_ranks = {}
for sys_name in top_systems:
    sys_data = final_massey[final_massey["SystemName"] == sys_name][["Season", "TeamID", "OrdinalRank"]]
    sys_data = sys_data.rename(columns={"OrdinalRank": f"Rank_{sys_name}"})
    system_ranks[sys_name] = sys_data

# Show 2026 top teams by average Massey ranking
print("2026 Men's Top 10 by Average Massey Ranking:")
top_2026 = avg_rank[avg_rank["Season"] == 2026].nsmallest(10, "AvgRank")
for _, row in top_2026.iterrows():
    name = team_name_map.get(row["TeamID"], "???")
    print(f"  {name:20s}  AvgRank: {row['AvgRank']:5.1f}  BestRank: {int(row['BestRank']):3d}  "
          f"({int(row['NumSystems'])} systems)")
print()

# ============================================================
# STEP 3: Compute Elo ratings (same as Level 5)
# ============================================================
print("=" * 60)
print("STEP 3: Computing Elo ratings...")
print("=" * 60)

def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

def run_elo(games, K=32, home_advantage=0, margin_factor=0.8, season_reversion=0.4,
            initial_rating=1500):
    ratings = {}
    current_season = None
    season_snapshots = {}

    for _, game in games.iterrows():
        season = game["Season"]
        winner = game["WTeamID"]
        loser = game["LTeamID"]
        wloc = game["WLoc"]
        daynum = game["DayNum"]

        if season != current_season:
            if current_season is not None:
                for team, rating in ratings.items():
                    season_snapshots[(current_season, team)] = rating
                for team in ratings:
                    ratings[team] = initial_rating + (ratings[team] - initial_rating) * (1 - season_reversion)
            current_season = season

        if winner not in ratings:
            ratings[winner] = initial_rating
        if loser not in ratings:
            ratings[loser] = initial_rating

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

    for team, rating in ratings.items():
        season_snapshots[(current_season, team)] = rating

    return ratings, season_snapshots

print("Running Elo...")
current_ratings, elo_snapshots = run_elo(all_games)
print(f"Elo snapshots: {len(elo_snapshots):,}")
print()

# ============================================================
# STEP 4: Compute season stats (same as Level 5)
# ============================================================
print("=" * 60)
print("STEP 4: Computing season stats...")
print("=" * 60)

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

winpct_lookup = {}
for (season, tid), r in records.items():
    winpct_lookup[(season, tid)] = r["Wins"] / r["Games"]

stats_lookup = {}
for (season, tid), r in records.items():
    opp_winpcts = [winpct_lookup.get((season, opp), 0.5) for opp in r["Opponents"]]
    stats_lookup[(season, tid)] = {
        "WinPct": r["Wins"] / r["Games"],
        "SOS": np.mean(opp_winpcts),
        "ScoringMargin": (r["PtsFor"] - r["PtsAgainst"]) / r["Games"],
    }

seed_lookup = {}
for _, row in all_seeds.iterrows():
    seed_lookup[(row["Season"], row["TeamID"])] = row["SeedNum"]

# Build Massey lookup: (season, team) -> {AvgRank, BestRank, Rank_POM, ...}
massey_lookup = {}
for _, row in avg_rank.iterrows():
    massey_lookup[(row["Season"], row["TeamID"])] = {
        "AvgRank": row["AvgRank"],
        "BestRank": row["BestRank"],
    }

for sys_name, sys_df in system_ranks.items():
    for _, row in sys_df.iterrows():
        key = (row["Season"], row["TeamID"])
        if key not in massey_lookup:
            massey_lookup[key] = {"AvgRank": 175, "BestRank": 175}
        massey_lookup[key][f"Rank_{sys_name}"] = row[f"Rank_{sys_name}"]

print(f"Season stats: {len(stats_lookup):,}")
print(f"Massey lookups: {len(massey_lookup):,}")
print()

# ============================================================
# STEP 5: Build training data with ALL features
# ============================================================
print("=" * 60)
print("STEP 5: Building training data...")
print("=" * 60)

# Default values for missing data
DEFAULT_RANK = 175  # middle of ~350 teams
DEFAULT_STATS = {"WinPct": 0.5, "SOS": 0.5, "ScoringMargin": 0.0}

def get_massey(season, team):
    m = massey_lookup.get((season, team), {})
    return {
        "AvgRank": m.get("AvgRank", DEFAULT_RANK),
        "BestRank": m.get("BestRank", DEFAULT_RANK),
        "Rank_POM": m.get("Rank_POM", DEFAULT_RANK),
    }

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

    team1 = min(wteam, lteam)
    team2 = max(wteam, lteam)
    team1_won = 1 if wteam == team1 else 0

    s1 = stats_lookup[(season, team1)]
    s2 = stats_lookup[(season, team2)]
    elo1 = elo_snapshots[(season, team1)]
    elo2 = elo_snapshots[(season, team2)]
    seed1 = seed_lookup.get((season, team1), 16)
    seed2 = seed_lookup.get((season, team2), 16)
    m1 = get_massey(season, team1)
    m2 = get_massey(season, team2)

    # Check if both teams are men's (Massey only available for men)
    is_mens = team1 < 2000

    training_rows.append({
        "Season": season,
        "Team1Won": team1_won,
        "IsMens": int(is_mens),
        # Level 5 features
        "EloDiff": elo1 - elo2,
        "WinPctDiff": s1["WinPct"] - s2["WinPct"],
        "SOSDiff": s1["SOS"] - s2["SOS"],
        "MarginDiff": s1["ScoringMargin"] - s2["ScoringMargin"],
        "SeedDiff": seed2 - seed1,
        # Massey features (men's only — defaults for women's)
        "AvgRankDiff": m2["AvgRank"] - m1["AvgRank"],  # positive = team1 ranked better
        "BestRankDiff": m2["BestRank"] - m1["BestRank"],
        "POMRankDiff": m2["Rank_POM"] - m1["Rank_POM"],
    })

train_df = pd.DataFrame(training_rows)

# Only use Massey features for men's games (2003+ when data exists)
# For women's, Massey features will be 0 (both teams get default rank 175)
print(f"Training examples: {len(train_df)}")
print(f"  Men's: {train_df['IsMens'].sum()}")
print(f"  Women's: {(~train_df['IsMens'].astype(bool)).sum()}")
print()

# ============================================================
# STEP 6: Compare feature combinations
# ============================================================
print("=" * 60)
print("STEP 6: Comparing feature combinations...")
print("=" * 60)

eval_years = [2022, 2023, 2024, 2025]

feature_sets = {
    "L5: All (no Massey)":          ["EloDiff", "WinPctDiff", "SOSDiff", "MarginDiff", "SeedDiff"],
    "L6a: L5 + AvgRank":           ["EloDiff", "WinPctDiff", "SOSDiff", "MarginDiff", "SeedDiff", "AvgRankDiff"],
    "L6b: L5 + POM":               ["EloDiff", "WinPctDiff", "SOSDiff", "MarginDiff", "SeedDiff", "POMRankDiff"],
    "L6c: L5 + Avg + Best + POM":  ["EloDiff", "WinPctDiff", "SOSDiff", "MarginDiff", "SeedDiff",
                                     "AvgRankDiff", "BestRankDiff", "POMRankDiff"],
    "L6d: Elo + POM + Seeds":      ["EloDiff", "POMRankDiff", "SeedDiff"],
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

# ============================================================
# STEP 7: Test probability clipping
# ============================================================
print("=" * 60)
print("STEP 7: Testing probability clipping...")
print("=" * 60)

# Re-evaluate best model with different clip levels
clip_levels = [
    (0.00, 1.00, "No clip"),
    (0.01, 0.99, "Clip [0.01, 0.99]"),
    (0.02, 0.98, "Clip [0.02, 0.98]"),
    (0.03, 0.97, "Clip [0.03, 0.97]"),
    (0.05, 0.95, "Clip [0.05, 0.95]"),
]

best_clip = (0.0, 1.0)
best_clip_brier = 1.0

for lo, hi, label in clip_levels:
    season_briers = []
    for season in eval_years:
        train_mask = train_df["Season"] < season
        test_mask = train_df["Season"] == season

        X_train = train_df.loc[train_mask, best_features].values
        y_train = train_df.loc[train_mask, "Team1Won"].values
        X_test = train_df.loc[test_mask, best_features].values
        y_test = train_df.loc[test_mask, "Team1Won"].values

        m = LogisticRegression(max_iter=1000)
        m.fit(X_train, y_train)
        preds = np.clip(m.predict_proba(X_test)[:, 1], lo, hi)
        brier = np.mean((preds - y_test) ** 2)
        season_briers.append(brier)

    overall = np.mean(season_briers)
    print(f"  {label:25s}: {overall:.4f}")

    if overall < best_clip_brier:
        best_clip_brier = overall
        best_clip = (lo, hi)

print(f"\nBest clip: [{best_clip[0]}, {best_clip[1]}] → Brier = {best_clip_brier:.4f}")
print()

# ============================================================
# STEP 8: Train final model
# ============================================================
print("=" * 60)
print("STEP 8: Training final model...")
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
# STEP 9: Generate 2026 predictions
# ============================================================
print("=" * 60)
print("STEP 9: Generating 2026 predictions...")
print("=" * 60)

submission = pd.read_csv(f"{DATA}SampleSubmissionStage2.csv")

stats_2026 = {tid: s for (season, tid), s in stats_lookup.items() if season == 2026}
seeds_2026 = {tid: sn for (season, tid), sn in seed_lookup.items() if season == 2026}

print(f"Teams with 2026 stats: {len(stats_2026)}")
print(f"Teams with 2026 seeds: {len(seeds_2026)}")
print(f"Teams with 2026 Massey: {len([k for k in massey_lookup if k[0]==2026])}")

CLIP_LO, CLIP_HI = best_clip

def get_features(team1, team2):
    s1 = stats_2026.get(team1, DEFAULT_STATS)
    s2 = stats_2026.get(team2, DEFAULT_STATS)
    elo1 = current_ratings.get(team1, 1500)
    elo2 = current_ratings.get(team2, 1500)
    seed1 = seeds_2026.get(team1, 16)
    seed2 = seeds_2026.get(team2, 16)
    m1 = get_massey(2026, team1)
    m2 = get_massey(2026, team2)

    feat_dict = {
        "EloDiff": elo1 - elo2,
        "WinPctDiff": s1["WinPct"] - s2["WinPct"],
        "SOSDiff": s1["SOS"] - s2["SOS"],
        "MarginDiff": s1["ScoringMargin"] - s2["ScoringMargin"],
        "SeedDiff": seed2 - seed1,
        "AvgRankDiff": m2["AvgRank"] - m1["AvgRank"],
        "BestRankDiff": m2["BestRank"] - m1["BestRank"],
        "POMRankDiff": m2["Rank_POM"] - m1["Rank_POM"],
    }
    return [feat_dict[f] for f in best_features]

print("Generating predictions...")
preds = []
for rid in submission["ID"].values:
    parts = rid.split("_")
    team1, team2 = int(parts[1]), int(parts[2])
    features = np.array(get_features(team1, team2)).reshape(1, -1)
    prob = final_model.predict_proba(features)[0][1]
    preds.append(np.clip(prob, CLIP_LO, CLIP_HI))

submission["Pred"] = preds

print(f"Predictions generated: {len(submission):,}")
print(f"Range: {submission['Pred'].min():.4f} to {submission['Pred'].max():.4f}")
print(f"Mean:  {submission['Pred'].mean():.4f}")
print()

# Comparison
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(f"  Coin flip:             0.2500")
print(f"  Level 2 (win%):        0.2273")
print(f"  Level 4 (Elo):         0.1809")
print(f"  Level 3 (SOS+margin):  0.1679")
print(f"  Level 1 (seeds):       0.1671")
print(f"  Level 5 (combined):    0.0979")
print(f"  Level 6 (Massey+clip): {best_clip_brier:.4f}")
print()

# Top men's matchup predictions
print("Top men's team predictions:")
top_men = sorted(
    [(tid, current_ratings[tid]) for tid in stats_2026 if tid < 2000 and tid in current_ratings],
    key=lambda x: -x[1]
)[:6]
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

output_path = "submission_level6_massey.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to: {output_path}")
print()
print("Done! 🏀")
