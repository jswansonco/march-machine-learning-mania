"""
Level 4: Elo Rating System
============================
Levels 1-3 summarize each team's ENTIRE season into a few numbers. But this misses
important dynamics:
  - A team on a 15-game win streak is hot
  - A team that lost its star player is declining
  - Games from last week tell us more than games from November

Elo fixes this by updating team ratings AFTER EVERY GAME.

HOW ELO WORKS (originally invented for chess):

1. Every team starts with a rating (we use 1500).
2. Before a game, we predict the outcome based on the rating gap.
3. After the game, the winner gains points and the loser loses points.
4. The amount gained/lost depends on how SURPRISING the result was:
   - If a strong team beats a weak team → small update (expected)
   - If a weak team beats a strong team → big update (upset!)

The key formula:
  Expected win probability = 1 / (1 + 10^((rating_B - rating_A) / 400))
  New rating = Old rating + K * (actual_outcome - expected_outcome)

K is the "learning rate" — how fast ratings react to new results.
  - High K (e.g. 40): ratings swing a lot after each game (responsive but noisy)
  - Low K (e.g. 10): ratings change slowly (stable but slow to adapt)

ADDITIONAL TWEAKS WE'LL TRY:
  - Home court advantage: home teams win ~60-65% in college basketball
  - Margin of victory: winning by 20 should boost your rating more than winning by 1
  - Season reset: partially reset ratings each new season (teams change year to year)
"""

import pandas as pd
import numpy as np
from itertools import product

DATA = "data/"

# ============================================================
# STEP 1: Load data
# ============================================================
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

m_regular = pd.read_csv(f"{DATA}MRegularSeasonCompactResults.csv")
w_regular = pd.read_csv(f"{DATA}WRegularSeasonCompactResults.csv")
m_tourney = pd.read_csv(f"{DATA}MNCAATourneyCompactResults.csv")
w_tourney = pd.read_csv(f"{DATA}WNCAATourneyCompactResults.csv")
m_conf_tourney = pd.read_csv(f"{DATA}MConferenceTourneyGames.csv")
w_conf_tourney = pd.read_csv(f"{DATA}WConferenceTourneyGames.csv")

# Combine all games (regular + conference tourney + NCAA tourney) in chronological order
# We need ALL games to build accurate Elo ratings
all_regular = pd.concat([m_regular, w_regular], ignore_index=True)
all_ncaa = pd.concat([m_tourney, w_tourney], ignore_index=True)

# Conference tourney games are already in the regular season file (DayNum <= 132)
# NCAA tourney games are separate. Combine them all.
all_games = pd.concat([all_regular, all_ncaa], ignore_index=True)
all_games = all_games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

# Load seeds for evaluation comparison
m_seeds = pd.read_csv(f"{DATA}MNCAATourneySeeds.csv")
w_seeds = pd.read_csv(f"{DATA}WNCAATourneySeeds.csv")
all_seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)
all_seeds["SeedNum"] = all_seeds["Seed"].apply(lambda s: int(s[1:3]))

# Team names for display
m_teams = pd.read_csv(f"{DATA}MTeams.csv")
w_teams = pd.read_csv(f"{DATA}WTeams.csv")
team_name_map = dict(zip(
    pd.concat([m_teams[["TeamID","TeamName"]], w_teams[["TeamID","TeamName"]]])["TeamID"],
    pd.concat([m_teams[["TeamID","TeamName"]], w_teams[["TeamID","TeamName"]]])["TeamName"]
))

print(f"Total games (regular + tourney): {len(all_games):,}")
print(f"Seasons: {all_games['Season'].min()} to {all_games['Season'].max()}")
print()

# ============================================================
# STEP 2: Basic Elo system
# ============================================================
print("=" * 60)
print("STEP 2: Building Elo rating system...")
print("=" * 60)

def expected_score(rating_a, rating_b):
    """Probability that team A beats team B, given their Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

def run_elo(games, K=20, home_advantage=100, margin_factor=0.0, season_reversion=0.5,
            initial_rating=1500):
    """
    Run Elo ratings through all games chronologically.

    Parameters:
    - K: learning rate (how much ratings change per game)
    - home_advantage: Elo points added to home team's rating for prediction
    - margin_factor: how much to weight margin of victory (0 = ignore margin)
    - season_reversion: how much to pull ratings back toward 1500 each new season
      0.0 = no reset (carry over fully), 1.0 = full reset to 1500
    - initial_rating: starting Elo for all teams

    Returns:
    - ratings: dict of {team_id: current_elo}
    - history: list of dicts with prediction details for each game
    """
    ratings = {}
    history = []
    current_season = None

    for _, game in games.iterrows():
        season = game["Season"]
        winner = game["WTeamID"]
        loser = game["LTeamID"]
        wloc = game["WLoc"]

        # Season reset: pull ratings toward the mean
        if season != current_season:
            if current_season is not None:
                for team in ratings:
                    ratings[team] = initial_rating + (ratings[team] - initial_rating) * (1 - season_reversion)
            current_season = season

        # Initialize new teams
        if winner not in ratings:
            ratings[winner] = initial_rating
        if loser not in ratings:
            ratings[loser] = initial_rating

        # Get current ratings
        r_winner = ratings[winner]
        r_loser = ratings[loser]

        # Apply home court advantage
        if wloc == "H":
            # Winner was home team
            pred_winner = expected_score(r_winner + home_advantage, r_loser)
        elif wloc == "A":
            # Winner was AWAY (loser was home)
            pred_winner = expected_score(r_winner, r_loser + home_advantage)
        else:
            # Neutral site
            pred_winner = expected_score(r_winner, r_loser)

        # Record this game's prediction (oriented to lower ID for submission format)
        team1 = min(winner, loser)
        team2 = max(winner, loser)
        team1_won = 1 if winner == team1 else 0

        r1 = ratings[team1]
        r2 = ratings[team2]
        pred_team1 = expected_score(r1, r2)  # neutral site prediction for submission

        history.append({
            "Season": season,
            "DayNum": game["DayNum"],
            "Team1": team1,
            "Team2": team2,
            "Team1Won": team1_won,
            "PredTeam1": pred_team1,
            "EloTeam1": r1,
            "EloTeam2": r2,
        })

        # Margin of victory multiplier
        if margin_factor > 0:
            margin = game["WScore"] - game["LScore"]
            # Log-based MOV multiplier: diminishing returns for blowouts
            mov_mult = np.log(1 + margin) * margin_factor
        else:
            mov_mult = 1.0

        # Update ratings
        update = K * mov_mult * (1 - pred_winner)
        ratings[winner] += update
        ratings[loser] -= update

    return ratings, history

# Run basic Elo first
print("Running basic Elo (K=20, no home advantage, no margin)...")
ratings_basic, history_basic = run_elo(
    all_games, K=20, home_advantage=0, margin_factor=0.0, season_reversion=0.5
)
print(f"Teams rated: {len(ratings_basic)}")
print()

# ============================================================
# STEP 3: Evaluate basic Elo on tournament games
# ============================================================
print("=" * 60)
print("STEP 3: Evaluating basic Elo...")
print("=" * 60)

def evaluate_elo(history, eval_years=[2022, 2023, 2024, 2025]):
    """Compute Brier score on tournament games only."""
    hist_df = pd.DataFrame(history)

    # Tournament games have DayNum > 132
    tourney = hist_df[(hist_df["DayNum"] > 132) & (hist_df["Season"].isin(eval_years))]

    brier_by_season = {}
    for season in eval_years:
        season_games = tourney[tourney["Season"] == season]
        if len(season_games) == 0:
            continue
        brier = np.mean((season_games["PredTeam1"] - season_games["Team1Won"]) ** 2)
        brier_by_season[season] = (brier, len(season_games))

    overall_brier = np.mean((tourney["PredTeam1"] - tourney["Team1Won"]) ** 2)
    return brier_by_season, overall_brier

season_briers, overall = evaluate_elo(history_basic)
for season, (brier, n) in sorted(season_briers.items()):
    print(f"  {season}: Brier = {brier:.4f} ({n} games)")
print(f"\n  OVERALL: {overall:.4f}")
print()

# ============================================================
# STEP 4: Tune Elo hyperparameters
# ============================================================
print("=" * 60)
print("STEP 4: Tuning Elo hyperparameters...")
print("=" * 60)
print("Testing different combinations of K, home advantage, margin, and reversion...")
print("(This takes a minute — running ~50 combinations)")
print()

# Grid search over key parameters
best_brier = 1.0
best_params = {}
results = []

K_values = [15, 20, 25, 32]
home_values = [0, 70, 100, 130]
margin_values = [0.0, 0.5, 0.8]
reversion_values = [0.4, 0.6, 0.8]

total = len(K_values) * len(home_values) * len(margin_values) * len(reversion_values)
count = 0

for K, home, margin, reversion in product(K_values, home_values, margin_values, reversion_values):
    count += 1
    if count % 20 == 0:
        print(f"  ...tested {count}/{total} combinations")

    _, hist = run_elo(all_games, K=K, home_advantage=home,
                      margin_factor=margin, season_reversion=reversion)
    _, brier = evaluate_elo(hist)
    results.append({
        "K": K, "Home": home, "Margin": margin, "Reversion": reversion, "Brier": brier
    })
    if brier < best_brier:
        best_brier = brier
        best_params = {"K": K, "home_advantage": home, "margin_factor": margin, "season_reversion": reversion}

results_df = pd.DataFrame(results).sort_values("Brier")

print(f"\nTop 5 parameter combinations:")
print(f"{'K':>4s} {'Home':>5s} {'Margin':>7s} {'Revert':>7s} {'Brier':>8s}")
print("-" * 35)
for _, row in results_df.head(5).iterrows():
    print(f"{int(row['K']):>4d} {int(row['Home']):>5d} {row['Margin']:>7.1f} "
          f"{row['Reversion']:>7.1f} {row['Brier']:>8.4f}")

print(f"\nBest: K={best_params['K']}, Home={best_params['home_advantage']}, "
      f"Margin={best_params['margin_factor']}, Reversion={best_params['season_reversion']}")
print(f"Brier: {best_brier:.4f}")
print()

# ============================================================
# STEP 5: Run final model with best parameters
# ============================================================
print("=" * 60)
print("STEP 5: Running final Elo with best parameters...")
print("=" * 60)

final_ratings, final_history = run_elo(
    all_games, **best_params
)

season_briers, overall = evaluate_elo(final_history)
for season, (brier, n) in sorted(season_briers.items()):
    print(f"  {season}: Brier = {brier:.4f} ({n} games)")
print(f"\n  OVERALL: {overall:.4f}")
print()

print("Comparison:")
print(f"  Coin flip:             0.2500")
print(f"  Level 2 (win%):        0.2273")
print(f"  Level 3 (SOS+margin):  0.1679")
print(f"  Level 1 (seeds):       0.1671")
print(f"  Level 4 (Elo):         {overall:.4f}")
print()

# ============================================================
# STEP 6: Show current 2026 Elo ratings
# ============================================================
print("=" * 60)
print("STEP 6: Current 2026 Elo ratings...")
print("=" * 60)

# Get all teams active in 2026
games_2026 = all_games[all_games["Season"] == 2026]
teams_2026 = set(games_2026["WTeamID"]) | set(games_2026["LTeamID"])

elo_2026 = [(tid, final_ratings.get(tid, 1500)) for tid in teams_2026]
elo_2026.sort(key=lambda x: -x[1])

# Separate men's and women's top 15
men_elo = [(tid, elo) for tid, elo in elo_2026 if tid < 2000]
women_elo = [(tid, elo) for tid, elo in elo_2026 if tid >= 3000]

print("\nTop 15 Men's Teams by Elo:")
for i, (tid, elo) in enumerate(men_elo[:15], 1):
    name = team_name_map.get(tid, str(tid))
    print(f"  {i:2d}. {name:20s}  Elo: {elo:.0f}")

print("\nTop 15 Women's Teams by Elo:")
for i, (tid, elo) in enumerate(women_elo[:15], 1):
    name = team_name_map.get(tid, str(tid))
    print(f"  {i:2d}. {name:20s}  Elo: {elo:.0f}")

print()

# ============================================================
# STEP 7: Generate 2026 predictions
# ============================================================
print("=" * 60)
print("STEP 7: Generating 2026 predictions...")
print("=" * 60)

submission = pd.read_csv(f"{DATA}SampleSubmissionStage2.csv")

def predict_matchup(row_id):
    parts = row_id.split("_")
    team1, team2 = int(parts[1]), int(parts[2])
    r1 = final_ratings.get(team1, 1500)
    r2 = final_ratings.get(team2, 1500)
    # Use neutral site prediction (no home advantage) for tournament
    return expected_score(r1, r2)

print("Generating predictions...")
submission["Pred"] = submission["ID"].apply(predict_matchup)

print(f"Predictions generated: {len(submission):,}")
print(f"Range: {submission['Pred'].min():.4f} to {submission['Pred'].max():.4f}")
print(f"Mean:  {submission['Pred'].mean():.4f}")
print()

# Show some marquee matchups
print("Some marquee 2026 predictions:")
marquee_teams = [t[0] for t in men_elo[:6]]  # top 6 men's teams
for i in range(len(marquee_teams)):
    for j in range(i+1, len(marquee_teams)):
        t1 = min(marquee_teams[i], marquee_teams[j])
        t2 = max(marquee_teams[i], marquee_teams[j])
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
output_path = "submission_level4_elo.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to: {output_path}")
print()
print("Done! 🏀")
