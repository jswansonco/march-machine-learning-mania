"""
Level 7: XGBoost + Box Score Features
=======================================
Two big changes:

1. BOX SCORE FEATURES:
   The "Detailed Results" files contain per-game stats: field goals, 3-pointers,
   free throws, rebounds, assists, turnovers, steals, blocks. We compute per-game
   EFFICIENCY metrics for each team:
     - Effective FG% (eFG%): accounts for 3-pointers being worth more
     - Turnover rate: turnovers per possession
     - Offensive rebound %: % of available offensive rebounds grabbed
     - Free throw rate: free throws attempted per field goal attempted
   These are the "Four Factors" — widely used in basketball analytics as the
   best summary of what wins games.

2. XGBOOST (instead of logistic regression):
   Logistic regression can only learn LINEAR relationships (feature × weight).
   XGBoost builds an ENSEMBLE of DECISION TREES that can capture:
     - Non-linear effects: "seed matters a lot for 1-seeds but less for 8-seeds"
     - Interactions: "high Elo + easy schedule = overrated" (without us having to
       manually create that feature)
     - Diminishing returns: the difference between rank #1 and #10 matters more
       than #100 vs #110

   WHAT IS A DECISION TREE?
   A simple flowchart of yes/no questions:
     "Is Elo gap > 200?" → Yes → "Is seed gap > 5?" → Yes → predict 92%
                         → No  → "Is margin gap > 10?" → Yes → predict 68%

   XGBoost builds HUNDREDS of these trees, each one learning from the mistakes
   of the previous ones. The final prediction is the average of all trees.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

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

m_regular_detail = pd.read_csv(f"{DATA}MRegularSeasonDetailedResults.csv")
w_regular_detail = pd.read_csv(f"{DATA}WRegularSeasonDetailedResults.csv")
all_detail = pd.concat([m_regular_detail, w_regular_detail], ignore_index=True)

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

massey = pd.read_csv(f"{DATA}MMasseyOrdinals.csv")

all_games = pd.concat([all_regular, all_tourney], ignore_index=True)
all_games = all_games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

print(f"Regular season games: {len(all_regular):,}")
print(f"Detailed box scores:  {len(all_detail):,}")
print(f"Tournament games:     {len(all_tourney):,}")
print()

# ============================================================
# STEP 2: Compute box score features ("Four Factors")
# ============================================================
print("=" * 60)
print("STEP 2: Computing box score features...")
print("=" * 60)

# For each game, compute stats for both winner and loser, then aggregate per team per season
def compute_box_stats(detail_df):
    """Compute per-team per-season advanced stats from detailed results."""
    team_stats = {}

    for _, row in detail_df.iterrows():
        season = row["Season"]

        for prefix, team_id in [("W", row["WTeamID"]), ("L", row["LTeamID"])]:
            opp_prefix = "L" if prefix == "W" else "W"
            key = (season, team_id)

            if key not in team_stats:
                team_stats[key] = {
                    "FGM": 0, "FGA": 0, "FGM3": 0, "FGA3": 0,
                    "FTM": 0, "FTA": 0, "OR": 0, "DR": 0,
                    "Ast": 0, "TO": 0, "Stl": 0, "Blk": 0,
                    # opponent stats for defensive metrics
                    "OppFGM": 0, "OppFGA": 0, "OppFGM3": 0, "OppFGA3": 0,
                    "OppOR": 0, "OppDR": 0, "OppTO": 0, "OppFTM": 0, "OppFTA": 0,
                    "Games": 0
                }

            s = team_stats[key]
            s["FGM"] += row[f"{prefix}FGM"]
            s["FGA"] += row[f"{prefix}FGA"]
            s["FGM3"] += row[f"{prefix}FGM3"]
            s["FGA3"] += row[f"{prefix}FGA3"]
            s["FTM"] += row[f"{prefix}FTM"]
            s["FTA"] += row[f"{prefix}FTA"]
            s["OR"] += row[f"{prefix}OR"]
            s["DR"] += row[f"{prefix}DR"]
            s["Ast"] += row[f"{prefix}Ast"]
            s["TO"] += row[f"{prefix}TO"]
            s["Stl"] += row[f"{prefix}Stl"]
            s["Blk"] += row[f"{prefix}Blk"]
            s["OppFGM"] += row[f"{opp_prefix}FGM"]
            s["OppFGA"] += row[f"{opp_prefix}FGA"]
            s["OppFGM3"] += row[f"{opp_prefix}FGM3"]
            s["OppFGA3"] += row[f"{opp_prefix}FGA3"]
            s["OppOR"] += row[f"{opp_prefix}OR"]
            s["OppDR"] += row[f"{opp_prefix}DR"]
            s["OppTO"] += row[f"{opp_prefix}TO"]
            s["OppFTM"] += row[f"{opp_prefix}FTM"]
            s["OppFTA"] += row[f"{opp_prefix}FTA"]
            s["Games"] += 1

    # Convert totals to per-game and efficiency metrics
    box_lookup = {}
    for (season, tid), s in team_stats.items():
        g = s["Games"]
        if g == 0 or s["FGA"] == 0:
            continue

        # Possessions estimate (standard formula)
        poss = s["FGA"] - s["OR"] + s["TO"] + 0.475 * s["FTA"]
        opp_poss = s["OppFGA"] - s["OppOR"] + s["OppTO"] + 0.475 * s["OppFTA"]
        avg_poss = (poss + opp_poss) / 2 if (poss + opp_poss) > 0 else 1

        # Offensive Four Factors
        efg_pct = (s["FGM"] + 0.5 * s["FGM3"]) / s["FGA"]  # effective FG%
        to_rate = s["TO"] / avg_poss if avg_poss > 0 else 0  # turnover rate
        or_pct = s["OR"] / (s["OR"] + s["OppDR"]) if (s["OR"] + s["OppDR"]) > 0 else 0
        ft_rate = s["FTA"] / s["FGA"]  # free throw rate

        # Defensive Four Factors (opponent's efficiency — lower is better for us)
        opp_efg = (s["OppFGM"] + 0.5 * s["OppFGM3"]) / s["OppFGA"] if s["OppFGA"] > 0 else 0.5
        opp_to_rate = s["OppTO"] / avg_poss if avg_poss > 0 else 0
        opp_or_pct = s["OppOR"] / (s["OppOR"] + s["DR"]) if (s["OppOR"] + s["DR"]) > 0 else 0
        opp_ft_rate = s["OppFTA"] / s["OppFGA"] if s["OppFGA"] > 0 else 0

        # Additional useful stats
        fg3_pct = s["FGM3"] / s["FGA3"] if s["FGA3"] > 0 else 0
        ft_pct = s["FTM"] / s["FTA"] if s["FTA"] > 0 else 0
        ast_rate = s["Ast"] / g
        stl_rate = s["Stl"] / g
        blk_rate = s["Blk"] / g

        # Tempo (possessions per game)
        tempo = avg_poss / g

        box_lookup[(season, tid)] = {
            # Offensive
            "eFG": efg_pct,
            "TORate": to_rate,
            "ORPct": or_pct,
            "FTRate": ft_rate,
            "FG3Pct": fg3_pct,
            "FTPct": ft_pct,
            "AstPerG": ast_rate,
            # Defensive
            "OppeFG": opp_efg,
            "OppTORate": opp_to_rate,
            "OppORPct": opp_or_pct,
            "OppFTRate": opp_ft_rate,
            # Other
            "StlPerG": stl_rate,
            "BlkPerG": blk_rate,
            "Tempo": tempo,
        }

    return box_lookup

print("Computing box score stats (this takes a moment)...")
box_lookup = compute_box_stats(all_detail)
print(f"Teams with box score stats: {len(box_lookup):,}")

# Show example
print("\n2026 example — Duke:")
duke_id = 1181
duke_box = box_lookup.get((2026, duke_id), {})
if duke_box:
    print(f"  eFG%:    {duke_box['eFG']:.1%} (effective shooting)")
    print(f"  TO Rate: {duke_box['TORate']:.1%} (turnovers per possession)")
    print(f"  OR%:     {duke_box['ORPct']:.1%} (offensive rebound rate)")
    print(f"  FT Rate: {duke_box['FTRate']:.3f} (FTA per FGA)")
    print(f"  Opp eFG: {duke_box['OppeFG']:.1%} (opponent shooting — lower = better defense)")
    print(f"  Tempo:   {duke_box['Tempo']:.1f} possessions/game")
print()

# ============================================================
# STEP 3: Compute Elo + season stats (same as Level 6)
# ============================================================
print("=" * 60)
print("STEP 3: Computing Elo + season stats...")
print("=" * 60)

def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

def run_elo(games, K=32, home_advantage=0, margin_factor=0.8, season_reversion=0.4,
            initial_rating=1500):
    ratings = {}
    current_season = None
    season_snapshots = {}
    for _, game in games.iterrows():
        season, winner, loser = game["Season"], game["WTeamID"], game["LTeamID"]
        wloc, daynum = game["WLoc"], game["DayNum"]
        if season != current_season:
            if current_season is not None:
                for team, rating in ratings.items():
                    season_snapshots[(current_season, team)] = rating
                for team in ratings:
                    ratings[team] = initial_rating + (ratings[team] - initial_rating) * (1 - season_reversion)
            current_season = season
        if winner not in ratings: ratings[winner] = initial_rating
        if loser not in ratings: ratings[loser] = initial_rating
        if daynum > 132:
            for tid in [winner, loser]:
                if (season, tid) not in season_snapshots:
                    season_snapshots[(season, tid)] = ratings[tid]
        r_w, r_l = ratings[winner], ratings[loser]
        if wloc == "H": pred = expected_score(r_w + home_advantage, r_l)
        elif wloc == "A": pred = expected_score(r_w, r_l + home_advantage)
        else: pred = expected_score(r_w, r_l)
        mov_mult = np.log(1 + game["WScore"] - game["LScore"]) * margin_factor if margin_factor > 0 else 1.0
        update = K * mov_mult * (1 - pred)
        ratings[winner] += update
        ratings[loser] -= update
    for team, rating in ratings.items():
        season_snapshots[(current_season, team)] = rating
    return ratings, season_snapshots

print("Running Elo...")
current_ratings, elo_snapshots = run_elo(all_games)

# Season stats
records = {}
for _, row in all_regular.iterrows():
    season, winner, loser = row["Season"], row["WTeamID"], row["LTeamID"]
    wscore, lscore = row["WScore"], row["LScore"]
    for tid, pf, pa, won in [(winner, wscore, lscore, True), (loser, lscore, wscore, False)]:
        key = (season, tid)
        if key not in records:
            records[key] = {"W": 0, "L": 0, "PF": 0, "PA": 0, "G": 0, "Opps": []}
        records[key]["W"] += int(won)
        records[key]["L"] += int(not won)
        records[key]["PF"] += pf
        records[key]["PA"] += pa
        records[key]["G"] += 1
        records[key]["Opps"].append(loser if won else winner)

winpct_lookup = {k: r["W"]/r["G"] for k, r in records.items()}
stats_lookup = {}
for (season, tid), r in records.items():
    opp_wp = [winpct_lookup.get((season, o), 0.5) for o in r["Opps"]]
    stats_lookup[(season, tid)] = {
        "WinPct": r["W"]/r["G"], "SOS": np.mean(opp_wp),
        "ScoringMargin": (r["PF"]-r["PA"])/r["G"],
    }

seed_lookup = {(r["Season"], r["TeamID"]): r["SeedNum"] for _, r in all_seeds.iterrows()}

# Massey (POM only) — get final ranking day per season per system
max_days = massey.groupby(["Season","SystemName"])["RankingDayNum"].max().reset_index()
max_days.columns = ["Season","SystemName","MaxDay"]
final_massey = massey.merge(max_days, on=["Season","SystemName"])
final_massey = final_massey[final_massey["RankingDayNum"] == final_massey["MaxDay"]]
pom = final_massey[final_massey["SystemName"] == "POM"][["Season","TeamID","OrdinalRank"]]
pom_lookup = {(r["Season"], r["TeamID"]): r["OrdinalRank"] for _, r in pom.iterrows()}

print(f"Stats: {len(stats_lookup):,}  Elo: {len(elo_snapshots):,}  Box: {len(box_lookup):,}  POM: {len(pom_lookup):,}")
print()

# ============================================================
# STEP 4: Build training data with ALL features
# ============================================================
print("=" * 60)
print("STEP 4: Building training data...")
print("=" * 60)

DEFAULT_BOX = {k: 0.0 for k in ["eFG","TORate","ORPct","FTRate","FG3Pct","FTPct","AstPerG",
               "OppeFG","OppTORate","OppORPct","OppFTRate","StlPerG","BlkPerG","Tempo"]}
DEFAULT_RANK = 175

box_features = ["eFG", "TORate", "ORPct", "FTRate", "OppeFG", "OppTORate",
                "OppORPct", "OppFTRate", "FG3Pct", "FTPct", "Tempo"]

training_rows = []
for _, row in all_tourney.iterrows():
    season, wteam, lteam = row["Season"], row["WTeamID"], row["LTeamID"]

    s_w = stats_lookup.get((season, wteam))
    s_l = stats_lookup.get((season, lteam))
    elo_w = elo_snapshots.get((season, wteam))
    elo_l = elo_snapshots.get((season, lteam))
    if s_w is None or s_l is None or elo_w is None or elo_l is None:
        continue

    team1, team2 = min(wteam, lteam), max(wteam, lteam)
    team1_won = 1 if wteam == team1 else 0

    s1, s2 = stats_lookup[(season, team1)], stats_lookup[(season, team2)]
    e1, e2 = elo_snapshots[(season, team1)], elo_snapshots[(season, team2)]
    sd1, sd2 = seed_lookup.get((season, team1), 16), seed_lookup.get((season, team2), 16)
    p1, p2 = pom_lookup.get((season, team1), DEFAULT_RANK), pom_lookup.get((season, team2), DEFAULT_RANK)
    b1 = box_lookup.get((season, team1), DEFAULT_BOX)
    b2 = box_lookup.get((season, team2), DEFAULT_BOX)

    feat = {
        "Season": season, "Team1Won": team1_won,
        "EloDiff": e1 - e2,
        "WinPctDiff": s1["WinPct"] - s2["WinPct"],
        "SOSDiff": s1["SOS"] - s2["SOS"],
        "MarginDiff": s1["ScoringMargin"] - s2["ScoringMargin"],
        "SeedDiff": sd2 - sd1,
        "POMRankDiff": p2 - p1,
    }
    # Add box score diffs
    for bf in box_features:
        feat[f"{bf}Diff"] = b1.get(bf, 0) - b2.get(bf, 0)

    training_rows.append(feat)

train_df = pd.DataFrame(training_rows)
print(f"Training examples: {len(train_df)}")
print()

# ============================================================
# STEP 5: Compare models and feature sets
# ============================================================
print("=" * 60)
print("STEP 5: Comparing models...")
print("=" * 60)

eval_years = [2022, 2023, 2024, 2025]
base_features = ["EloDiff", "WinPctDiff", "SOSDiff", "MarginDiff", "SeedDiff", "POMRankDiff"]
box_diff_features = [f"{bf}Diff" for bf in box_features]
all_features = base_features + box_diff_features

def eval_model(model_class, features, model_kwargs=None, label=""):
    if model_kwargs is None:
        model_kwargs = {}
    season_briers = []
    for season in eval_years:
        train_mask = train_df["Season"] < season
        test_mask = train_df["Season"] == season
        X_tr = train_df.loc[train_mask, features].values
        y_tr = train_df.loc[train_mask, "Team1Won"].values
        X_te = train_df.loc[test_mask, features].values
        y_te = train_df.loc[test_mask, "Team1Won"].values
        if len(X_tr) == 0 or len(X_te) == 0:
            season_briers.append(0.25)
            continue
        m = model_class(**model_kwargs)
        m.fit(X_tr, y_tr)
        preds = m.predict_proba(X_te)[:, 1]
        season_briers.append(np.mean((preds - y_te) ** 2))
    overall = np.mean(season_briers)
    print(f"  {label:<40s} {season_briers[0]:>7.4f} {season_briers[1]:>7.4f} "
          f"{season_briers[2]:>7.4f} {season_briers[3]:>7.4f} {overall:>8.4f}")
    return overall

print(f"{'Model':<42s} {'2022':>7s} {'2023':>7s} {'2024':>7s} {'2025':>7s} {'Overall':>8s}")
print("-" * 80)

# Logistic regression baselines
lr_base = eval_model(LogisticRegression, base_features,
                     {"max_iter": 1000}, "LR: base features (Level 6)")
lr_all = eval_model(LogisticRegression, all_features,
                    {"max_iter": 1000}, "LR: base + box score")

# XGBoost with different configs
xgb_configs = [
    ("XGB: base features", base_features,
     {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,
      "eval_metric": "logloss", "random_state": 42}),
    ("XGB: base + box", all_features,
     {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,
      "eval_metric": "logloss", "random_state": 42}),
    ("XGB: base + box (deeper)", all_features,
     {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
      "eval_metric": "logloss", "random_state": 42}),
    ("XGB: base + box (shallow)", all_features,
     {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.05,
      "eval_metric": "logloss", "random_state": 42}),
    ("XGB: base + box (regularized)", all_features,
     {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
      "reg_alpha": 0.1, "reg_lambda": 1.0,
      "eval_metric": "logloss", "random_state": 42}),
]

best_brier = min(lr_base, lr_all)
best_label = "LR: base features" if lr_base < lr_all else "LR: base + box"
best_model_type = "lr"
best_model_features = base_features if lr_base < lr_all else all_features
best_model_kwargs = {"max_iter": 1000}

for label, feats, kwargs in xgb_configs:
    brier = eval_model(XGBClassifier, feats, kwargs, label)
    if brier < best_brier:
        best_brier = brier
        best_label = label
        best_model_type = "xgb"
        best_model_features = feats
        best_model_kwargs = kwargs

print()
print(f"Best: {best_label} → Brier = {best_brier:.4f}")
print()

# ============================================================
# STEP 6: Ensemble (average LR + XGB)
# ============================================================
print("=" * 60)
print("STEP 6: Testing ensemble (LR + XGB average)...")
print("=" * 60)

# Sometimes averaging a linear and non-linear model beats both
ensemble_briers = []
for season in eval_years:
    train_mask = train_df["Season"] < season
    test_mask = train_df["Season"] == season
    X_tr = train_df.loc[train_mask, all_features].values
    y_tr = train_df.loc[train_mask, "Team1Won"].values
    X_te = train_df.loc[test_mask, all_features].values
    y_te = train_df.loc[test_mask, "Team1Won"].values

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_tr, y_tr)
    lr_preds = lr.predict_proba(X_te)[:, 1]

    xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                         reg_alpha=0.1, reg_lambda=1.0,
                         eval_metric="logloss", random_state=42)
    xgb.fit(X_tr, y_tr)
    xgb_preds = xgb.predict_proba(X_te)[:, 1]

    # Test different blend weights
    best_w = 0.5
    best_b = 1.0
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blend = w * lr_preds + (1-w) * xgb_preds
        b = np.mean((blend - y_te) ** 2)
        if b < best_b:
            best_b = b
            best_w = w

    ensemble_briers.append(best_b)

ensemble_overall = np.mean(ensemble_briers)
print(f"  Ensemble (best LR/XGB blend): {ensemble_overall:.4f}")
print()

# Determine overall best
if ensemble_overall < best_brier:
    print(f"Ensemble wins! Brier = {ensemble_overall:.4f}")
    use_ensemble = True
    final_brier = ensemble_overall
else:
    print(f"Single model wins: {best_label} → Brier = {best_brier:.4f}")
    use_ensemble = False
    final_brier = best_brier

print()
print("Full comparison:")
print(f"  Coin flip:             0.2500")
print(f"  Level 5 (combined):    0.0979")
print(f"  Level 6 (+ POM):       0.0967")
print(f"  Level 7:               {final_brier:.4f}")
print()

# ============================================================
# STEP 7: Train final model(s) and show feature importance
# ============================================================
print("=" * 60)
print("STEP 7: Training final model...")
print("=" * 60)

X_all = train_df[all_features].values
y_all = train_df["Team1Won"].values

final_lr = LogisticRegression(max_iter=1000)
final_lr.fit(X_all, y_all)

final_xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                           reg_alpha=0.1, reg_lambda=1.0,
                           eval_metric="logloss", random_state=42)
final_xgb.fit(X_all, y_all)

# Feature importance from XGBoost
importance = dict(zip(all_features, final_xgb.feature_importances_))
sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

print("XGBoost feature importance:")
for feat, imp in sorted_imp:
    bar = "█" * int(imp * 100)
    print(f"  {feat:20s}: {imp:.3f}  {bar}")
print()

# ============================================================
# STEP 8: Generate 2026 predictions
# ============================================================
print("=" * 60)
print("STEP 8: Generating 2026 predictions...")
print("=" * 60)

submission = pd.read_csv(f"{DATA}SampleSubmissionStage2.csv")

stats_2026 = {tid: s for (season, tid), s in stats_lookup.items() if season == 2026}
seeds_2026 = {tid: sn for (season, tid), sn in seed_lookup.items() if season == 2026}
DEFAULT_STATS = {"WinPct": 0.5, "SOS": 0.5, "ScoringMargin": 0.0}

def get_all_features(team1, team2):
    s1 = stats_2026.get(team1, DEFAULT_STATS)
    s2 = stats_2026.get(team2, DEFAULT_STATS)
    e1 = current_ratings.get(team1, 1500)
    e2 = current_ratings.get(team2, 1500)
    sd1 = seeds_2026.get(team1, 16)
    sd2 = seeds_2026.get(team2, 16)
    p1 = pom_lookup.get((2026, team1), DEFAULT_RANK)
    p2 = pom_lookup.get((2026, team2), DEFAULT_RANK)
    b1 = box_lookup.get((2026, team1), DEFAULT_BOX)
    b2 = box_lookup.get((2026, team2), DEFAULT_BOX)

    feat = {
        "EloDiff": e1 - e2,
        "WinPctDiff": s1["WinPct"] - s2["WinPct"],
        "SOSDiff": s1["SOS"] - s2["SOS"],
        "MarginDiff": s1["ScoringMargin"] - s2["ScoringMargin"],
        "SeedDiff": sd2 - sd1,
        "POMRankDiff": p2 - p1,
    }
    for bf in box_features:
        feat[f"{bf}Diff"] = b1.get(bf, 0) - b2.get(bf, 0)

    return [feat[f] for f in all_features]

print("Generating predictions...")
preds = []
for rid in submission["ID"].values:
    parts = rid.split("_")
    team1, team2 = int(parts[1]), int(parts[2])
    features = np.array(get_all_features(team1, team2)).reshape(1, -1)

    if use_ensemble:
        lr_pred = final_lr.predict_proba(features)[0][1]
        xgb_pred = final_xgb.predict_proba(features)[0][1]
        pred = 0.5 * lr_pred + 0.5 * xgb_pred
    else:
        if best_model_type == "lr":
            pred = final_lr.predict_proba(features)[0][1]
        else:
            pred = final_xgb.predict_proba(features)[0][1]

    preds.append(pred)

submission["Pred"] = preds

print(f"Predictions: {len(submission):,}")
print(f"Range: {submission['Pred'].min():.4f} to {submission['Pred'].max():.4f}")
print(f"Mean:  {submission['Pred'].mean():.4f}")
print()

# Show top matchups
print("Top men's team predictions:")
top_men = sorted(
    [(tid, current_ratings[tid]) for tid in stats_2026 if tid < 2000 and tid in current_ratings],
    key=lambda x: -x[1]
)[:6]
for i in range(len(top_men)):
    for j in range(i+1, len(top_men)):
        t1, t2 = min(top_men[i][0], top_men[j][0]), max(top_men[i][0], top_men[j][0])
        match = submission[submission["ID"] == f"2026_{t1}_{t2}"]
        if len(match) > 0:
            n1, n2 = team_name_map.get(t1, str(t1)), team_name_map.get(t2, str(t2))
            print(f"  {n1} vs {n2}: P({n1} wins) = {match.iloc[0]['Pred']:.1%}")
print()

output_path = "submission_level7_xgboost.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to: {output_path}")
print("Done! 🏀")
