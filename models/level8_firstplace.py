"""
Level 8: First-Place Inspired Model
=====================================
Incorporates key ideas from modeh7's 2025 1st place solution:

1. SYMMETRIC DOUBLING: Each game entered twice (swap T1/T2). Prevents the model
   from learning a spurious "lower ID advantage."

2. OVERTIME NORMALIZATION: Box scores adjusted to 40-minute equivalents so OT
   games don't inflate averages.

3. GLM TEAM QUALITY: Fits a Gaussian GLM per season: PointDiff ~ T1_Team + T2_Team.
   Unlike Elo (sequential updates), GLM sees ALL games simultaneously and produces
   a "points above average" coefficient for each team. This is a Bradley-Terry model
   variant using point differential instead of binary outcomes.

4. PREDICT POINT DIFFERENTIAL: XGBoost regression on margin of victory instead of
   binary win/loss classification. A 20-point blowout carries more signal than a
   1-point squeaker. This produces smoother, better-calibrated probabilities.

5. SPLINE CALIBRATION: Convert predicted point diff → win probability using a
   fitted spline curve (non-parametric), rather than assuming a logistic shape.

6. LOSO ENSEMBLE: Leave-One-Season-Out cross-validation. Train one model per
   held-out season, then average ALL models' predictions at inference. This is
   a free ensemble that reduces variance.
"""

import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline
from xgboost import DMatrix, train as xgb_train
from sklearn.metrics import brier_score_loss, mean_absolute_error

warnings.filterwarnings("ignore")

DATA = "data/"
MIN_SEASON = 2003  # earliest season with detailed box scores

# ============================================================
# STEP 1: Load data
# ============================================================
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

m_detail = pd.read_csv(f"{DATA}MRegularSeasonDetailedResults.csv")
w_detail = pd.read_csv(f"{DATA}WRegularSeasonDetailedResults.csv")
m_tourney_detail = pd.read_csv(f"{DATA}MNCAATourneyDetailedResults.csv")
w_tourney_detail = pd.read_csv(f"{DATA}WNCAATourneyDetailedResults.csv")
m_seeds = pd.read_csv(f"{DATA}MNCAATourneySeeds.csv")
w_seeds = pd.read_csv(f"{DATA}WNCAATourneySeeds.csv")

regular_results = pd.concat([m_detail, w_detail])
tourney_results = pd.concat([m_tourney_detail, w_tourney_detail])
seeds = pd.concat([m_seeds, w_seeds])

regular_results = regular_results[regular_results["Season"] >= MIN_SEASON]
tourney_results = tourney_results[tourney_results["Season"] >= MIN_SEASON]
seeds = seeds[seeds["Season"] >= MIN_SEASON]
seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))

# Team names for display
m_teams = pd.read_csv(f"{DATA}MTeams.csv")
w_teams = pd.read_csv(f"{DATA}WTeams.csv")
team_name_map = dict(zip(
    pd.concat([m_teams[["TeamID","TeamName"]], w_teams[["TeamID","TeamName"]]])["TeamID"],
    pd.concat([m_teams[["TeamID","TeamName"]], w_teams[["TeamID","TeamName"]]])["TeamName"]
))

print(f"Regular season detailed games: {len(regular_results):,}")
print(f"Tournament detailed games:     {len(tourney_results):,}")
print(f"Seasons: {MIN_SEASON} to {regular_results['Season'].max()}")
print()

# ============================================================
# STEP 2: Prepare data (symmetric doubling + OT normalization)
# ============================================================
print("=" * 60)
print("STEP 2: Preparing data (symmetric doubling + OT normalization)...")
print("=" * 60)

def prepare_data(df):
    """Double the dataset with swapped T1/T2 positions, normalize for OT."""
    cols = ["Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "NumOT",
            "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR",
            "LAst", "LTO", "LStl", "LBlk", "LPF",
            "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR",
            "WAst", "WTO", "WStl", "WBlk", "WPF"]
    df = df[cols].copy()

    # OT normalization: scale counting stats to 40-minute equivalents
    adjot = (40 + 5 * df["NumOT"]) / 40
    no_adjust = ["Season", "DayNum", "NumOT", "LTeamID", "WTeamID"]
    stat_cols = [c for c in cols if c not in no_adjust]
    for col in stat_cols:
        df[col] = df[col] / adjot

    # Create two copies: original and swapped
    dfswap = df.copy()
    df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in df.columns]
    dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in dfswap.columns]

    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output["PointDiff"] = output["T1_Score"] - output["T2_Score"]
    output["win"] = (output["PointDiff"] > 0).astype(int)
    output["men_women"] = (output["T1_TeamID"].astype(str).str.startswith("1")).astype(int)
    return output

regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)

print(f"Regular data (doubled): {len(regular_data):,}")
print(f"Tourney data (doubled): {len(tourney_data):,}")
print()

# ============================================================
# STEP 3: Easy features (seeds)
# ============================================================
print("=" * 60)
print("STEP 3: Adding seed features...")
print("=" * 60)

seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]]
tourney_data = tourney_data.merge(seeds_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(seeds_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

print(f"Tourney data with seeds: {len(tourney_data):,}")
print()

# ============================================================
# STEP 4: Medium features (season-average box scores)
# ============================================================
print("=" * 60)
print("STEP 4: Computing season-average box scores...")
print("=" * 60)

boxcols = [
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
    "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
    "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
    "PointDiff",
]

# Season averages: because data is doubled, T1 stats = team's offensive stats,
# T2 stats = what opponents did against this team (defensive quality)
ss = regular_data.groupby(["Season", "T1_TeamID"])[boxcols].mean().reset_index()

ss_T1 = ss.copy()
ss_T1.columns = ["T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in ss_T1.columns]
ss_T1 = ss_T1.rename({"T1_avg_Season": "Season", "T1_avg_TeamID": "T1_TeamID"}, axis=1)

ss_T2 = ss.copy()
ss_T2.columns = ["T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in ss_T2.columns]
ss_T2 = ss_T2.rename({"T2_avg_Season": "Season", "T2_avg_TeamID": "T2_TeamID"}, axis=1)

tourney_data = tourney_data.merge(ss_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(ss_T2, on=["Season", "T2_TeamID"], how="left")

print(f"Box score features added. Columns: {len(tourney_data.columns)}")
print()

# ============================================================
# STEP 5: Hard features (Elo)
# ============================================================
print("=" * 60)
print("STEP 5: Computing Elo ratings...")
print("=" * 60)

def compute_elo_all_seasons(regular_data, seeds):
    base_elo, elo_width, k_factor = 1000, 400, 100

    def expected_result(elo_a, elo_b):
        return 1.0 / (1 + 10 ** ((elo_b - elo_a) / elo_width))

    all_elos = []
    for season in sorted(seeds["Season"].unique()):
        ss = regular_data[(regular_data["Season"] == season) & (regular_data["win"] == 1)].reset_index(drop=True)
        teams = set(ss["T1_TeamID"]) | set(ss["T2_TeamID"])
        elo = {t: base_elo for t in teams}

        for _, row in ss.iterrows():
            w, l = int(row["T1_TeamID"]), int(row["T2_TeamID"])
            exp_w = expected_result(elo.get(w, base_elo), elo.get(l, base_elo))
            change = k_factor * (1 - exp_w)
            elo[w] = elo.get(w, base_elo) + change
            elo[l] = elo.get(l, base_elo) - change

        for tid, rating in elo.items():
            all_elos.append({"Season": season, "TeamID": tid, "elo": rating})

    return pd.DataFrame(all_elos)

elos = compute_elo_all_seasons(regular_data, seeds)

elos_T1 = elos.rename(columns={"TeamID": "T1_TeamID", "elo": "T1_elo"})
elos_T2 = elos.rename(columns={"TeamID": "T2_TeamID", "elo": "T2_elo"})
tourney_data = tourney_data.merge(elos_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(elos_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]

print(f"Elo ratings computed for {len(elos):,} team-seasons")
print()

# ============================================================
# STEP 6: Hardest features (GLM team quality)
# ============================================================
print("=" * 60)
print("STEP 6: Computing GLM team quality...")
print("=" * 60)

# Identify tournament teams + teams that beat them
seeds_T1_st = seeds_T1.copy()
seeds_T1_st["ST1"] = seeds_T1_st["Season"].astype(str) + "/" + seeds_T1_st["T1_TeamID"].astype(str)
seeds_T2_st = seeds_T2.copy()
seeds_T2_st["ST2"] = seeds_T2_st["Season"].astype(str) + "/" + seeds_T2_st["T2_TeamID"].astype(str)

regular_data["ST1"] = regular_data["Season"].astype(int).astype(str) + "/" + regular_data["T1_TeamID"].astype(int).astype(str)
regular_data["ST2"] = regular_data["Season"].astype(int).astype(str) + "/" + regular_data["T2_TeamID"].astype(int).astype(str)

st = set(seeds_T1_st["ST1"]) | set(seeds_T2_st["ST2"])
# Add non-tourney teams that beat at least one tourney team
st = st | set(regular_data[(regular_data["T1_Score"] > regular_data["T2_Score"]) &
                           (regular_data["ST2"].isin(st))]["ST1"])

def team_quality(season, men_women, dt):
    """Fit GLM: PointDiff ~ T1_Team + T2_Team for one season/gender."""
    subset = dt[(dt["Season"] == season) & (dt["men_women"] == men_women)].copy()
    if len(subset) < 50:
        return pd.DataFrame(columns=["TeamID", "quality", "Season"])

    try:
        formula = "PointDiff ~ -1 + T1_TeamID + T2_TeamID"
        glm = sm.GLM.from_formula(
            formula=formula,
            data=subset,
            family=sm.families.Gaussian(),
        ).fit()

        # Extract T1 coefficients (team strength)
        t1_params = glm.params[glm.params.index.str.startswith("T1_")]
        quality = pd.DataFrame({"TeamID_raw": t1_params.index, "quality": t1_params.values})
        quality["Season"] = season
        # Parse TeamID from coefficient names like "T1_TeamID[T.1181]" or "T1_TeamID[1181]"
        quality["TeamID"] = quality["TeamID_raw"].str.extract(r'(\d{4})').astype(int)
        return quality[["TeamID", "quality", "Season"]]
    except Exception as e:
        print(f"    GLM failed for {season}/{men_women}: {e}")
        return pd.DataFrame(columns=["TeamID", "quality", "Season"])

# Prepare data for GLM (collapse non-relevant teams to "0000")
dt = regular_data[regular_data["ST1"].isin(st) | regular_data["ST2"].isin(st)].copy()
# Convert TeamID to int first (handles float from OT normalization), then to str
dt["T1_TeamID"] = dt["T1_TeamID"].round().astype(int).astype(str)
dt["T2_TeamID"] = dt["T2_TeamID"].round().astype(int).astype(str)
dt.loc[~dt["ST1"].isin(st), "T1_TeamID"] = "0000"
dt.loc[~dt["ST2"].isin(st), "T2_TeamID"] = "0000"

print("Fitting GLM per season/gender (this takes ~1-2 minutes)...")
glm_quality = []
for s in sorted(seeds["Season"].unique()):
    if s >= 2010:
        glm_quality.append(team_quality(s, 0, dt))  # women
    if s >= 2003:
        glm_quality.append(team_quality(s, 1, dt))  # men
    if s % 5 == 0:
        print(f"  ...processed through {s}")

glm_quality = pd.concat(glm_quality).reset_index(drop=True)

glm_T1 = glm_quality.rename(columns={"TeamID": "T1_TeamID", "quality": "T1_quality"})
glm_T2 = glm_quality.rename(columns={"TeamID": "T2_TeamID", "quality": "T2_quality"})
tourney_data = tourney_data.merge(glm_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(glm_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["diff_quality"] = tourney_data["T1_quality"] - tourney_data["T2_quality"]

print(f"GLM quality computed for {len(glm_quality):,} team-seasons")
print()

# ============================================================
# STEP 7: Define features and train XGBoost (LOSO)
# ============================================================
print("=" * 60)
print("STEP 7: Training XGBoost with LOSO...")
print("=" * 60)

# Curated feature set (following 1st place approach)
features = [
    "men_women",
    "T1_seed", "T2_seed", "Seed_diff",
    "T1_avg_Score", "T1_avg_FGA", "T1_avg_OR", "T1_avg_DR",
    "T1_avg_Blk", "T1_avg_PF",
    "T1_avg_opponent_FGA", "T1_avg_opponent_Blk", "T1_avg_opponent_PF",
    "T1_avg_PointDiff",
    "T2_avg_Score", "T2_avg_FGA", "T2_avg_OR", "T2_avg_DR",
    "T2_avg_Blk", "T2_avg_PF",
    "T2_avg_opponent_FGA", "T2_avg_opponent_Blk", "T2_avg_opponent_PF",
    "T2_avg_PointDiff",
    "T1_elo", "T2_elo", "elo_diff",
    "T1_quality", "T2_quality",
]

print(f"Features: {len(features)}")

# XGBoost params (from 1st place solution)
param = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "eta": 0.0093,
    "subsample": 0.6,
    "colsample_bynode": 0.8,
    "num_parallel_tree": 2,
    "min_child_weight": 4,
    "max_depth": 4,
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_bin": 38,
}
num_rounds = 704

models = {}
oof_preds = []
oof_targets = []
oof_seasons = []

seasons = sorted(tourney_data["Season"].unique())
for oof_season in seasons:
    train_mask = tourney_data["Season"] != oof_season
    val_mask = tourney_data["Season"] == oof_season

    X_tr = tourney_data.loc[train_mask, features].values
    y_tr = tourney_data.loc[train_mask, "PointDiff"].values
    X_val = tourney_data.loc[val_mask, features].values
    y_val = tourney_data.loc[val_mask, "PointDiff"].values

    dtrain = DMatrix(X_tr, label=y_tr)
    dval = DMatrix(X_val, label=y_val)

    models[oof_season] = xgb_train(params=param, dtrain=dtrain, num_boost_round=num_rounds)

    preds = models[oof_season].predict(dval)
    mae = mean_absolute_error(y_val, preds)
    print(f"  {oof_season}: MAE = {mae:.2f}")

    oof_preds.extend(preds.tolist())
    oof_targets.extend(y_val.tolist())
    oof_seasons.extend([oof_season] * len(y_val))

avg_mae = mean_absolute_error(oof_targets, oof_preds)
print(f"\n  Average MAE: {avg_mae:.2f}")
print()

# ============================================================
# STEP 8: Spline calibration (point diff → probability)
# ============================================================
print("=" * 60)
print("STEP 8: Fitting spline calibration...")
print("=" * 60)

CLIP_DIFF = 25  # clip point diffs before spline

# Sort by predicted diff, fit spline on (pred_diff, actual_win)
dat = sorted(zip(oof_preds, [int(t > 0) for t in oof_targets]), key=lambda x: x[0])
pred_sorted, label_sorted = zip(*dat)

spline_model = UnivariateSpline(
    np.clip(pred_sorted, -CLIP_DIFF, CLIP_DIFF),
    label_sorted,
    k=5
)

# Evaluate: convert OOF point-diff predictions to probabilities via spline
spline_probs = np.clip(spline_model(np.clip(oof_preds, -CLIP_DIFF, CLIP_DIFF)), 0.01, 0.99)
oof_labels = [int(t > 0) for t in oof_targets]
overall_brier = brier_score_loss(oof_labels, spline_probs)

print(f"Overall Brier (LOSO + spline): {overall_brier:.5f}")
print()

# Per-season breakdown
print("Per-season Brier scores:")
eval_years = [2022, 2023, 2024, 2025]
for season in seasons:
    mask = [s == season for s in oof_seasons]
    if sum(mask) == 0:
        continue
    s_probs = spline_probs[np.array(mask)]
    s_labels = np.array(oof_labels)[np.array(mask)]
    s_brier = brier_score_loss(s_labels, s_probs)
    marker = " ←" if season in eval_years else ""
    print(f"  {season}: {s_brier:.5f} ({sum(mask)//2} games){marker}")

# Brier for just 2022-2025
mask_eval = [s in eval_years for s in oof_seasons]
eval_brier = brier_score_loss(
    np.array(oof_labels)[mask_eval],
    spline_probs[np.array(mask_eval)]
)
print(f"\n  2022-2025 Brier: {eval_brier:.5f}")
print()

print("Comparison (2022-2025 evaluation):")
print(f"  Level 5 (LR combined):    0.0979")
print(f"  Level 6 (+ POM):          0.0967")
print(f"  Level 7 (LR+XGB):         0.0964")
print(f"  Level 8 (1st place):      {eval_brier:.4f}")
print()

# ============================================================
# STEP 9: Generate 2026 predictions
# ============================================================
print("=" * 60)
print("STEP 9: Generating 2026 predictions...")
print("=" * 60)

submission = pd.read_csv(f"{DATA}SampleSubmissionStage2.csv")
X = submission.copy()
X["Season"] = X["ID"].apply(lambda t: int(t.split("_")[0]))
X["T1_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[1]))
X["T2_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[2]))
X["men_women"] = (X["T1_TeamID"].astype(str).str.startswith("1")).astype(int)

# Merge all features
X = X.merge(ss_T1, on=["Season", "T1_TeamID"], how="left")
X = X.merge(ss_T2, on=["Season", "T2_TeamID"], how="left")
X = X.merge(seeds_T1, on=["Season", "T1_TeamID"], how="left")
X = X.merge(seeds_T2, on=["Season", "T2_TeamID"], how="left")
X = X.merge(glm_T1, on=["Season", "T1_TeamID"], how="left")
X = X.merge(glm_T2, on=["Season", "T2_TeamID"], how="left")
X = X.merge(elos_T1, on=["Season", "T1_TeamID"], how="left")
X = X.merge(elos_T2, on=["Season", "T2_TeamID"], how="left")
X["Seed_diff"] = X["T2_seed"] - X["T1_seed"]
X["elo_diff"] = X["T1_elo"] - X["T2_elo"]
X["diff_quality"] = X["T1_quality"] - X["T2_quality"]

print(f"Submission rows: {len(X):,}")
print(f"NaN in features: {X[features].isna().sum().sum()}")

# Ensemble: average predictions from all LOSO models
dtest = DMatrix(X[features].values)
all_preds = []
for oof_season in seasons:
    margin_preds = models[oof_season].predict(dtest)
    probs = np.clip(spline_model(np.clip(margin_preds, -CLIP_DIFF, CLIP_DIFF)), 0.01, 0.99)
    all_preds.append(probs)

X["Pred"] = np.mean(all_preds, axis=0)

print(f"Predictions generated: {len(X):,}")
print(f"Range: {X['Pred'].min():.4f} to {X['Pred'].max():.4f}")
print(f"Mean:  {X['Pred'].mean():.4f}")
print()

# Show top matchups
print("Top men's matchup predictions:")
top_seeds = X[(X["T1_seed"] == 1) & (X["T2_seed"] == 1) & (X["men_women"] == 1)]
for _, row in top_seeds.iterrows():
    n1 = team_name_map.get(row["T1_TeamID"], str(int(row["T1_TeamID"])))
    n2 = team_name_map.get(row["T2_TeamID"], str(int(row["T2_TeamID"])))
    print(f"  {n1} vs {n2}: P({n1} wins) = {row['Pred']:.1%}")

# Some 1v2 matchups
print("\nSample 1-seed vs 2-seed matchups:")
matchups_12 = X[(X["T1_seed"] == 1) & (X["T2_seed"] == 2) & (X["men_women"] == 1)]
for _, row in matchups_12.head(6).iterrows():
    n1 = team_name_map.get(row["T1_TeamID"], str(int(row["T1_TeamID"])))
    n2 = team_name_map.get(row["T2_TeamID"], str(int(row["T2_TeamID"])))
    print(f"  {n1} vs {n2}: P({n1} wins) = {row['Pred']:.1%}")
print()

# Save
output_path = "submission_level8_firstplace.csv"
X[["ID", "Pred"]].to_csv(output_path, index=False)
print(f"Submission saved to: {output_path}")
print()
print("Done! 🏀")
