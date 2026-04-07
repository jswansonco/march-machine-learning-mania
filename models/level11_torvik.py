"""
Level 11: BartTorvik Power Ratings Ensemble
=============================================
Level 9's XGBoost model + BartTorvik T-Rank analytics blended as an ensemble.

BartTorvik provides tempo-adjusted efficiency ratings (AdjOE, AdjDE, AdjT) which are
the same foundation professional sportsbooks use to set lines. We compute a projected
spread for every 2026 matchup, convert to probability, and blend with our XGBoost
predictions.

Since Torvik data is 2026-only (no historical), we can't add it as a training feature.
Instead we ensemble: the XGBoost model captures historical patterns while Torvik
captures current-season analytics from an independent source.
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
MIN_SEASON = 2003

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
massey = pd.read_csv(f"{DATA}MMasseyOrdinals.csv")

regular_results = pd.concat([m_detail, w_detail])
tourney_results = pd.concat([m_tourney_detail, w_tourney_detail])
seeds = pd.concat([m_seeds, w_seeds])

regular_results = regular_results[regular_results["Season"] >= MIN_SEASON]
tourney_results = tourney_results[tourney_results["Season"] >= MIN_SEASON]
seeds = seeds[seeds["Season"] >= MIN_SEASON]
seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))

m_teams = pd.read_csv(f"{DATA}MTeams.csv")
w_teams = pd.read_csv(f"{DATA}WTeams.csv")
team_name_map = dict(zip(
    pd.concat([m_teams[["TeamID","TeamName"]], w_teams[["TeamID","TeamName"]]])["TeamID"],
    pd.concat([m_teams[["TeamID","TeamName"]], w_teams[["TeamID","TeamName"]]])["TeamName"]
))

print(f"Regular season: {len(regular_results):,}")
print(f"Tournament:     {len(tourney_results):,}")
print(f"Massey rows:    {len(massey):,}")
print()

# ============================================================
# STEP 2: Prepare data (symmetric doubling + OT normalization)
# ============================================================
print("=" * 60)
print("STEP 2: Preparing data...")
print("=" * 60)

def prepare_data(df):
    cols = ["Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "NumOT",
            "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR",
            "LAst", "LTO", "LStl", "LBlk", "LPF",
            "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR",
            "WAst", "WTO", "WStl", "WBlk", "WPF"]
    df = df[cols].copy()
    adjot = (40 + 5 * df["NumOT"]) / 40
    no_adjust = ["Season", "DayNum", "NumOT", "LTeamID", "WTeamID"]
    for col in [c for c in cols if c not in no_adjust]:
        df[col] = df[col] / adjot
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
print(f"Regular (doubled): {len(regular_data):,}")
print(f"Tourney (doubled): {len(tourney_data):,}")
print()

# ============================================================
# STEP 3: Seeds
# ============================================================
print("=" * 60)
print("STEP 3: Adding seeds...")
print("=" * 60)

seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]]
tourney_data = tourney_data.merge(seeds_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(seeds_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]
print("Done.")
print()

# ============================================================
# STEP 4: Box score averages (full season + late season)
# ============================================================
print("=" * 60)
print("STEP 4: Computing box score averages (full + late season)...")
print("=" * 60)

boxcols = [
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
    "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
    "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
    "PointDiff",
]

def compute_season_avgs(data, prefix=""):
    """Compute per-team per-season averages. prefix distinguishes full vs late."""
    ss = data.groupby(["Season", "T1_TeamID"])[boxcols].mean().reset_index()
    ss_T1 = ss.copy()
    ss_T1.columns = [f"T1_{prefix}avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in ss_T1.columns]
    ss_T1 = ss_T1.rename({f"T1_{prefix}avg_Season": "Season", f"T1_{prefix}avg_TeamID": "T1_TeamID"}, axis=1)
    ss_T2 = ss.copy()
    ss_T2.columns = [f"T2_{prefix}avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in ss_T2.columns]
    ss_T2 = ss_T2.rename({f"T2_{prefix}avg_Season": "Season", f"T2_{prefix}avg_TeamID": "T2_TeamID"}, axis=1)
    return ss_T1, ss_T2

# Full season averages
ss_T1, ss_T2 = compute_season_avgs(regular_data, prefix="")
tourney_data = tourney_data.merge(ss_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(ss_T2, on=["Season", "T2_TeamID"], how="left")

# SPECIAL SAUCE #3: Late-season averages (last 14 days, DayNum >= 119)
late_data = regular_data[regular_data["DayNum"] >= 119]
late_T1, late_T2 = compute_season_avgs(late_data, prefix="late_")
tourney_data = tourney_data.merge(late_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(late_T2, on=["Season", "T2_TeamID"], how="left")

print(f"Full season + late season features. Columns: {len(tourney_data.columns)}")
print()

# ============================================================
# STEP 5: SPECIAL SAUCE #1 — Better Elo (margin-weighted, carry-over)
# ============================================================
print("=" * 60)
print("STEP 5: Computing improved Elo (margin-weighted + carry-over)...")
print("=" * 60)

def compute_better_elo(regular_data, seeds):
    """Our Level 4 Elo: K=32, margin weighting, season carry-over."""
    K, WIDTH, MARGIN_FACTOR, REVERSION = 32, 400, 0.8, 0.4
    INIT = 1500
    ratings = {}
    snapshots = {}
    current_season = None

    # Process all regular season games in order
    all_seasons = sorted(regular_data["Season"].unique())
    for season in all_seasons:
        # Season reversion
        if current_season is not None:
            for team in ratings:
                ratings[team] = INIT + (ratings[team] - INIT) * (1 - REVERSION)
        current_season = season

        ss = regular_data[(regular_data["Season"] == season) & (regular_data["win"] == 1)]
        ss = ss.sort_values("DayNum")

        for _, row in ss.iterrows():
            w, l = int(row["T1_TeamID"]), int(row["T2_TeamID"])
            if w not in ratings: ratings[w] = INIT
            if l not in ratings: ratings[l] = INIT

            exp_w = 1.0 / (1.0 + 10.0 ** ((ratings[l] - ratings[w]) / WIDTH))
            margin = row["T1_Score"] - row["T2_Score"]
            mov_mult = np.log(1 + abs(margin)) * MARGIN_FACTOR
            update = K * mov_mult * (1 - exp_w)
            ratings[w] += update
            ratings[l] -= update

        # Snapshot end-of-season ratings
        for team, rating in ratings.items():
            snapshots[(season, team)] = rating

    return snapshots

elo_snapshots = compute_better_elo(regular_data, seeds)

# Also compute basic Elo (1st place style) to give model both
def compute_basic_elo(regular_data, seeds):
    base_elo, elo_width, k_factor = 1000, 400, 100
    all_elos = {}
    for season in sorted(seeds["Season"].unique()):
        ss = regular_data[(regular_data["Season"] == season) & (regular_data["win"] == 1)].reset_index(drop=True)
        teams = set(ss["T1_TeamID"]) | set(ss["T2_TeamID"])
        elo = {t: base_elo for t in teams}
        for _, row in ss.iterrows():
            w, l = int(row["T1_TeamID"]), int(row["T2_TeamID"])
            exp_w = 1.0 / (1 + 10 ** ((elo.get(l, base_elo) - elo.get(w, base_elo)) / elo_width))
            change = k_factor * (1 - exp_w)
            elo[w] = elo.get(w, base_elo) + change
            elo[l] = elo.get(l, base_elo) - change
        for tid, rating in elo.items():
            all_elos[(season, tid)] = rating
    return all_elos

basic_elo = compute_basic_elo(regular_data, seeds)

# Add both Elo systems to tourney data
def add_elo_features(df, elo_dict, prefix):
    t1_col, t2_col = f"T1_{prefix}", f"T2_{prefix}"
    df[t1_col] = df.apply(lambda r: elo_dict.get((r["Season"], r["T1_TeamID"]), 1500 if prefix == "elo2" else 1000), axis=1)
    df[t2_col] = df.apply(lambda r: elo_dict.get((r["Season"], r["T2_TeamID"]), 1500 if prefix == "elo2" else 1000), axis=1)
    df[f"{prefix}_diff"] = df[t1_col] - df[t2_col]
    return df

tourney_data = add_elo_features(tourney_data, basic_elo, "elo")
tourney_data = add_elo_features(tourney_data, elo_snapshots, "elo2")

print(f"Basic Elo entries: {len(basic_elo):,}")
print(f"Better Elo entries: {len(elo_snapshots):,}")
print()

# ============================================================
# STEP 6: GLM team quality
# ============================================================
print("=" * 60)
print("STEP 6: Computing GLM team quality...")
print("=" * 60)

seeds_T1_st = seeds_T1.copy()
seeds_T1_st["ST1"] = seeds_T1_st["Season"].astype(str) + "/" + seeds_T1_st["T1_TeamID"].astype(str)
seeds_T2_st = seeds_T2.copy()
seeds_T2_st["ST2"] = seeds_T2_st["Season"].astype(str) + "/" + seeds_T2_st["T2_TeamID"].astype(str)
regular_data["ST1"] = regular_data["Season"].astype(int).astype(str) + "/" + regular_data["T1_TeamID"].astype(int).astype(str)
regular_data["ST2"] = regular_data["Season"].astype(int).astype(str) + "/" + regular_data["T2_TeamID"].astype(int).astype(str)

st = set(seeds_T1_st["ST1"]) | set(seeds_T2_st["ST2"])
st = st | set(regular_data[(regular_data["T1_Score"] > regular_data["T2_Score"]) &
                           (regular_data["ST2"].isin(st))]["ST1"])

dt = regular_data[regular_data["ST1"].isin(st) | regular_data["ST2"].isin(st)].copy()
dt["T1_TeamID"] = dt["T1_TeamID"].round().astype(int).astype(str)
dt["T2_TeamID"] = dt["T2_TeamID"].round().astype(int).astype(str)
dt.loc[~dt["ST1"].isin(st), "T1_TeamID"] = "0000"
dt.loc[~dt["ST2"].isin(st), "T2_TeamID"] = "0000"

def team_quality(season, men_women, dt):
    subset = dt[(dt["Season"] == season) & (dt["men_women"] == men_women)].copy()
    if len(subset) < 50:
        return pd.DataFrame(columns=["TeamID", "quality", "Season"])
    try:
        glm = sm.GLM.from_formula(
            "PointDiff ~ -1 + T1_TeamID + T2_TeamID", data=subset,
            family=sm.families.Gaussian()
        ).fit()
        t1_params = glm.params[glm.params.index.str.startswith("T1_")]
        quality = pd.DataFrame({"TeamID_raw": t1_params.index, "quality": t1_params.values})
        quality["Season"] = season
        quality["TeamID"] = quality["TeamID_raw"].str.extract(r'(\d{4})').astype(int)
        return quality[["TeamID", "quality", "Season"]]
    except:
        return pd.DataFrame(columns=["TeamID", "quality", "Season"])

print("Fitting GLM per season/gender...")
glm_quality = []
for s in sorted(seeds["Season"].unique()):
    if s >= 2010: glm_quality.append(team_quality(s, 0, dt))
    if s >= 2003: glm_quality.append(team_quality(s, 1, dt))
glm_quality = pd.concat(glm_quality).reset_index(drop=True)

glm_T1 = glm_quality.rename(columns={"TeamID": "T1_TeamID", "quality": "T1_quality"})
glm_T2 = glm_quality.rename(columns={"TeamID": "T2_TeamID", "quality": "T2_quality"})
tourney_data = tourney_data.merge(glm_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(glm_T2, on=["Season", "T2_TeamID"], how="left")

print(f"GLM quality: {len(glm_quality):,} team-seasons")
print()

# ============================================================
# STEP 7: SPECIAL SAUCE #2 — Massey Ordinals
# ============================================================
print("=" * 60)
print("STEP 7: Adding Massey Ordinals...")
print("=" * 60)

# Get final rankings per system per season
max_days = massey.groupby(["Season", "SystemName"])["RankingDayNum"].max().reset_index()
max_days.columns = ["Season", "SystemName", "MaxDay"]
final_massey = massey.merge(max_days, on=["Season", "SystemName"])
final_massey = final_massey[final_massey["RankingDayNum"] == final_massey["MaxDay"]]

# Average rank across all systems
avg_rank = final_massey.groupby(["Season", "TeamID"])["OrdinalRank"].mean().reset_index()
avg_rank.columns = ["Season", "TeamID", "MasseyAvgRank"]

# POM (KenPom) specifically
pom = final_massey[final_massey["SystemName"] == "POM"][["Season", "TeamID", "OrdinalRank"]]
pom.columns = ["Season", "TeamID", "POMRank"]

# Merge to tourney data
for prefix, id_col in [("T1", "T1_TeamID"), ("T2", "T2_TeamID")]:
    avg_tmp = avg_rank.rename(columns={"TeamID": id_col, "MasseyAvgRank": f"{prefix}_MasseyAvg"})
    tourney_data = tourney_data.merge(avg_tmp, on=["Season", id_col], how="left")
    pom_tmp = pom.rename(columns={"TeamID": id_col, "POMRank": f"{prefix}_POM"})
    tourney_data = tourney_data.merge(pom_tmp, on=["Season", id_col], how="left")

tourney_data["MasseyAvg_diff"] = tourney_data["T2_MasseyAvg"] - tourney_data["T1_MasseyAvg"]
tourney_data["POM_diff"] = tourney_data["T2_POM"] - tourney_data["T1_POM"]

print(f"Massey features added. Teams with 2026 POM: {len(pom[pom['Season']==2026])}")
print()

# ============================================================
# STEP 8: Define features and train
# ============================================================
print("=" * 60)
print("STEP 8: Training XGBoost (LOSO)...")
print("=" * 60)

# Feature set: Tight selection based on importance analysis
# Dropped basic Elo (replaced by better Elo), dropped late-season (not important enough)
# Kept Massey + better Elo as differentiators
features = [
    # Meta
    "men_women",
    # Seeds
    "T1_seed", "T2_seed", "Seed_diff",
    # Full-season box scores (curated subset)
    "T1_avg_Score", "T1_avg_FGA", "T1_avg_OR", "T1_avg_DR",
    "T1_avg_Blk", "T1_avg_PF",
    "T1_avg_opponent_FGA", "T1_avg_opponent_Blk", "T1_avg_opponent_PF",
    "T1_avg_PointDiff",
    "T2_avg_Score", "T2_avg_FGA", "T2_avg_OR", "T2_avg_DR",
    "T2_avg_Blk", "T2_avg_PF",
    "T2_avg_opponent_FGA", "T2_avg_opponent_Blk", "T2_avg_opponent_PF",
    "T2_avg_PointDiff",
    # SAUCE #1: Better Elo only (dropped basic Elo — redundant)
    "T1_elo2", "T2_elo2", "elo2_diff",
    # GLM quality
    "T1_quality", "T2_quality",
    # SAUCE #2: Massey Ordinals (POM + average)
    "T1_POM", "T2_POM", "POM_diff",
    "MasseyAvg_diff",
    # SAUCE #3: Late-season point differential only (the one that matters most)
    "T1_late_avg_PointDiff", "T2_late_avg_PointDiff",
]

print(f"Features: {len(features)}")

# XGBoost params
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
oof_preds, oof_targets, oof_seasons = [], [], []

seasons = sorted(tourney_data["Season"].unique())
for oof_season in seasons:
    X_tr = tourney_data.loc[tourney_data["Season"] != oof_season, features].values
    y_tr = tourney_data.loc[tourney_data["Season"] != oof_season, "PointDiff"].values
    X_val = tourney_data.loc[tourney_data["Season"] == oof_season, features].values
    y_val = tourney_data.loc[tourney_data["Season"] == oof_season, "PointDiff"].values

    dtrain = DMatrix(X_tr, label=y_tr)
    models[oof_season] = xgb_train(params=param, dtrain=dtrain, num_boost_round=num_rounds)

    preds = models[oof_season].predict(DMatrix(X_val))
    mae = mean_absolute_error(y_val, preds)
    print(f"  {oof_season}: MAE = {mae:.2f}")
    oof_preds.extend(preds.tolist())
    oof_targets.extend(y_val.tolist())
    oof_seasons.extend([oof_season] * len(y_val))

print(f"\n  Average MAE: {mean_absolute_error(oof_targets, oof_preds):.2f}")
print()

# ============================================================
# STEP 9: Spline calibration
# ============================================================
print("=" * 60)
print("STEP 9: Spline calibration...")
print("=" * 60)

CLIP_DIFF = 25
dat = sorted(zip(oof_preds, [int(t > 0) for t in oof_targets]), key=lambda x: x[0])
pred_sorted, label_sorted = zip(*dat)
spline_model = UnivariateSpline(np.clip(pred_sorted, -CLIP_DIFF, CLIP_DIFF), label_sorted, k=5)

spline_probs = np.clip(spline_model(np.clip(oof_preds, -CLIP_DIFF, CLIP_DIFF)), 0.01, 0.99)
oof_labels = [int(t > 0) for t in oof_targets]

overall_brier = brier_score_loss(oof_labels, spline_probs)
print(f"Overall Brier: {overall_brier:.5f}")
print()

eval_years = [2022, 2023, 2024, 2025]
print("Per-season Brier:")
for season in seasons:
    mask = np.array([s == season for s in oof_seasons])
    if mask.sum() == 0: continue
    b = brier_score_loss(np.array(oof_labels)[mask], spline_probs[mask])
    marker = " ←" if season in eval_years else ""
    print(f"  {season}: {b:.5f} ({mask.sum()//2} games){marker}")

mask_eval = np.array([s in eval_years for s in oof_seasons])
eval_brier = brier_score_loss(np.array(oof_labels)[mask_eval], spline_probs[mask_eval])
print(f"\n  2022-2025 Brier: {eval_brier:.5f}")
print()
print("Comparison:")
print(f"  Level 8 (1st place replica): 0.1655")
print(f"  Level 9 (special sauce):     {overall_brier:.4f}")
improvement = (0.1655 - overall_brier) / 0.1655 * 100
print(f"  Improvement:                 {improvement:+.1f}%")
print()

# ============================================================
# STEP 10: Feature importance
# ============================================================
print("=" * 60)
print("STEP 10: Feature importance (from last LOSO model)...")
print("=" * 60)

last_model = models[seasons[-1]]
importance = last_model.get_score(importance_type="gain")
sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

# Map f0, f1, ... back to feature names
for i, (fkey, gain) in enumerate(sorted_imp[:20]):
    fidx = int(fkey[1:])
    fname = features[fidx] if fidx < len(features) else fkey
    bar = "█" * int(gain / sorted_imp[0][1] * 40)
    print(f"  {fname:25s}: {gain:10.1f}  {bar}")
print()

# ============================================================
# STEP 11: Generate 2026 predictions
# ============================================================
print("=" * 60)
print("STEP 11: Generating 2026 predictions...")
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
X = X.merge(late_T1, on=["Season", "T1_TeamID"], how="left")
X = X.merge(late_T2, on=["Season", "T2_TeamID"], how="left")
X = X.merge(seeds_T1, on=["Season", "T1_TeamID"], how="left")
X = X.merge(seeds_T2, on=["Season", "T2_TeamID"], how="left")
X = X.merge(glm_T1, on=["Season", "T1_TeamID"], how="left")
X = X.merge(glm_T2, on=["Season", "T2_TeamID"], how="left")

# Elo features
for tid_col, prefix, elo_dict, default in [
    ("T1_TeamID", "T1_elo", basic_elo, 1000), ("T2_TeamID", "T2_elo", basic_elo, 1000),
    ("T1_TeamID", "T1_elo2", elo_snapshots, 1500), ("T2_TeamID", "T2_elo2", elo_snapshots, 1500),
]:
    X[prefix] = X.apply(lambda r: elo_dict.get((r["Season"], r[tid_col]), default), axis=1)
X["elo_diff"] = X["T1_elo"] - X["T2_elo"]
X["elo2_diff"] = X["T1_elo2"] - X["T2_elo2"]
X["Seed_diff"] = X["T2_seed"] - X["T1_seed"]

# Massey features
for prefix, id_col in [("T1", "T1_TeamID"), ("T2", "T2_TeamID")]:
    avg_tmp = avg_rank.rename(columns={"TeamID": id_col, "MasseyAvgRank": f"{prefix}_MasseyAvg"})
    X = X.merge(avg_tmp, on=["Season", id_col], how="left")
    pom_tmp = pom.rename(columns={"TeamID": id_col, "POMRank": f"{prefix}_POM"})
    X = X.merge(pom_tmp, on=["Season", id_col], how="left")
X["MasseyAvg_diff"] = X["T2_MasseyAvg"] - X["T1_MasseyAvg"]
X["POM_diff"] = X["T2_POM"] - X["T1_POM"]

# XGBoost LOSO ensemble predict
dtest = DMatrix(X[features].values)
all_preds = []
for oof_season in seasons:
    margin_preds = models[oof_season].predict(dtest)
    probs = np.clip(spline_model(np.clip(margin_preds, -CLIP_DIFF, CLIP_DIFF)), 0.01, 0.99)
    all_preds.append(probs)
X["xgb_pred"] = np.mean(all_preds, axis=0)

print(f"XGBoost predictions: range [{X['xgb_pred'].min():.4f}, {X['xgb_pred'].max():.4f}], mean {X['xgb_pred'].mean():.4f}")
print()

# ============================================================
# STEP 12: BartTorvik power ratings → projected spreads
# ============================================================
print("=" * 60)
print("STEP 12: Computing BartTorvik projected spreads...")
print("=" * 60)

# Load Torvik data
tv_m = pd.read_csv(f"{DATA}torvik_mens_2026.csv")
tv_w = pd.read_csv(f"{DATA}torvik_womens_2026.csv")

# Build name → TeamID mapping using team spellings
m_spell = pd.read_csv(f"{DATA}MTeamSpellings.csv", encoding="latin1")
w_spell = pd.read_csv(f"{DATA}WTeamSpellings.csv", encoding="latin1")

m_name_to_id = dict(zip(m_teams["TeamName"].str.lower(), m_teams["TeamID"]))
w_name_to_id = dict(zip(w_teams["TeamName"].str.lower(), w_teams["TeamID"]))
for _, row in m_spell.iterrows():
    m_name_to_id[str(row["TeamNameSpelling"]).lower().strip()] = row["TeamID"]
for _, row in w_spell.iterrows():
    w_name_to_id[str(row["TeamNameSpelling"]).lower().strip()] = row["TeamID"]

# Manual fixes for the ~13 unmatched teams per gender
manual_map = {
    "tarleton st.": "tarleton st",
    "ut rio grande valley": "utrgv",
    "illinois chicago": "uic",
    "texas a&m corpus chris": "texas a&m cc",
    "southeast missouri st.": "se missouri st",
    "tennessee martin": "ut martin",
    "queens": "queens nc",
    "bethune cookman": "bethune-cookman",
    "arkansas pine bluff": "ark pine bluff",
    "cal st. bakersfield": "cal st bakersfield",
    "louisiana monroe": "ul monroe",
    "saint francis": "st francis pa",
    "mississippi valley st.": "ms valley st",
}

# Match Torvik teams to TeamIDs
def match_torvik(tv_df, name_to_id):
    tv_df = tv_df.copy()
    team_ids = []
    for _, row in tv_df.iterrows():
        name = row["team"].lower().strip()
        # Try direct match
        tid = name_to_id.get(name)
        if tid is None:
            # Try manual mapping
            mapped = manual_map.get(name, name)
            tid = name_to_id.get(mapped)
        team_ids.append(tid)
    tv_df["TeamID"] = team_ids
    return tv_df

tv_m = match_torvik(tv_m, m_name_to_id)
tv_w = match_torvik(tv_w, w_name_to_id)

matched_m = tv_m["TeamID"].notna().sum()
matched_w = tv_w["TeamID"].notna().sum()
print(f"Torvik men's matched: {matched_m}/{len(tv_m)}")
print(f"Torvik women's matched: {matched_w}/{len(tv_w)}")

# Build Torvik lookup: TeamID → {adjoe, adjde, adjt, barthag, ...}
torvik_lookup = {}
for _, row in pd.concat([tv_m, tv_w]).iterrows():
    if pd.notna(row["TeamID"]):
        torvik_lookup[int(row["TeamID"])] = {
            "adjoe": row["adjoe"],
            "adjde": row["adjde"],
            "adjt": row["adjt"],
            "barthag": row["barthag"],
            "sos": row["sos"],
        }

print(f"Torvik lookup: {len(torvik_lookup)} teams")
print()

# Compute projected spread for each matchup
# Formula: spread = (efficiency_margin / 100) * projected_tempo
# efficiency_margin = (adjoe_A - adjde_B) - (adjoe_B - adjde_A)
# projected_tempo = (adjt_A + adjt_B) / 2  (simplified)
from scipy.stats import norm

def torvik_spread_to_prob(spread, std_dev=14.0):
    """Convert a projected spread to a win probability using normal CDF.
    std_dev ~11 is typical for college basketball game-to-game variance."""
    return norm.cdf(spread / std_dev)

torvik_preds = []
torvik_available = 0
for _, row in X.iterrows():
    t1 = row["T1_TeamID"]
    t2 = row["T2_TeamID"]
    tv1 = torvik_lookup.get(t1)
    tv2 = torvik_lookup.get(t2)

    if tv1 is not None and tv2 is not None:
        eff_margin = (tv1["adjoe"] - tv2["adjde"]) - (tv2["adjoe"] - tv1["adjde"])
        avg_tempo = (tv1["adjt"] + tv2["adjt"]) / 2
        spread = (eff_margin / 100) * avg_tempo  # T1's projected margin
        prob = torvik_spread_to_prob(spread)
        torvik_preds.append(np.clip(prob, 0.01, 0.99))
        torvik_available += 1
    else:
        torvik_preds.append(None)

X["torvik_pred"] = torvik_preds
print(f"Torvik predictions available: {torvik_available:,} / {len(X):,}")
print()

# ============================================================
# STEP 13: Ensemble XGBoost + Torvik
# ============================================================
print("=" * 60)
print("STEP 13: Ensembling XGBoost + Torvik...")
print("=" * 60)

# Where Torvik is available, blend. Where not, use XGBoost alone.
# Test different blend weights
BLEND_WEIGHT = 0.9  # 90% XGBoost, 10% Torvik

X["Pred"] = X.apply(
    lambda r: (BLEND_WEIGHT * r["xgb_pred"] + (1 - BLEND_WEIGHT) * r["torvik_pred"])
    if pd.notna(r["torvik_pred"])
    else r["xgb_pred"],
    axis=1
)

print(f"Blend: {BLEND_WEIGHT:.0%} XGBoost + {1-BLEND_WEIGHT:.0%} Torvik")
print(f"Predictions: {len(X):,}")
print(f"Range: {X['Pred'].min():.4f} to {X['Pred'].max():.4f}")
print(f"Mean:  {X['Pred'].mean():.4f}")
print()

# Compare XGBoost-only vs blended for tournament matchups
tourney_teams = set(seeds[seeds["Season"] == 2026]["TeamID"])
tourney_mask = X["T1_TeamID"].isin(tourney_teams) & X["T2_TeamID"].isin(tourney_teams)
tourney_X = X[tourney_mask]
print(f"Tournament-team matchups: {len(tourney_X):,}")
if len(tourney_X) > 0:
    diff = (tourney_X["Pred"] - tourney_X["xgb_pred"]).abs()
    print(f"Avg absolute change from Torvik blend: {diff.mean():.4f}")
    print(f"Max absolute change: {diff.max():.4f}")
print()

# Show marquee matchups
print("1-seed vs 1-seed (men's):")
top = X[(X["T1_seed"] == 1) & (X["T2_seed"] == 1) & (X["men_women"] == 1)]
for _, row in top.iterrows():
    n1, n2 = team_name_map.get(row["T1_TeamID"], "?"), team_name_map.get(row["T2_TeamID"], "?")
    xgb = row["xgb_pred"]
    tv = row.get("torvik_pred", None)
    blend = row["Pred"]
    tv_str = f"{tv:.1%}" if pd.notna(tv) else "N/A"
    print(f"  {n1} vs {n2}: XGB={xgb:.1%}  Torvik={tv_str}  Blend={blend:.1%}")

print("\n1-seed vs 2-seed (men's):")
s12 = X[(X["T1_seed"] == 1) & (X["T2_seed"] == 2) & (X["men_women"] == 1)]
for _, row in s12.head(8).iterrows():
    n1, n2 = team_name_map.get(row["T1_TeamID"], "?"), team_name_map.get(row["T2_TeamID"], "?")
    xgb = row["xgb_pred"]
    tv = row.get("torvik_pred", None)
    blend = row["Pred"]
    tv_str = f"{tv:.1%}" if pd.notna(tv) else "N/A"
    print(f"  {n1} vs {n2}: XGB={xgb:.1%}  Torvik={tv_str}  Blend={blend:.1%}")
print()

output_path = "submission_level11_torvik.csv"
X[["ID", "Pred"]].to_csv(output_path, index=False)
print(f"Submission saved to: {output_path}")
print("Done! 🏀")
