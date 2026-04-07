"""
Level 12: BartTorvik Features as XGBoost Training Data
=======================================================
The big upgrade. Instead of using Torvik as a post-hoc ensemble (Level 11),
we now have 10+ years of historical BartTorvik T-Rank data. This means we can
add tempo-adjusted efficiency metrics DIRECTLY as XGBoost features.

These replace our crude hand-computed box score averages with the gold standard
of college basketball analytics: opponent-adjusted, tempo-adjusted ratings.

Key Torvik features:
  - AdjOE / AdjDE: Tempo-adjusted offensive/defensive efficiency
  - Barthag: Power rating (expected win% vs average team)
  - WAB: Wins Above Bubble
  - SOS: Strength of schedule (Torvik's version, better than ours)
  - Tempo (adjt): Adjusted pace of play

Data availability:
  - Men's: 2015-2019, 2021-2026 (11 seasons)
  - Women's: 2021-2026 (6 seasons)
  - No 2020 (COVID, no tournament)
"""

import numpy as np
import pandas as pd
import warnings
import os
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline
from xgboost import DMatrix, train as xgb_train
from sklearn.metrics import brier_score_loss, mean_absolute_error

warnings.filterwarnings("ignore")

DATA = "data/"
TORVIK = "data/torvik/"
MIN_SEASON = 2003

# ============================================================
# STEP 1: Load all data
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
m_spell = pd.read_csv(f"{DATA}MTeamSpellings.csv", encoding="latin1")
w_spell = pd.read_csv(f"{DATA}WTeamSpellings.csv", encoding="latin1")
team_name_map = dict(zip(
    pd.concat([m_teams[["TeamID","TeamName"]], w_teams[["TeamID","TeamName"]]])["TeamID"],
    pd.concat([m_teams[["TeamID","TeamName"]], w_teams[["TeamID","TeamName"]]])["TeamName"]
))

print(f"Regular season: {len(regular_results):,}")
print(f"Tournament:     {len(tourney_results):,}")
print()

# ============================================================
# STEP 2: Load and match ALL historical Torvik data
# ============================================================
print("=" * 60)
print("STEP 2: Loading historical BartTorvik data...")
print("=" * 60)

# Build name → TeamID mappings
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

def load_torvik_file(filepath, name_to_id, season):
    """Load one Torvik CSV, parse adjt, match teams to IDs."""
    df = pd.read_csv(filepath, index_col=False)
    df["Season"] = season

    # Handle the adjt column inconsistency
    # Older files have "Fun Rk, adjt" as one header but data has separate values,
    # causing a column shift. Detect and fix.
    if "adjt" not in df.columns:
        if "Fun Rk, adjt" in df.columns:
            df["adjt"] = df["Fun Rk, adjt"]
        else:
            df["adjt"] = np.nan

    # Match team names to IDs
    team_ids = []
    for _, row in df.iterrows():
        name = row["team"].lower().strip()
        tid = name_to_id.get(name) or name_to_id.get(manual_map.get(name, ""), None)
        team_ids.append(tid)
    df["TeamID"] = team_ids

    return df

# Load all files
all_torvik = []
for f in sorted(os.listdir(TORVIK)):
    if not f.endswith(".csv"):
        continue
    is_womens = f.startswith("womens")
    # filenames like mens_2015.csv or womens_2021.csv
    parts = f.replace(".csv", "").split("_")
    year = int(parts[1])  # mens_2015 → 2015
    name_to_id = w_name_to_id if is_womens else m_name_to_id
    filepath = os.path.join(TORVIK, f)

    df = load_torvik_file(filepath, name_to_id, year)
    df["is_womens"] = int(is_womens)
    all_torvik.append(df)
    matched = df["TeamID"].notna().sum()
    print(f"  {f}: {matched}/{len(df)} matched")

torvik_all = pd.concat(all_torvik, ignore_index=True)
print(f"\nTotal Torvik records: {len(torvik_all):,}")
print(f"Matched to TeamID: {torvik_all['TeamID'].notna().sum():,}")
print()

# Build lookup: (season, TeamID) -> torvik stats
TORVIK_COLS = ["adjoe", "adjde", "adjt", "barthag", "sos", "WAB"]
torvik_lookup = {}
for _, row in torvik_all.iterrows():
    if pd.notna(row["TeamID"]):
        entry = {}
        for col in TORVIK_COLS:
            entry[col] = row[col] if col in row and pd.notna(row[col]) else np.nan
        torvik_lookup[(row["Season"], int(row["TeamID"]))] = entry

print(f"Torvik lookup entries: {len(torvik_lookup):,}")
print()

# ============================================================
# STEP 3: Prepare data (same as Level 9)
# ============================================================
print("=" * 60)
print("STEP 3: Preparing data...")
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
# STEP 4: Seeds + box scores + late season (from Level 9)
# ============================================================
print("=" * 60)
print("STEP 4: Seeds + box scores + late season...")
print("=" * 60)

seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]]
tourney_data = tourney_data.merge(seeds_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(seeds_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

boxcols = ["T1_Score", "T1_FGA", "T1_OR", "T1_DR", "T1_Blk", "T1_PF",
           "T2_FGA", "T2_Blk", "T2_PF", "PointDiff"]
ss = regular_data.groupby(["Season", "T1_TeamID"])[
    ["T1_Score", "T1_FGA", "T1_OR", "T1_DR", "T1_Blk", "T1_PF",
     "T2_FGA", "T2_Blk", "T2_PF", "PointDiff"]
].mean().reset_index()
ss_T1 = ss.copy()
ss_T1.columns = ["T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in ss_T1.columns]
ss_T1 = ss_T1.rename({"T1_avg_Season": "Season", "T1_avg_TeamID": "T1_TeamID"}, axis=1)
ss_T2 = ss.copy()
ss_T2.columns = ["T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in ss_T2.columns]
ss_T2 = ss_T2.rename({"T2_avg_Season": "Season", "T2_avg_TeamID": "T2_TeamID"}, axis=1)

tourney_data = tourney_data.merge(ss_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(ss_T2, on=["Season", "T2_TeamID"], how="left")

# Late season
late_data = regular_data[regular_data["DayNum"] >= 119]
late_ss = late_data.groupby(["Season", "T1_TeamID"])[["PointDiff"]].mean().reset_index()
late_T1 = late_ss.rename(columns={"T1_TeamID": "T1_TeamID", "PointDiff": "T1_late_avg_PointDiff"})
late_T2 = late_ss.rename(columns={"T1_TeamID": "T2_TeamID", "PointDiff": "T2_late_avg_PointDiff"})
tourney_data = tourney_data.merge(late_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(late_T2, on=["Season", "T2_TeamID"], how="left")

print("Done.")
print()

# ============================================================
# STEP 5: Better Elo (from Level 9)
# ============================================================
print("=" * 60)
print("STEP 5: Computing Elo...")
print("=" * 60)

def compute_better_elo(regular_data):
    K, WIDTH, MARGIN_FACTOR, REVERSION, INIT = 32, 400, 0.8, 0.4, 1500
    ratings, snapshots, current_season = {}, {}, None
    for season in sorted(regular_data["Season"].unique()):
        if current_season is not None:
            for team in ratings:
                ratings[team] = INIT + (ratings[team] - INIT) * (1 - REVERSION)
        current_season = season
        ss = regular_data[(regular_data["Season"] == season) & (regular_data["win"] == 1)].sort_values("DayNum")
        for _, row in ss.iterrows():
            w, l = int(row["T1_TeamID"]), int(row["T2_TeamID"])
            if w not in ratings: ratings[w] = INIT
            if l not in ratings: ratings[l] = INIT
            exp_w = 1.0 / (1.0 + 10.0 ** ((ratings[l] - ratings[w]) / WIDTH))
            margin = row["T1_Score"] - row["T2_Score"]
            update = K * np.log(1 + abs(margin)) * MARGIN_FACTOR * (1 - exp_w)
            ratings[w] += update
            ratings[l] -= update
        for team, rating in ratings.items():
            snapshots[(season, team)] = rating
    return snapshots

elo_snapshots = compute_better_elo(regular_data)

def add_elo(df, elo_dict, prefix, default=1500):
    t1_col, t2_col = f"T1_{prefix}", f"T2_{prefix}"
    df[t1_col] = df.apply(lambda r: elo_dict.get((r["Season"], r["T1_TeamID"]), default), axis=1)
    df[t2_col] = df.apply(lambda r: elo_dict.get((r["Season"], r["T2_TeamID"]), default), axis=1)
    df[f"{prefix}_diff"] = df[t1_col] - df[t2_col]
    return df

tourney_data = add_elo(tourney_data, elo_snapshots, "elo2")
print(f"Elo entries: {len(elo_snapshots):,}")
print()

# ============================================================
# STEP 6: GLM quality (from Level 9)
# ============================================================
print("=" * 60)
print("STEP 6: GLM quality...")
print("=" * 60)

seeds_T1_st = seeds_T1.copy()
seeds_T1_st["ST1"] = seeds_T1_st["Season"].astype(str) + "/" + seeds_T1_st["T1_TeamID"].astype(str)
seeds_T2_st = seeds_T2.copy()
seeds_T2_st["ST2"] = seeds_T2_st["Season"].astype(str) + "/" + seeds_T2_st["T2_TeamID"].astype(str)
regular_data["ST1"] = regular_data["Season"].astype(int).astype(str) + "/" + regular_data["T1_TeamID"].astype(int).astype(str)
regular_data["ST2"] = regular_data["Season"].astype(int).astype(str) + "/" + regular_data["T2_TeamID"].astype(int).astype(str)
st = set(seeds_T1_st["ST1"]) | set(seeds_T2_st["ST2"])
st = st | set(regular_data[(regular_data["T1_Score"] > regular_data["T2_Score"]) & (regular_data["ST2"].isin(st))]["ST1"])
dt = regular_data[regular_data["ST1"].isin(st) | regular_data["ST2"].isin(st)].copy()
dt["T1_TeamID"] = dt["T1_TeamID"].round().astype(int).astype(str)
dt["T2_TeamID"] = dt["T2_TeamID"].round().astype(int).astype(str)
dt.loc[~dt["ST1"].isin(st), "T1_TeamID"] = "0000"
dt.loc[~dt["ST2"].isin(st), "T2_TeamID"] = "0000"

glm_quality = []
for s in sorted(seeds["Season"].unique()):
    for mw in ([0, 1] if s >= 2010 else [1]):
        subset = dt[(dt["Season"] == s) & (dt["men_women"] == mw)].copy()
        if len(subset) < 50: continue
        try:
            glm = sm.GLM.from_formula("PointDiff ~ -1 + T1_TeamID + T2_TeamID", data=subset, family=sm.families.Gaussian()).fit()
            t1_params = glm.params[glm.params.index.str.startswith("T1_")]
            q = pd.DataFrame({"TeamID_raw": t1_params.index, "quality": t1_params.values})
            q["Season"] = s
            q["TeamID"] = q["TeamID_raw"].str.extract(r'(\d{4})').astype(int)
            glm_quality.append(q[["TeamID", "quality", "Season"]])
        except:
            pass

glm_quality = pd.concat(glm_quality).reset_index(drop=True)
glm_T1 = glm_quality.rename(columns={"TeamID": "T1_TeamID", "quality": "T1_quality"})
glm_T2 = glm_quality.rename(columns={"TeamID": "T2_TeamID", "quality": "T2_quality"})
tourney_data = tourney_data.merge(glm_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(glm_T2, on=["Season", "T2_TeamID"], how="left")
print(f"GLM: {len(glm_quality):,}")
print()

# ============================================================
# STEP 7: Massey POM (from Level 9)
# ============================================================
print("=" * 60)
print("STEP 7: Massey POM...")
print("=" * 60)

max_days = massey.groupby(["Season", "SystemName"])["RankingDayNum"].max().reset_index()
max_days.columns = ["Season", "SystemName", "MaxDay"]
final_massey = massey.merge(max_days, on=["Season", "SystemName"])
final_massey = final_massey[final_massey["RankingDayNum"] == final_massey["MaxDay"]]
avg_rank = final_massey.groupby(["Season", "TeamID"])["OrdinalRank"].mean().reset_index()
avg_rank.columns = ["Season", "TeamID", "MasseyAvgRank"]
pom = final_massey[final_massey["SystemName"] == "POM"][["Season", "TeamID", "OrdinalRank"]]
pom.columns = ["Season", "TeamID", "POMRank"]

for prefix, id_col in [("T1", "T1_TeamID"), ("T2", "T2_TeamID")]:
    tourney_data = tourney_data.merge(
        pom.rename(columns={"TeamID": id_col, "POMRank": f"{prefix}_POM"}),
        on=["Season", id_col], how="left")
    tourney_data = tourney_data.merge(
        avg_rank.rename(columns={"TeamID": id_col, "MasseyAvgRank": f"{prefix}_MasseyAvg"}),
        on=["Season", id_col], how="left")
tourney_data["POM_diff"] = tourney_data["T2_POM"] - tourney_data["T1_POM"]
tourney_data["MasseyAvg_diff"] = tourney_data["T2_MasseyAvg"] - tourney_data["T1_MasseyAvg"]
print("Done.")
print()

# ============================================================
# STEP 8: NEW — Add BartTorvik features directly
# ============================================================
print("=" * 60)
print("STEP 8: Adding BartTorvik features as training data...")
print("=" * 60)

for prefix, id_col in [("T1", "T1_TeamID"), ("T2", "T2_TeamID")]:
    for tcol in TORVIK_COLS:
        col_name = f"{prefix}_tv_{tcol}"
        tourney_data[col_name] = tourney_data.apply(
            lambda r: torvik_lookup.get((r["Season"], r[id_col]), {}).get(tcol, np.nan), axis=1
        )

# Compute diffs
tourney_data["tv_adjoe_diff"] = tourney_data["T1_tv_adjoe"] - tourney_data["T2_tv_adjoe"]
tourney_data["tv_adjde_diff"] = tourney_data["T1_tv_adjde"] - tourney_data["T2_tv_adjde"]
tourney_data["tv_barthag_diff"] = tourney_data["T1_tv_barthag"] - tourney_data["T2_tv_barthag"]
tourney_data["tv_WAB_diff"] = tourney_data["T1_tv_WAB"] - tourney_data["T2_tv_WAB"]
tourney_data["tv_sos_diff"] = tourney_data["T1_tv_sos"] - tourney_data["T2_tv_sos"]

# Computed spread feature
tourney_data["tv_spread"] = tourney_data.apply(
    lambda r: ((r["T1_tv_adjoe"] - r["T2_tv_adjde"]) - (r["T2_tv_adjoe"] - r["T1_tv_adjde"])) / 100
              * ((r["T1_tv_adjt"] + r["T2_tv_adjt"]) / 2)
    if pd.notna(r["T1_tv_adjoe"]) and pd.notna(r["T2_tv_adjoe"]) else np.nan, axis=1
)

tv_avail = tourney_data["T1_tv_adjoe"].notna().sum()
print(f"Tournament games with Torvik data: {tv_avail} / {len(tourney_data)}")
print(f"  (Men 2015+ and Women 2021+ have Torvik; older seasons are NaN → XGBoost handles natively)")
print()

# ============================================================
# STEP 9: Define features and train
# ============================================================
print("=" * 60)
print("STEP 9: Training XGBoost (LOSO)...")
print("=" * 60)

features = [
    "men_women",
    "T1_seed", "T2_seed", "Seed_diff",
    # Box score averages
    "T1_avg_Score", "T1_avg_FGA", "T1_avg_OR", "T1_avg_DR", "T1_avg_Blk", "T1_avg_PF",
    "T1_avg_opponent_FGA", "T1_avg_opponent_Blk", "T1_avg_opponent_PF", "T1_avg_PointDiff",
    "T2_avg_Score", "T2_avg_FGA", "T2_avg_OR", "T2_avg_DR", "T2_avg_Blk", "T2_avg_PF",
    "T2_avg_opponent_FGA", "T2_avg_opponent_Blk", "T2_avg_opponent_PF", "T2_avg_PointDiff",
    # Late season
    "T1_late_avg_PointDiff", "T2_late_avg_PointDiff",
    # Elo
    "T1_elo2", "T2_elo2", "elo2_diff",
    # GLM
    "T1_quality", "T2_quality",
    # Massey
    "T1_POM", "T2_POM", "POM_diff", "MasseyAvg_diff",
    # NEW: BartTorvik features
    "T1_tv_adjoe", "T2_tv_adjoe", "tv_adjoe_diff",
    "T1_tv_adjde", "T2_tv_adjde", "tv_adjde_diff",
    "T1_tv_barthag", "T2_tv_barthag", "tv_barthag_diff",
    "T1_tv_WAB", "T2_tv_WAB", "tv_WAB_diff",
    "T1_tv_sos", "T2_tv_sos", "tv_sos_diff",
    "T1_tv_adjt", "T2_tv_adjt",
    "tv_spread",
]

print(f"Features: {len(features)}")

param = {
    "objective": "reg:squarederror", "booster": "gbtree",
    "eta": 0.0093, "subsample": 0.6, "colsample_bynode": 0.8,
    "num_parallel_tree": 2, "min_child_weight": 4, "max_depth": 4,
    "tree_method": "hist", "grow_policy": "lossguide", "max_bin": 38,
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

    dtrain = DMatrix(X_tr, label=y_tr, feature_names=features)
    models[oof_season] = xgb_train(params=param, dtrain=dtrain, num_boost_round=num_rounds)

    preds = models[oof_season].predict(DMatrix(X_val, feature_names=features))
    mae = mean_absolute_error(y_val, preds)
    print(f"  {oof_season}: MAE = {mae:.2f}")
    oof_preds.extend(preds.tolist())
    oof_targets.extend(y_val.tolist())
    oof_seasons.extend([oof_season] * len(y_val))

print(f"\n  Average MAE: {mean_absolute_error(oof_targets, oof_preds):.2f}")
print()

# ============================================================
# STEP 10: Spline calibration
# ============================================================
print("=" * 60)
print("STEP 10: Spline calibration...")
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
print(f"  Level 8 (1st place):          0.1655")
print(f"  Level 9 (special sauce):      0.1650")
print(f"  Level 12 (Torvik features):   {overall_brier:.4f}")
improvement = (0.1650 - overall_brier) / 0.1650 * 100
print(f"  vs Level 9:                   {improvement:+.1f}%")
print()

# ============================================================
# STEP 11: Feature importance
# ============================================================
print("=" * 60)
print("STEP 11: Feature importance...")
print("=" * 60)

last_model = models[seasons[-1]]
importance = last_model.get_score(importance_type="gain")
sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
for fkey, gain in sorted_imp[:25]:
    bar = "█" * int(gain / sorted_imp[0][1] * 40)
    print(f"  {fkey:25s}: {gain:10.1f}  {bar}")
print()

# ============================================================
# STEP 12: Generate 2026 predictions
# ============================================================
print("=" * 60)
print("STEP 12: Generating 2026 predictions...")
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
X = add_elo(X, elo_snapshots, "elo2")
X["Seed_diff"] = X["T2_seed"] - X["T1_seed"]

for prefix, id_col in [("T1", "T1_TeamID"), ("T2", "T2_TeamID")]:
    X = X.merge(pom.rename(columns={"TeamID": id_col, "POMRank": f"{prefix}_POM"}), on=["Season", id_col], how="left")
    X = X.merge(avg_rank.rename(columns={"TeamID": id_col, "MasseyAvgRank": f"{prefix}_MasseyAvg"}), on=["Season", id_col], how="left")
X["POM_diff"] = X["T2_POM"] - X["T1_POM"]
X["MasseyAvg_diff"] = X["T2_MasseyAvg"] - X["T1_MasseyAvg"]

# Torvik features
for prefix, id_col in [("T1", "T1_TeamID"), ("T2", "T2_TeamID")]:
    for tcol in TORVIK_COLS:
        col_name = f"{prefix}_tv_{tcol}"
        X[col_name] = X.apply(lambda r: torvik_lookup.get((r["Season"], r[id_col]), {}).get(tcol, np.nan), axis=1)
X["tv_adjoe_diff"] = X["T1_tv_adjoe"] - X["T2_tv_adjoe"]
X["tv_adjde_diff"] = X["T1_tv_adjde"] - X["T2_tv_adjde"]
X["tv_barthag_diff"] = X["T1_tv_barthag"] - X["T2_tv_barthag"]
X["tv_WAB_diff"] = X["T1_tv_WAB"] - X["T2_tv_WAB"]
X["tv_sos_diff"] = X["T1_tv_sos"] - X["T2_tv_sos"]
X["tv_spread"] = X.apply(
    lambda r: ((r["T1_tv_adjoe"] - r["T2_tv_adjde"]) - (r["T2_tv_adjoe"] - r["T1_tv_adjde"])) / 100
              * ((r["T1_tv_adjt"] + r["T2_tv_adjt"]) / 2)
    if pd.notna(r["T1_tv_adjoe"]) and pd.notna(r["T2_tv_adjoe"]) else np.nan, axis=1
)

# Ensemble predict
dtest = DMatrix(X[features].values, feature_names=features)
all_preds = []
for oof_season in seasons:
    margin_preds = models[oof_season].predict(dtest)
    probs = np.clip(spline_model(np.clip(margin_preds, -CLIP_DIFF, CLIP_DIFF)), 0.01, 0.99)
    all_preds.append(probs)
X["Pred"] = np.mean(all_preds, axis=0)

print(f"Predictions: {len(X):,}")
print(f"Range: {X['Pred'].min():.4f} to {X['Pred'].max():.4f}")
print(f"Mean:  {X['Pred'].mean():.4f}")
print()

print("1-seed vs 1-seed (men's):")
top = X[(X["T1_seed"] == 1) & (X["T2_seed"] == 1) & (X["men_women"] == 1)]
for _, row in top.iterrows():
    n1, n2 = team_name_map.get(row["T1_TeamID"], "?"), team_name_map.get(row["T2_TeamID"], "?")
    print(f"  {n1} vs {n2}: P({n1} wins) = {row['Pred']:.1%}")

print("\n1-seed vs 2-seed (men's):")
s12 = X[(X["T1_seed"] == 1) & (X["T2_seed"] == 2) & (X["men_women"] == 1)]
for _, row in s12.head(8).iterrows():
    n1, n2 = team_name_map.get(row["T1_TeamID"], "?"), team_name_map.get(row["T2_TeamID"], "?")
    print(f"  {n1} vs {n2}: P({n1} wins) = {row['Pred']:.1%}")
print()

output_path = "submission_level12_torvik_features.csv"
X[["ID", "Pred"]].to_csv(output_path, index=False)
print(f"Submission saved to: {output_path}")
print("Done! 🏀")
