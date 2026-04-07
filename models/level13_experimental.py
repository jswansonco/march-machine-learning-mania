"""
Level 13: Experimental — Testing 2026 Tournament Findings
==========================================================
Based on Level 12, with these improvements:

1. POSSESSION VARIANCE MODIFIER: Low-pace games have higher variance,
   so compress predictions toward 0.5 for slow expected game pace.
   (Law of large numbers: fewer possessions = less certainty)

2. "UNDERSEEDED" FEATURE: Gap between actual seed and power-rating-implied
   seed. Teams ranked much higher than their seed are dangerous underdogs.

3. SEPARATE WOMEN'S CALIBRATION: Women's tournament is more chalk;
   using one calibration curve for both genders may miscalibrate.

4. SEED GAP COMPRESSION IN LATER ROUNDS (post-hoc): Not applicable to
   pre-tournament predictions, but noted for future live-updating model.

We compare L13 vs L12 on:
  a) LOSO Brier (historical)
  b) 2026 actual R64+R32 results (74 games)
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
# STEP 1-7: Same as Level 12 (load data, features, Elo, GLM, Massey)
# ============================================================
print("=" * 60)
print("LEVEL 13: EXPERIMENTAL MODEL")
print("=" * 60)
print("\nLoading data (same as L12)...")

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

# --- Torvik ---
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
    df = pd.read_csv(filepath, index_col=False)
    df["Season"] = season
    if "adjt" not in df.columns:
        if "Fun Rk, adjt" in df.columns:
            df["adjt"] = df["Fun Rk, adjt"]
        else:
            df["adjt"] = np.nan
    team_ids = []
    for _, row in df.iterrows():
        name = row["team"].lower().strip()
        tid = name_to_id.get(name) or name_to_id.get(manual_map.get(name, ""), None)
        team_ids.append(tid)
    df["TeamID"] = team_ids
    return df

all_torvik = []
for f in sorted(os.listdir(TORVIK)):
    if not f.endswith(".csv"): continue
    is_womens = f.startswith("womens")
    parts = f.replace(".csv", "").split("_")
    year = int(parts[1])
    name_to_id = w_name_to_id if is_womens else m_name_to_id
    df = load_torvik_file(os.path.join(TORVIK, f), name_to_id, year)
    df["is_womens"] = int(is_womens)
    all_torvik.append(df)

torvik_all = pd.concat(all_torvik, ignore_index=True)

TORVIK_COLS = ["adjoe", "adjde", "adjt", "barthag", "sos", "WAB"]
torvik_lookup = {}
for _, row in torvik_all.iterrows():
    if pd.notna(row["TeamID"]):
        entry = {}
        for col in TORVIK_COLS:
            entry[col] = row[col] if col in row and pd.notna(row[col]) else np.nan
        torvik_lookup[(row["Season"], int(row["TeamID"]))] = entry

# --- Also build a rank lookup for "underseeded" feature ---
# Torvik rank = power ranking. Compare to actual seed.
torvik_rank_lookup = {}
for _, row in torvik_all.iterrows():
    if pd.notna(row["TeamID"]) and "rank" in row and pd.notna(row.iloc[0]):
        # First column is rank
        torvik_rank_lookup[(row["Season"], int(row["TeamID"]))] = int(row.iloc[0])

print(f"Torvik lookup: {len(torvik_lookup):,} entries")
print(f"Torvik rank lookup: {len(torvik_rank_lookup):,} entries")

# --- Prepare data ---
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

# --- Seeds + box scores + late season ---
seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]]
tourney_data = tourney_data.merge(seeds_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(seeds_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

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

late_data = regular_data[regular_data["DayNum"] >= 119]
late_ss = late_data.groupby(["Season", "T1_TeamID"])[["PointDiff"]].mean().reset_index()
late_T1 = late_ss.rename(columns={"T1_TeamID": "T1_TeamID", "PointDiff": "T1_late_avg_PointDiff"})
late_T2 = late_ss.rename(columns={"T1_TeamID": "T2_TeamID", "PointDiff": "T2_late_avg_PointDiff"})
tourney_data = tourney_data.merge(late_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(late_T2, on=["Season", "T2_TeamID"], how="left")

# --- Elo ---
def compute_better_elo(regular_data):
    K, WIDTH, MARGIN_FACTOR, REVERSION, INIT = 32, 400, 0.8, 0.4, 1500
    ratings, snapshots, current_season = {}, {}, None
    for season in sorted(regular_data["Season"].unique()):
        if current_season is not None:
            for team in ratings:
                ratings[team] = INIT + (ratings[team] - INIT) * (1 - REVERSION)
        current_season = season
        ss_elo = regular_data[(regular_data["Season"] == season) & (regular_data["win"] == 1)].sort_values("DayNum")
        for _, row in ss_elo.iterrows():
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

# --- GLM quality ---
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

# --- Massey ---
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

# --- Torvik features ---
for prefix, id_col in [("T1", "T1_TeamID"), ("T2", "T2_TeamID")]:
    for tcol in TORVIK_COLS:
        col_name = f"{prefix}_tv_{tcol}"
        tourney_data[col_name] = tourney_data.apply(
            lambda r: torvik_lookup.get((r["Season"], r[id_col]), {}).get(tcol, np.nan), axis=1)

tourney_data["tv_adjoe_diff"] = tourney_data["T1_tv_adjoe"] - tourney_data["T2_tv_adjoe"]
tourney_data["tv_adjde_diff"] = tourney_data["T1_tv_adjde"] - tourney_data["T2_tv_adjde"]
tourney_data["tv_barthag_diff"] = tourney_data["T1_tv_barthag"] - tourney_data["T2_tv_barthag"]
tourney_data["tv_WAB_diff"] = tourney_data["T1_tv_WAB"] - tourney_data["T2_tv_WAB"]
tourney_data["tv_sos_diff"] = tourney_data["T1_tv_sos"] - tourney_data["T2_tv_sos"]
tourney_data["tv_spread"] = tourney_data.apply(
    lambda r: ((r["T1_tv_adjoe"] - r["T2_tv_adjde"]) - (r["T2_tv_adjoe"] - r["T1_tv_adjde"])) / 100
              * ((r["T1_tv_adjt"] + r["T2_tv_adjt"]) / 2)
    if pd.notna(r["T1_tv_adjoe"]) and pd.notna(r["T2_tv_adjoe"]) else np.nan, axis=1)

print("Base features loaded (same as L12).")
print()

# ============================================================
# STEP 8: NEW FEATURES FOR LEVEL 13
# ============================================================
print("=" * 60)
print("STEP 8: NEW Level 13 Features")
print("=" * 60)

# --- Feature 1: Expected Game Pace (average of both teams' tempo) ---
# Lower pace = higher variance = predictions should compress toward 0.5
tourney_data["expected_pace"] = (tourney_data["T1_tv_adjt"] + tourney_data["T2_tv_adjt"]) / 2
tourney_data["pace_diff"] = tourney_data["T1_tv_adjt"] - tourney_data["T2_tv_adjt"]
tourney_data["abs_pace_diff"] = tourney_data["pace_diff"].abs()

# --- Feature 2: "Underseeded" — gap between Torvik rank and actual seed ---
# A team ranked #10 by Torvik but seeded #11 is "underseeded" (dangerous)
# Positive = underseeded (ranked better than seed implies)
# We map seed to approximate expected rank: seed 1 → rank ~4, seed 16 → rank ~200+
SEED_TO_EXPECTED_RANK = {
    1: 4, 2: 10, 3: 16, 4: 22, 5: 30, 6: 38, 7: 46, 8: 55,
    9: 65, 10: 75, 11: 90, 12: 110, 13: 140, 14: 170, 15: 220, 16: 300
}

def underseed_score(season, team_id, seed):
    """How much better is this team than their seed implies?
    Positive = ranked better than expected for their seed (dangerous underdog)."""
    tv_rank = torvik_rank_lookup.get((season, team_id))
    if tv_rank is None or pd.isna(seed):
        return np.nan
    expected_rank = SEED_TO_EXPECTED_RANK.get(int(seed), 150)
    return expected_rank - tv_rank  # positive = better than expected

tourney_data["T1_underseed"] = tourney_data.apply(
    lambda r: underseed_score(r["Season"], r["T1_TeamID"], r.get("T1_seed")), axis=1)
tourney_data["T2_underseed"] = tourney_data.apply(
    lambda r: underseed_score(r["Season"], r["T2_TeamID"], r.get("T2_seed")), axis=1)
tourney_data["underseed_diff"] = tourney_data["T1_underseed"] - tourney_data["T2_underseed"]

# --- Feature 3: Close-game record (proxy for "clutch") ---
# Win rate in games decided by 5 or fewer points
close_games = regular_data[regular_data["PointDiff"].abs() <= 5].copy()
close_record = close_games.groupby(["Season", "T1_TeamID"])["win"].agg(["mean", "count"]).reset_index()
close_record.columns = ["Season", "TeamID", "close_win_pct", "close_game_count"]
# Only use if team had at least 3 close games (otherwise noisy)
close_record.loc[close_record["close_game_count"] < 3, "close_win_pct"] = np.nan

close_T1 = close_record[["Season", "TeamID", "close_win_pct"]].rename(
    columns={"TeamID": "T1_TeamID", "close_win_pct": "T1_close_win_pct"})
close_T2 = close_record[["Season", "TeamID", "close_win_pct"]].rename(
    columns={"TeamID": "T2_TeamID", "close_win_pct": "T2_close_win_pct"})
tourney_data = tourney_data.merge(close_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = tourney_data.merge(close_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["close_win_pct_diff"] = tourney_data["T1_close_win_pct"] - tourney_data["T2_close_win_pct"]

n_underseed = tourney_data["T1_underseed"].notna().sum()
n_pace = tourney_data["expected_pace"].notna().sum()
n_close = tourney_data["T1_close_win_pct"].notna().sum()
print(f"  Underseed feature available: {n_underseed}/{len(tourney_data)}")
print(f"  Expected pace available:     {n_pace}/{len(tourney_data)}")
print(f"  Close-game record available: {n_close}/{len(tourney_data)}")
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
    # BartTorvik (same as L12)
    "T1_tv_adjoe", "T2_tv_adjoe", "tv_adjoe_diff",
    "T1_tv_adjde", "T2_tv_adjde", "tv_adjde_diff",
    "T1_tv_barthag", "T2_tv_barthag", "tv_barthag_diff",
    "T1_tv_WAB", "T2_tv_WAB", "tv_WAB_diff",
    "T1_tv_sos", "T2_tv_sos", "tv_sos_diff",
    "T1_tv_adjt", "T2_tv_adjt",
    "tv_spread",
    # NEW Level 13 features
    "expected_pace",       # avg tempo → low pace = high variance game
    "pace_diff",           # tempo mismatch
    "abs_pace_diff",       # absolute tempo mismatch
    "T1_underseed",        # team1 ranked better than seed implies
    "T2_underseed",        # team2 ranked better than seed implies
    "underseed_diff",      # relative underseeding
    "T1_close_win_pct",    # team1 clutch record
    "T2_close_win_pct",    # team2 clutch record
    "close_win_pct_diff",  # clutch differential
]

print(f"Features: {len(features)} (L12 had 53, added {len(features) - 53} new)")

param = {
    "objective": "reg:squarederror", "booster": "gbtree",
    "eta": 0.0093, "subsample": 0.6, "colsample_bynode": 0.8,
    "num_parallel_tree": 2, "min_child_weight": 4, "max_depth": 4,
    "tree_method": "hist", "grow_policy": "lossguide", "max_bin": 38,
}
num_rounds = 704

models = {}
oof_preds, oof_targets, oof_seasons, oof_mw = [], [], [], []

seasons = sorted(tourney_data["Season"].unique())
for oof_season in seasons:
    X_tr = tourney_data.loc[tourney_data["Season"] != oof_season, features].values
    y_tr = tourney_data.loc[tourney_data["Season"] != oof_season, "PointDiff"].values
    X_val = tourney_data.loc[tourney_data["Season"] == oof_season, features].values
    y_val = tourney_data.loc[tourney_data["Season"] == oof_season, "PointDiff"].values
    mw_val = tourney_data.loc[tourney_data["Season"] == oof_season, "men_women"].values

    dtrain = DMatrix(X_tr, label=y_tr, feature_names=features)
    models[oof_season] = xgb_train(params=param, dtrain=dtrain, num_boost_round=num_rounds)

    preds = models[oof_season].predict(DMatrix(X_val, feature_names=features))
    mae = mean_absolute_error(y_val, preds)
    oof_preds.extend(preds.tolist())
    oof_targets.extend(y_val.tolist())
    oof_seasons.extend([oof_season] * len(y_val))
    oof_mw.extend(mw_val.tolist())

print(f"\n  Average MAE: {mean_absolute_error(oof_targets, oof_preds):.2f}")
print()

# ============================================================
# STEP 10: SEPARATE calibration for men's and women's
# ============================================================
print("=" * 60)
print("STEP 10: Separate M/W spline calibration...")
print("=" * 60)

CLIP_DIFF = 25
oof_preds = np.array(oof_preds)
oof_targets = np.array(oof_targets)
oof_seasons = np.array(oof_seasons)
oof_mw = np.array(oof_mw)
oof_labels = np.array([int(t > 0) for t in oof_targets])

# Combined spline (same as L12 for comparison)
dat = sorted(zip(oof_preds, oof_labels), key=lambda x: x[0])
pred_sorted, label_sorted = zip(*dat)
spline_combined = UnivariateSpline(np.clip(pred_sorted, -CLIP_DIFF, CLIP_DIFF), label_sorted, k=5)

# Men's spline
men_mask = oof_mw == 1
dat_m = sorted(zip(oof_preds[men_mask], oof_labels[men_mask]), key=lambda x: x[0])
pred_m, lab_m = zip(*dat_m)
spline_men = UnivariateSpline(np.clip(pred_m, -CLIP_DIFF, CLIP_DIFF), lab_m, k=5)

# Women's spline
women_mask = oof_mw == 0
dat_w = sorted(zip(oof_preds[women_mask], oof_labels[women_mask]), key=lambda x: x[0])
pred_w, lab_w = zip(*dat_w)
spline_women = UnivariateSpline(np.clip(pred_w, -CLIP_DIFF, CLIP_DIFF), lab_w, k=5)

# Apply separate calibration
spline_probs_separate = np.zeros(len(oof_preds))
for i in range(len(oof_preds)):
    clipped = np.clip(oof_preds[i], -CLIP_DIFF, CLIP_DIFF)
    if oof_mw[i] == 1:
        spline_probs_separate[i] = np.clip(spline_men(clipped), 0.01, 0.99)
    else:
        spline_probs_separate[i] = np.clip(spline_women(clipped), 0.01, 0.99)

# Also compute combined calibration for comparison
spline_probs_combined = np.clip(spline_combined(np.clip(oof_preds, -CLIP_DIFF, CLIP_DIFF)), 0.01, 0.99)

# ============================================================
# STEP 10b: Pace-adjusted calibration (post-hoc variance compression)
# ============================================================
print("\nApplying pace-adjusted variance compression...")

# For games with tempo data, compress predictions toward 0.5 based on pace
# The idea: a 70% favorite in a 75-possession game might only be 63% in a 58-possession game
# We use a simple multiplicative compression: prob_adjusted = 0.5 + (prob - 0.5) * pace_factor
# where pace_factor = min(1.0, expected_pace / REFERENCE_PACE)
REFERENCE_PACE = 70.0  # "normal" game pace — at or above this, no compression

# Get expected pace for each OOF sample
oof_pace = []
for i, (season, idx) in enumerate(zip(oof_seasons,
    tourney_data.index if len(tourney_data) == len(oof_seasons) else range(len(oof_seasons)))):
    # We need to get the pace from the original data
    pass

# Actually, let's get pace directly from tourney_data aligned with OOF
tourney_pace = tourney_data["expected_pace"].values
# The OOF predictions are generated in season order, matching tourney_data order
# Let's rebuild the mapping properly
oof_pace = []
for oof_season in seasons:
    mask = tourney_data["Season"] == oof_season
    pace_vals = tourney_data.loc[mask, "expected_pace"].values
    oof_pace.extend(pace_vals.tolist())
oof_pace = np.array(oof_pace)

spline_probs_pace = spline_probs_separate.copy()
has_pace = ~np.isnan(oof_pace)
pace_factor = np.ones(len(oof_pace))
pace_factor[has_pace] = np.clip(oof_pace[has_pace] / REFERENCE_PACE, 0.85, 1.0)
spline_probs_pace = 0.5 + (spline_probs_pace - 0.5) * pace_factor

# ============================================================
# STEP 11: Compare all calibration approaches
# ============================================================
print()
print("=" * 60)
print("STEP 11: LOSO Brier Score Comparison")
print("=" * 60)

brier_l12_style = brier_score_loss(oof_labels, spline_probs_combined)
brier_separate = brier_score_loss(oof_labels, spline_probs_separate)
brier_pace = brier_score_loss(oof_labels, spline_probs_pace)

print(f"\n  L12 style (combined spline):         {brier_l12_style:.5f}")
print(f"  L13a (separate M/W splines):          {brier_separate:.5f}")
print(f"  L13b (separate + pace compression):   {brier_pace:.5f}")

# Men's only
brier_men_l12 = brier_score_loss(oof_labels[men_mask], spline_probs_combined[men_mask])
brier_men_sep = brier_score_loss(oof_labels[men_mask], spline_probs_separate[men_mask])
brier_men_pace = brier_score_loss(oof_labels[men_mask], spline_probs_pace[men_mask])
print(f"\n  Men's only:")
print(f"    L12 style:        {brier_men_l12:.5f}")
print(f"    L13a (sep cal):   {brier_men_sep:.5f}")
print(f"    L13b (+ pace):    {brier_men_pace:.5f}")

brier_w_l12 = brier_score_loss(oof_labels[women_mask], spline_probs_combined[women_mask])
brier_w_sep = brier_score_loss(oof_labels[women_mask], spline_probs_separate[women_mask])
brier_w_pace = brier_score_loss(oof_labels[women_mask], spline_probs_pace[women_mask])
print(f"\n  Women's only:")
print(f"    L12 style:        {brier_w_l12:.5f}")
print(f"    L13a (sep cal):   {brier_w_sep:.5f}")
print(f"    L13b (+ pace):    {brier_w_pace:.5f}")

# Per-season breakdown for best approach
print(f"\n  Per-season (best L13 approach):")
eval_years = [2022, 2023, 2024, 2025]
for season in seasons:
    mask = oof_seasons == season
    if mask.sum() == 0: continue
    b = brier_score_loss(oof_labels[mask], spline_probs_pace[mask])
    marker = " ←" if season in eval_years else ""
    print(f"    {season}: {b:.5f} ({mask.sum()//2} games){marker}")

# ============================================================
# STEP 12: Feature importance (new features)
# ============================================================
print()
print("=" * 60)
print("STEP 12: Feature importance (focus on new features)...")
print("=" * 60)

last_model = models[seasons[-1]]
importance = last_model.get_score(importance_type="gain")
sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

# Show all features
for fkey, gain in sorted_imp[:30]:
    new_marker = " ← NEW" if fkey in ["expected_pace", "pace_diff", "abs_pace_diff",
        "T1_underseed", "T2_underseed", "underseed_diff",
        "T1_close_win_pct", "T2_close_win_pct", "close_win_pct_diff"] else ""
    bar = "█" * int(gain / sorted_imp[0][1] * 30)
    print(f"  {fkey:25s}: {gain:10.1f}  {bar}{new_marker}")

# ============================================================
# STEP 13: Generate 2026 predictions and compare to L12
# ============================================================
print()
print("=" * 60)
print("STEP 13: Generating 2026 predictions...")
print("=" * 60)

submission = pd.read_csv(f"{DATA}SampleSubmissionStage2.csv")
X = submission.copy()
X["Season"] = X["ID"].apply(lambda t: int(t.split("_")[0]))
X["T1_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[1]))
X["T2_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[2]))
X["men_women"] = (X["T1_TeamID"].astype(str).str.startswith("1")).astype(int)

# Merge all base features
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
    if pd.notna(r["T1_tv_adjoe"]) and pd.notna(r["T2_tv_adjoe"]) else np.nan, axis=1)

# NEW features
X["expected_pace"] = (X["T1_tv_adjt"] + X["T2_tv_adjt"]) / 2
X["pace_diff"] = X["T1_tv_adjt"] - X["T2_tv_adjt"]
X["abs_pace_diff"] = X["pace_diff"].abs()

X["T1_underseed"] = X.apply(
    lambda r: underseed_score(r["Season"], r["T1_TeamID"], r.get("T1_seed")), axis=1)
X["T2_underseed"] = X.apply(
    lambda r: underseed_score(r["Season"], r["T2_TeamID"], r.get("T2_seed")), axis=1)
X["underseed_diff"] = X["T1_underseed"] - X["T2_underseed"]

X = X.merge(close_T1, on=["Season", "T1_TeamID"], how="left")
X = X.merge(close_T2, on=["Season", "T2_TeamID"], how="left")
X["close_win_pct_diff"] = X["T1_close_win_pct"] - X["T2_close_win_pct"]

# Ensemble predict with separate M/W calibration + pace adjustment
dtest = DMatrix(X[features].values, feature_names=features)
all_preds_raw = []
for oof_season in seasons:
    margin_preds = models[oof_season].predict(dtest)
    all_preds_raw.append(margin_preds)

avg_margins = np.mean(all_preds_raw, axis=0)

# Apply gender-specific calibration
probs = np.zeros(len(avg_margins))
for i in range(len(avg_margins)):
    clipped = np.clip(avg_margins[i], -CLIP_DIFF, CLIP_DIFF)
    if X.iloc[i]["men_women"] == 1:
        probs[i] = np.clip(spline_men(clipped), 0.01, 0.99)
    else:
        probs[i] = np.clip(spline_women(clipped), 0.01, 0.99)

# Apply pace compression
pace_vals = X["expected_pace"].values
has_pace_x = ~np.isnan(pace_vals)
pace_factor_x = np.ones(len(pace_vals))
pace_factor_x[has_pace_x] = np.clip(pace_vals[has_pace_x] / REFERENCE_PACE, 0.85, 1.0)
probs_pace = 0.5 + (probs - 0.5) * pace_factor_x

X["Pred_l13"] = probs_pace
X["Pred_l13_nopace"] = probs  # without pace adjustment for comparison

# Also generate L12-style predictions (combined spline, no new features)
probs_l12_style = np.zeros(len(avg_margins))
for i in range(len(avg_margins)):
    clipped = np.clip(avg_margins[i], -CLIP_DIFF, CLIP_DIFF)
    probs_l12_style[i] = np.clip(spline_combined(clipped), 0.01, 0.99)
X["Pred_l12_style"] = probs_l12_style

# Load original L12 predictions for comparison
l12_preds = pd.read_csv("submission_level12_blend80.csv")
l12_preds.columns = ["ID", "Pred_l12_original"]
X = X.merge(l12_preds, on="ID", how="left")

# Save L13 submission
X[["ID", "Pred_l13"]].rename(columns={"Pred_l13": "Pred"}).to_csv(
    "submission_level13_experimental.csv", index=False)

print(f"Predictions generated: {len(X):,}")
print(f"Saved: submission_level13_experimental.csv")
print()

# ============================================================
# STEP 14: Score against actual 2026 results
# ============================================================
print("=" * 60)
print("STEP 14: Scoring against actual 2026 R64+R32 results...")
print("=" * 60)

# Load team mappings
m_teams_dict = dict(zip(m_teams["TeamName"], m_teams["TeamID"]))
w_teams_dict = dict(zip(w_teams["TeamName"], w_teams["TeamID"]))

aliases = {
    'Prairie View A&M': 'Prairie View', 'McNeese': 'McNeese St',
    "Saint Mary's": "St Mary's CA", 'Siena': 'Siena', 'TCU': 'TCU',
    'Northern Iowa': 'Northern Iowa', "St. John's": "St John's",
    'Cal Baptist': 'Cal Baptist', 'South Florida': 'South Florida',
    'Michigan State': 'Michigan St', 'North Dakota State': 'N Dakota St',
    'UCF': 'UCF', 'UConn': 'Connecticut', 'Furman': 'Furman',
    'Howard': 'Howard', 'Saint Louis': 'St Louis', 'Texas Tech': 'Texas Tech',
    'Akron': 'Akron', 'Hofstra': 'Hofstra', 'Miami (OH)': 'Miami OH',
    'Wright State': 'Wright St', 'Santa Clara': 'Santa Clara',
    'Iowa State': 'Iowa St', 'Tennessee State': 'Tennessee St',
    'LIU': 'LIU Brooklyn', 'Utah State': 'Utah St', 'High Point': 'High Point',
    'BYU': 'BYU', 'Kennesaw State': 'Kennesaw', 'Miami (FL)': 'Miami FL',
    'Queens': 'Queens NC', 'Ohio State': 'Ohio St', 'Texas A&M': 'Texas A&M',
    'North Carolina': 'North Carolina',
    # Women's
    'Southern': 'Southern Univ', 'South Carolina': 'South Carolina',
    'UTSA': 'UT San Antonio', 'Missouri State': 'Missouri St',
    'FDU': 'F Dickinson', 'Georgia': 'Georgia', 'Oregon': 'Oregon',
    'Syracuse': 'Syracuse', 'USC': 'USC', 'NC State': 'NC State',
    'Colorado': 'Colorado', 'Nebraska': 'Nebraska', 'Baylor': 'Baylor',
    'Green Bay': 'WI Green Bay', 'Colorado State': 'Colorado St',
    'Oklahoma State': 'Oklahoma St', 'Alabama': 'Alabama',
    'Washington': 'Washington', 'Ole Miss': 'Mississippi',
    'Maryland': 'Maryland', 'Michigan State': 'Michigan St',
}

def get_id(name, is_women=False):
    lookup = w_teams_dict if is_women else m_teams_dict
    if name in lookup: return lookup[name]
    alias = aliases.get(name, name)
    if alias in lookup: return lookup[alias]
    for k, v in lookup.items():
        if k.lower() == name.lower(): return v
    return None

def score_model(games, preds_dict, label, is_women=False):
    total_brier = 0
    correct = 0
    n = 0
    details = []
    for winner, loser in games:
        w_id = get_id(winner, is_women)
        l_id = get_id(loser, is_women)
        if w_id is None or l_id is None: continue
        low, high = min(w_id, l_id), max(w_id, l_id)
        key = f"2026_{low}_{high}"
        if key not in preds_dict: continue
        p_low = preds_dict[key]
        p_winner = p_low if w_id == low else 1 - p_low
        brier = (1 - p_winner) ** 2
        total_brier += brier
        picked = p_winner > 0.5
        if picked: correct += 1
        n += 1
        details.append((winner, loser, p_winner, brier, picked))
    return total_brier, n, correct, details

# All actual results
men_r64 = [
    ('Florida', 'Prairie View A&M'), ('Iowa', 'Clemson'), ('Vanderbilt', 'McNeese'),
    ('Nebraska', 'Troy'), ('VCU', 'North Carolina'), ('Illinois', 'Penn'),
    ('Texas A&M', "Saint Mary's"), ('Houston', 'Idaho'), ('Duke', 'Siena'),
    ('TCU', 'Ohio State'), ("St. John's", 'Northern Iowa'), ('Kansas', 'Cal Baptist'),
    ('Louisville', 'South Florida'), ('Michigan State', 'North Dakota State'),
    ('UCLA', 'UCF'), ('UConn', 'Furman'), ('Michigan', 'Howard'),
    ('Saint Louis', 'Georgia'), ('Texas Tech', 'Akron'), ('Alabama', 'Hofstra'),
    ('Tennessee', 'Miami (OH)'), ('Virginia', 'Wright State'),
    ('Kentucky', 'Santa Clara'), ('Iowa State', 'Tennessee State'),
    ('Arizona', 'LIU'), ('Utah State', 'Villanova'), ('High Point', 'Wisconsin'),
    ('Arkansas', 'Hawaii'), ('Texas', 'BYU'), ('Gonzaga', 'Kennesaw State'),
    ('Miami (FL)', 'Missouri'), ('Purdue', 'Queens'),
]
men_r32 = [
    ('Iowa', 'Florida'), ('Texas', 'Gonzaga'), ("St. John's", 'Kansas'),
    ('Tennessee', 'Virginia'), ('Alabama', 'Texas Tech'), ('UConn', 'UCLA'),
    ('Purdue', 'Miami (FL)'), ('Iowa State', 'Kentucky'), ('Arizona', 'Utah State'),
    ('Michigan', 'Saint Louis'), ('Duke', 'TCU'), ('Nebraska', 'Vanderbilt'),
    ('Illinois', 'VCU'), ('Michigan State', 'Louisville'),
    ('Arkansas', 'High Point'), ('Houston', 'Texas A&M'),
]
women_r64 = [
    ('South Carolina', 'Southern'), ('UConn', 'UTSA'), ('Texas', 'Missouri State'),
    ('UCLA', 'Cal Baptist'), ('LSU', 'Jacksonville'), ('Iowa', 'FDU'),
    ('Michigan', 'Holy Cross'), ('Vanderbilt', 'High Point'),
    ('Virginia', 'Georgia'), ('Oregon', 'Purdue'), ('Syracuse', 'Iowa State'),
    ('USC', 'Clemson'), ('NC State', 'Tennessee'), ('Illinois', 'Colorado'),
    ('Baylor', 'Nebraska'), ('Minnesota', 'Green Bay'),
    ('Michigan State', 'Colorado State'),
]
women_r32 = [
    ('Louisville', 'Alabama'), ('Texas', 'Oregon'), ('LSU', 'Texas Tech'),
    ('TCU', 'Washington'), ('North Carolina', 'Maryland'), ('Minnesota', 'Ole Miss'),
    ('Oklahoma', 'Michigan State'), ('Michigan', 'NC State'), ('Duke', 'Baylor'),
]

# Build prediction dicts for each model version
def build_pred_dict(col):
    return dict(zip(X["ID"], X[col]))

pred_l12_orig = dict(zip(l12_preds["ID"], l12_preds["Pred_l12_original"]))
pred_l13 = build_pred_dict("Pred_l13")
pred_l13_nopace = build_pred_dict("Pred_l13_nopace")
pred_l12_style = build_pred_dict("Pred_l12_style")

all_games = [
    (men_r64, "Men's R64", False),
    (men_r32, "Men's R32", False),
    (women_r64, "Women's R64", True),
    (women_r32, "Women's R32", True),
]

model_versions = [
    ("L12 Original (submitted)", pred_l12_orig),
    ("L13 new feats, combined cal", pred_l12_style),
    ("L13 new feats, sep M/W cal", pred_l13_nopace),
    ("L13 full (sep cal + pace)", pred_l13),
]

print(f"\n{'Model':<35} {'Record':<12} {'Accuracy':<12} {'Avg Brier'}")
print("-" * 75)

for model_name, pred_dict in model_versions:
    total_b, total_n, total_c = 0, 0, 0
    for games, label, is_w in all_games:
        b, n, c, _ = score_model(games, pred_dict, label, is_w)
        total_b += b
        total_n += n
        total_c += c
    avg_brier = total_b / total_n if total_n > 0 else 0
    print(f"  {model_name:<33} {total_c}/{total_n:<10} {100*total_c/total_n:>5.1f}%      {avg_brier:.5f}")

# Detailed round-by-round for best model
print(f"\n\nDetailed breakdown — L13 full vs L12 original:")
print(f"{'Round':<15} {'L12 Record':<12} {'L12 Brier':<12} {'L13 Record':<12} {'L13 Brier'}")
print("-" * 65)

for games, label, is_w in all_games:
    b12, n12, c12, _ = score_model(games, pred_l12_orig, label, is_w)
    b13, n13, c13, _ = score_model(games, pred_l13, label, is_w)
    avg12 = b12/n12 if n12 > 0 else 0
    avg13 = b13/n13 if n13 > 0 else 0
    better = "✓" if avg13 < avg12 else "✗" if avg13 > avg12 else "="
    print(f"  {label:<13} {c12}/{n12:<10} {avg12:.5f}     {c13}/{n13:<10} {avg13:.5f}  {better}")

# Show biggest prediction changes
print(f"\n\nBiggest L12→L13 prediction shifts (actual games):")
print(f"{'Game':<35} {'L12':<8} {'L13':<8} {'Shift':<8} {'Result'}")
print("-" * 70)

all_game_list = []
for games, label, is_w in all_games:
    for winner, loser in games:
        w_id = get_id(winner, is_w)
        l_id = get_id(loser, is_w)
        if w_id is None or l_id is None: continue
        low, high = min(w_id, l_id), max(w_id, l_id)
        key = f"2026_{low}_{high}"
        if key not in pred_l12_orig or key not in pred_l13: continue
        p12 = pred_l12_orig[key] if w_id == low else 1 - pred_l12_orig[key]
        p13 = pred_l13[key] if w_id == low else 1 - pred_l13[key]
        shift = p13 - p12
        all_game_list.append((f"{winner} > {loser}", p12, p13, shift, label))

all_game_list.sort(key=lambda x: abs(x[3]), reverse=True)
for game, p12, p13, shift, label in all_game_list[:15]:
    print(f"  {game:<33} {p12:.3f}   {p13:.3f}   {shift:+.3f}   {label}")

print("\nDone!")
