import numpy as np
import pandas as pd
from metadata.duckdb.model_metadata_db import AnalysisStore
from performance_prediction.train_xgboost import train_regressor

# CONFIG
DB_PATH   = "./metadata/duckdb/2025_03_03.duckdb"
BENCH     = "truthfulqa_mc1"
SHOT      = "0-shot"
MODEL_ID  = "EleutherAI/pythia-410m"
FEAT1     = "domain_web_pct_mean"
FEAT2     = None

REG_KWARGS = dict(regressor="xgboost", lr=0.1, max_depth=10, n_estimators=100)

# 1) load & filter
store = AnalysisStore.from_existing(DB_PATH)
df = (
    store.con
         .execute(f"""
           SELECT m.id, d.*,
                  e.benchmark, e.setting, e.metric_value AS value
           FROM model_annotations m
           JOIN dataset_info       d USING(id)
           JOIN evaluation_results e USING(id)
           WHERE e.metric='accuracy'
         """)
         .df()
)
store.con.close()
df = df[(df.benchmark==BENCH)&(df.setting==SHOT)].copy()

# join with pseudo_feats

pseudo_feats_csv = '/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/all_models_feature_stats_3_03_with_ratios.csv'

pseudo_feats_df = pd.read_csv(pseudo_feats_csv)

df = df.merge(
    pseudo_feats_df,
    how="left",
    left_on="id",
    right_on="id",
    suffixes=("", ""),
)

# 2) build X,y
X = (
    df
    .drop(columns=["benchmark","setting","value"])
    .drop_duplicates(subset="id")
    .set_index("id")
)
y = df.set_index("id")["value"]

# 3) train
res = train_regressor(X, y, **REG_KWARGS)
model = res[0] if isinstance(res, tuple) else res

# 4) grab base vector & zero out other percentages
base = X.loc[MODEL_ID].copy()
pct_cols = [c for c in X.columns if c.startswith("pretraining_summary_percentage_")]
for c in pct_cols:
    if c not in (FEAT1, FEAT2):
        base[c] = 0.0

# 5) compute sum constant & sweep
S = 100
vals = np.linspace(0, S, 21)
out = []

if FEAT2 is None:
    # Single feature sweep
    for v1 in vals:
        v = base.copy()
        v[FEAT1] = v1
        pred = model.predict(v.to_frame().T)[0]
        out.append((v1, pred))
    df_out = pd.DataFrame(out, columns=[FEAT1, f"pred_{BENCH}_{SHOT}"])
else:
    # Two feature sweep with constant sum
    for v1 in vals:
        v = base.copy()
        v[FEAT1] = v1
        v[FEAT2] = S - v1
        pred = model.predict(v.to_frame().T)[0]
        out.append((v1, S-v1, pred))
    df_out = pd.DataFrame(out, columns=[FEAT1, FEAT2, f"pred_{BENCH}_{SHOT}"])

print(df_out.to_string(index=False))