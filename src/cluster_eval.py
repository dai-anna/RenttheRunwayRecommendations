# %%
import pandas as pd


# %%
# load data from csv
c0 = pd.read_csv("../artifacts/cluster_0.csv")
c1 = pd.read_csv("../artifacts/cluster_1.csv")
c2 = pd.read_csv("../artifacts/cluster_2.csv")

c0["cluster_label"] = 0
c1["cluster_label"] = 1
c2["cluster_label"] = 2

# concatenate clusters
df_full = pd.read_parquet("../artifacts/imputeddata.parquet")
df = pd.concat([c0, c1, c2])
# filter data by what we used to cluster
df = df[
    [
        "user_id",
        "age",
        "size",
        "bust_size_letter",
        "height_in",
        "weight_lbs",
        "cluster_label",
    ]
]

# %%
evaldf = df.groupby("cluster_label").mean().reset_index()
evaldf = (
    evaldf[["cluster_label", "size", "weight_lbs"]]
    .round(2)
    .rename({"cluster_label": "cluster"}, axis=1)
)
evaldf["label"] = ["large (L)", "medium (M)", "small (S)"]

# %%
with open("../artifacts/clustereval.tex", "w") as tf:
    tf.write(evaldf.to_latex(index=False))

