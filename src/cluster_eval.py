# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# %%

# load data from csv
c0 = pd.read_csv("../artifacts/Clauster_0.csv")
c1 = pd.read_csv("../artifacts/Clauster_1.csv")
c2 = pd.read_csv("../artifacts/Clauster_2.csv")

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


# %%
# # load data from parquet
# clusters = pd.read_parquet("../artifacts/clustered_users.parquet")
# df_clean = pd.read_parquet("../artifacts/cleandata.parquet")
# df = pd.read_parquet("../artifacts/imputeddata.parquet")

# # %%
# # merge data
# df = pd.merge(df, clusters[["cluser_label", "user_id"]], on="user_id")
# df = pd.concat([df, df_clean["bust_size_letter"]], axis=1)

# # filter data by what we used to cluster
# df = df[
#     [
#         "user_id",
#         "age",
#         "size",
#         "bust_size_letter",
#         "height_in",
#         "weight_lbs",
#         "cluser_label",
#     ]
# ]
# le = LabelEncoder()
# df["bust_size_letter"] = le.fit_transform(df["bust_size_letter"])

# # %%
# # groupby to evaluate
# print(df.groupby("cluser_label").mean().reset_index())
# # %%
