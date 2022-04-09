# %%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.dataset import Dataset
from surprise import Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import accuracy
from sklearn.model_selection import train_test_split
from helperfunctions import accuracy, ranking


# %%
# read in data
train = pd.read_parquet("../artifacts/train.parquet")
val = pd.read_parquet("../artifacts/val.parquet")
test = pd.read_parquet("../artifacts/test.parquet")

df = pd.read_parquet("../artifacts/cleandata.parquet")  # full data set -> do not use

dfs = [train, val, test]


# %%
# Load data into surprise
datasets = []
for df in dfs:
    df_cluster = df[["user_id", "item_id", "recommend"]].rename(
        {"user_id": "userID", "item_id": "itemID"}, axis=1
    )
    reader = Reader(rating_scale=(0, 1))
    ds = Dataset.load_from_df(df_cluster, reader)
    datasets.append(ds)


# %%
# Collaborative filtering approach: Matrix Factorization

###############################################################################
######################## FIT SVD (MATRIX FACTORIZATION) #######################

IWANTTORERUNMF = True

if IWANTTORERUNMF:
    mf = SVD()
    mf.fit(train)
    # save to disk
    with open("mf_model.pkl", "wb") as file:
        pickle.dump(mf, file)
else:
    # load model from disk
    with open("mf_model.pkl", "rb") as file:
        mf = pickle.load(file)

preds_mf = mf.test(test)


#%%
# Check accuracy of mf
accuracy(preds_mf)


# %%
###############################################################################
######### EXPERIMENT WITH LEAVING OUT THOSE WHO ONLY BOUGHT ONE ITEM ##########

# try find users who only have one item in their history
users_items = df.groupby("user_id").count()["item_id"].sort_values(ascending=True)
# find user ids that only have more than one item in their history
user_ids_morethan1 = users_items[users_items > 1].index
df_reduced = df[df["user_id"].isin(user_ids_morethan1)]

# # Train/Test split
# X = df_reduced.drop(["recommend"], axis=1)
# y = df_reduced["recommend"]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )

# Train/Test split
train_reduced, test_reduced = train_test_split(
    df_reduced, test_size=0.3, random_state=42
)

dfs = [train_reduced, test_reduced]

# Load data into surprise
datasets = []
for df in dfs:
    df_cluster = df[["user_id", "item_id", "recommend"]].rename(
        {"user_id": "userID", "item_id": "itemID"}, axis=1
    )
    reader = Reader(rating_scale=(0, 1))
    ds = Dataset.load_from_df(df_cluster, reader)
    datasets.append(ds)


mf_reduced = SVD()
mf_reduced.fit(train)

# %%
