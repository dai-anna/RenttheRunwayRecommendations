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
from surprise.model_selection.split import train_test_split
from surprise import accuracy


# %%
# read in data
# NOTE: we will only do EDA on training set
df = pd.read_parquet("../artifacts/train.parquet")


# %%
# Load data into surprise
df_cluster = df[["user_id", "item_id", "recommend"]].rename(
    {"user_id": "userID", "item_id": "itemID"}, axis=1
)
reader = Reader(rating_scale=(0, 1))
ds = Dataset.load_from_df(df_cluster, reader)
train, test = train_test_split(ds, test_size=0.25, random_state=1)


# %%
# A few helper functions
def accuracy(preds):
    correct = 0
    for pred in preds:
        p = 1 if pred.est > 0.5 else 0
        if p == pred.r_ui:
            correct += 1
    return correct / len(preds)


def ranking(user_id):
    recs = []
    rated = df.loc[df["user_id"] == user_id, "item_id"].unique()
    print(rated)
    for idx in range(5000):
        if idx in rated:
            continue
        p = mf.predict(uid=user_id, iid=idx)
        recs.append((idx, p.est))
    return recs


# %%
# Fit SVD (Matrix factorization)
# This is a collaborative filtering approach

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
rec = ranking(2000)
rec.sort(key=lambda x: x[1], reverse=True)
print(rec[:20])
