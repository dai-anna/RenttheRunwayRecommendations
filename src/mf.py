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
train = pd.read_parquet("../artifacts/train.parquet")
val = pd.read_parquet("../artifacts/val.parquet")
test = pd.read_parquet("../artifacts/test.parquet")

df = pd.read_parquet("../artifacts/cleandata.parquet")  # full data set -> do not use

dfs = [train, val, test]

# %%
# try to predict


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
