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
# Random chance

###############################################################################
############################## RANDOM CHANCE ##################################

# Check Baseline random chance on validation set
t_hat = 0
for t in val.iloc[:, -1]:
    t_hat += t

t_hat / len(val)  # 65% accurate


# %%
rec = ranking(2000)
rec.sort(key=lambda x: x[1], reverse=True)
print(rec[:20])
