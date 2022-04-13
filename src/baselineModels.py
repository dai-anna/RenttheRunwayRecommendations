# %%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# %%
# read in data
train = pd.read_parquet("../artifacts/train.parquet")
val = pd.read_parquet("../artifacts/val.parquet")
test = pd.read_parquet("../artifacts/test.parquet")

df = pd.read_parquet("../artifacts/cleandata.parquet")  # full data set -> do not use

dfs = [train, val, test]

# %%
# Random chance

###############################################################################
############################## RANDOM CHANCE ##################################

# Check Baseline random chance on validation set
t_hat = 0
for t in val.iloc[:, -1]:
    t_hat += t

t_hat / len(val)  # 65% accurate

