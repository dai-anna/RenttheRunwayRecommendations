# %%
# import libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # experimental feature
from sklearn.impute import IterativeImputer

mpl.rcParams["font.sans-serif"] = "Arial"
mpl.rcParams["font.family"] = "sans-serif"

# stitch fix colors - can change
BLUE = "#66B2A3"
PINK = "#EA8988"

# %%
# read in data
# NOTE: we will only do EDA on training set
df = pd.read_parquet("../artifacts/train.parquet")


# %%
###############################################################################
################################ SIMPLE EDA ###################################


# %%
# check how many items each user rated
df["user_id"].value_counts().sort_values(ascending=True)
# >> Each user rated 1-239 items

# plot distribution of ratings per user
fig, ax = plt.subplots()
sns.distplot(df["user_id"].value_counts().sort_values(ascending=True))

# find 99.9% percentile of # ratings per user
df["user_id"].value_counts().sort_values(ascending=True).quantile(
    0.999
)  # >> 22 ratings/user


# %%
# check how many times each item was rated
df["item_id"].value_counts().sort_values(ascending=True)
# >> Each item was rated by 1-1354 users

sns.distplot(df["item_id"].value_counts().sort_values(ascending=True))

# find 99.9% percentile of # ratings per item
df["item_id"].value_counts().sort_values(ascending=True).quantile(
    0.999
)  # >> 788 ratings/item


# %%
# calculate average rating per item
df.groupby("item_id").mean()["recommend"].sort_values(ascending=True)
# people like items from 0% of the time to 100% of the time -> very large difference

# plot distribution of average rating per item
sns.barplot(x="item_id", y=df["recommend"], data=df, ci=None, palette=[BLUE, PINK])

# %%
# Check how how many times each item was rated
df_pivot = pd.pivot_table(df, index="user_id", columns="item_id", values="recommend")
df_pivot.notna().mean()
# >> less than 0.01% of users have rated each item -> immense cold start issues


# %%
###############################################################################
########################## IMPUTE MISSING VALUES ##############################
# %%
# check missing values
df.isnull().sum()

# %%
imp_mean = IterativeImputer(random_state=42)
imp_mean.fit(df)
# %%

# %%
df["recommend"].value_counts()
# %%
# impute missing values
df["bust_size_num"].fillna(df.bust_size_num.mean(), inplace=True)

# %%
# check those with missing bust_size_num
df[df.bust_size_num.isnull()]
# %%
