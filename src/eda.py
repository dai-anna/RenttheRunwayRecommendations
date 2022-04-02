# %%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # experimental feature
from sklearn.impute import IterativeImputer

# %%
# read in data
df = pd.read_csv("../artifacts/cleandata.csv")

###############################################################################
########################## IMPUTE MISSING VALUES ##############################
# %%
# check missing values
df.isnull().sum()

# %%
imp_mean = IterativeImputer(random_state=42)
imp_mean.fit(df)
# %%
# add response variable
df["recommend"] = df.rating == 10
# %%
df["recommend"].value_counts()
# %%
# impute missing values
df["bust_size_num"].fillna(df.bust_size_num.mean(), inplace=True)
