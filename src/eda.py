# %%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # experimental feature
from sklearn.impute import IterativeImputer

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import KFold
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import validation_curve
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import cross_val_score


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

# %%
# check those with missing bust_size_num
df[df.bust_size_num.isnull()]
# %%
