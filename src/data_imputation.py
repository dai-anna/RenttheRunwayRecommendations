#%%
# import libraries
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # experimental feature
from sklearn.impute import IterativeImputer

imp_mean = IterativeImputer(random_state=42)

# %%
# read in data
df_cleaned = pd.read_parquet(
    "../artifacts/cleandata.parquet",
    engine="pyarrow",
)
# drop time features due to our assumptions

print(df_cleaned.isna().sum())
df_cleaned = df_cleaned.drop(
    ["review_month", "review_day_of_month", "review_year", "review_date"], axis=1
)
features = pd.get_dummies(df_cleaned)
imp_mean.fit(features)

X = pd.DataFrame(imp_mean.transform(features), columns=features.columns)

# %%
# save to disk
X.to_parquet("../artifacts/imputeddata.parquet")
