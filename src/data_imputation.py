import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # experimental feature
from sklearn.impute import IterativeImputer

imp_mean = IterativeImputer(random_state=42)


df_cleaned = pd.read_parquet(
    "/Users/sarwaridas/RenttheRunwayRecommendations/artifacts/cleandata.parquet",
    engine="pyarrow",
)
print(df_cleaned.isna().sum())
df_cleaned = df_cleaned.drop(
    ["review_month", "review_day_of_month", "review_year", "review_date"], axis=1
)
features = pd.get_dummies(df_cleaned)
imp_mean.fit(features)

X = pd.DataFrame(imp_mean.transform(features), columns=features.columns)

X.to_parquet("imputeddta.parquet")
