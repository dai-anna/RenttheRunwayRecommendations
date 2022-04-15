##Initial Data Processing and EDA

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # experimental feature
from sklearn.impute import IterativeImputer
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans


df = pd.read_parquet(
    "/Users/sarwaridas/RenttheRunwayRecommendations/artifacts/cleandata.parquet"
)
df.head()

# sns.color_palette("rocket", as_cmap=True)
# sns.histplot(data=df, x="rating",binwidth=1,color="thistle").set(title='Frequency of ratings',xlabel="Rating out of 10")

# how many times did users rate multiple times?
count_purchases = df["user_id"].value_counts().sort_values(ascending=True)
print(
    f"Total number of users are {df.user_id.nunique()}. Things were rented multiple times by {count_purchases[count_purchases > 1].shape[0]} (unique) users. The max number of rentals by a single user was {count_purchases[count_purchases > 1].max()} and on average users rented {count_purchases[count_purchases > 1].mean():.2f} items."
)

# check how many times each item was rated more than once
count_ratings = df["item_id"].value_counts().sort_values(ascending=True)
print(
    f"There are {df['item_id'].nunique()} unique items out of which {count_ratings[count_ratings > 1].shape[0]} have been rated more than once. "
)


# dropping features related to time
features = df.drop(
    [
        "recommend",
        "review_month",
        "review_day_of_month",
        "rating",
        "review_year",
        "review_date",
        "keep",
    ],
    axis=1,
)
features = pd.get_dummies(features)
imp_mean = IterativeImputer(random_state=42)
imp_mean.fit(features)

imputed = pd.DataFrame(imp_mean.transform(features), columns=features.columns)
imputed.to_parquet("imputeddta.parquet")  ##saving imputed, OHE data


##trying clustering
filter_col = [col for col in imputed if col.startswith("bust_size_letter")]
bust_data = imputed.loc[:, filter_col]
imputed.drop(filter_col, axis=1, inplace=True)
bust_size = [x[-1] for x in pd.Series(bust_data.idxmax(axis=1))]
assert len(bust_size) == df.shape[0]
df["bust_size"] = bust_size

physical_features = [
    "user_id",
    "weight_lbs",
    "age",
    "size",
    "bust_size",
    "height_in",
]
df_cluster = imputed.loc[:, physical_features]
df_cluster["bust_size_code"] = pd.Categorical(df_cluster["bust_size"]).codes
users = (
    df_cluster[["user_id", "age", "size", "height_in", "weight_lbs", "bust_size"]]
    .groupby("user_id")
    .median()
    .reset_index()
)
kmeans = KMeans(n_clusters=3, random_state=1234).fit(users)
users["cluser_label"] = kmeans.labels_

x = users[["size", "weight_lbs", "cluser_label"]]
sns.set(rc={"figure.figsize": (10, 10)})  # width=3, #height=4
sns.color_palette("rocket", as_cmap=True)
sns.scatterplot(data=x, x="size", y="weight_lbs", hue="cluser_label")
