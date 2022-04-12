# %%
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tensorflow import keras
import wandb
import yaml
from wandb.keras import WandbCallback
import time
import pickle

# %%
df = pd.read_parquet("../../artifacts/nndata.parquet")


# %%
"""
1) split into user/item data

https://www.youtube.com/watch?v=jz0-satrmrA&list=PLQY2H8rRoyvy2MiyUBz5RWZr5MPFkV3qz&index=3

"""

user_specific = [
    "user_id",
    "weight_lbs",
    "body_type",
    "age",
    "bust_size_num",
    "bust_size_letter",
    "height_in",
]


# %%
# train-val-test split
train, val_test = train_test_split(
    df.copy(),
    test_size=0.4,
    random_state=42,
)

val, test = train_test_split(
    val_test,
    test_size=0.5,
    random_state=42,
)

# %%
def df_to_dataset(df: pd.DataFrame, labels: str, shuffle: bool = True):
    df = df.copy()
    labels = df.pop(labels)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    return ds


# %%
# %%
cardinalities = df.nunique().to_dict()

train_ds = df_to_dataset(
    df=train,
    labels="recommend",
)

validation_ds = df_to_dataset(
    df=val,
    labels="recommend",
)

test_ds = df_to_dataset(
    df=test,
    labels="recommend",
)


#%%
# try keras encoder
category_lookup = tf.keras.layers.StringLookup()
category_lookup.adapt(train_ds.map(lambda x: x["category"]))


# %%
user_id_lookup = tf.keras.layers.StringLookup()
user_id_lookup.adapt(df["recommend"].map(lambda x: x["user_id"]))


# %%


# class UserModel(tf.keras.Model):
#     def __init__(self, use_timestamps):
#         super().__init__()

#         self._use_timestamps = use_timestamps

#         self.user_embedding = tf.keras.Sequential(
#             [
#                 tf.keras.layers.StringLookup(
#                     vocabulary=unique_user_ids, mask_token=None
#                 ),
#                 tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
#             ]
#         )

#         if use_timestamps:
#             self.timestamp_embedding = tf.keras.Sequential(
#                 [
#                     tf.keras.layers.Discretization(timestamp_buckets.tolist()),
#                     tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
#                 ]
#             )
#             self.normalized_timestamp = tf.keras.layers.Normalization(axis=None)

#             self.normalized_timestamp.adapt(timestamps)

#     def call(self, inputs):
#         if not self._use_timestamps:
#             return self.user_embedding(inputs["user_id"])

#         return tf.concat(
#             [
#                 self.user_embedding(inputs["user_id"]),
#                 self.timestamp_embedding(inputs["timestamp"]),
#                 tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1)),
#             ],
#             axis=1,
#         )


# class MovieModel(tf.keras.Model):
#     def __init__(self):
#         super().__init__()

#         max_tokens = 10_000

#         self.title_embedding = tf.keras.Sequential(
#             [
#                 tf.keras.layers.StringLookup(
#                     vocabulary=unique_movie_titles, mask_token=None
#                 ),
#                 tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 32),
#             ]
#         )

#         self.title_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)

#         self.title_text_embedding = tf.keras.Sequential(
#             [
#                 self.title_vectorizer,
#                 tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
#                 tf.keras.layers.GlobalAveragePooling1D(),
#             ]
#         )

#         self.title_vectorizer.adapt(movies)

#     def call(self, titles):
#         return tf.concat(
#             [
#                 self.title_embedding(titles),
#                 self.title_text_embedding(titles),
#             ],
#             axis=1,
#         )


# class MovielensModel(tfrs.models.Model):
#     def __init__(self, use_timestamps):
#         super().__init__()
#         self.query_model = tf.keras.Sequential(
#             [UserModel(use_timestamps), tf.keras.layers.Dense(32)]
#         )
#         self.candidate_model = tf.keras.Sequential(
#             [MovieModel(), tf.keras.layers.Dense(32)]
#         )
#         self.task = tfrs.tasks.Retrieval(
#             metrics=tfrs.metrics.FactorizedTopK(
#                 candidates=movies.batch(128).map(self.candidate_model),
#             ),
#         )

#     def compute_loss(self, features, training=False):
#         # We only pass the user id and timestamp features into the query model. This
#         # is to ensure that the training inputs would have the same keys as the
#         # query inputs. Otherwise the discrepancy in input structure would cause an
#         # error when loading the query model after saving it.
#         query_embeddings = self.query_model(
#             {
#                 "user_id": features["user_id"],
#                 "timestamp": features["timestamp"],
#             }
#         )
#         movie_embeddings = self.candidate_model(features["movie_title"])

#         return self.task(query_embeddings, movie_embeddings)
