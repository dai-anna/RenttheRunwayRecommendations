# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tensorflow import keras
import tensorflow_recommenders as tfrs
import wandb
import yaml
from wandb.keras import WandbCallback
import time
import pickle

# %%
# load data parquet
df = pd.read_parquet("../../artifacts/nndata.parquet")

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
class DataLoader:
    def __init__(
        self,
        df: pd.DataFrame,
        embedding_cols: list,
        ohe_cols: list,
        continuous_cols: list,
    ):
        self.df = df
        self.embedding_cols = embedding_cols
        self.ohe_cols = ohe_cols
        self.continuous_cols = continuous_cols

    def _process(self, df: pd.DataFrame):
        # cat to integer
        if not self.embedding_cols:
            self.oe = OrdinalEncoder(dtype=np.int8)
            if self.training:
                df[self.embedding_cols] = self.oe.fit_transform(df[self.embedding_cols])
            else:
                df[self.embedding_cols] = self.oe.transform(df[self.embedding_cols])

        # cat to ohe
        if not self.ohe_cols:
            self.ohe = OneHotEncoder(sparse=False, dtype=np.int8)
            if self.training:
                df[self.ohe_cols] = self.ohe.fit_transform(df[self.ohe_cols])
            else:
                df[self.ohe_cols] = self.ohe.transform(df[self.ohe_cols])

        # scale continuous
        if not self.continuous_cols:
            self.scaler = StandardScaler()
            if self.training:
                df[self.continuous_cols] = self.scaler.fit_transform(
                    df[self.continuous_cols]
                )
            else:
                df[self.continuous_cols] = self.scaler.transform(
                    df[self.continuous_cols]
                )

        return df

    def load(self, training=True):
        raw = self.df
        self.training = training
        processed = self._process(raw)
        return processed


# %%
# load configs
with open("config.yaml") as f:
    config = yaml.safe_load(f)


EMBEDDING_COLS = config.get("data_config").get("embedding_cols")
CONTINUOUS_COLS = config.get("data_config").get("continuous_cols")
OHE_COLS = config.get("data_config").get("ohe_cols")
MF_COLS = config.get("data_config").get("mf_cols")
# USER_COLS =
# ITEM_COLS =

# check if all data is loaded
check = set(EMBEDDING_COLS) | set(CONTINUOUS_COLS) | set(OHE_COLS) | set(MF_COLS)
set(df.columns) - check  # only recommend left which is the response


# %%
def df_to_dataset(df: pd.DataFrame, labels: str, shuffle: bool = True):
    df = df.copy()
    labels = df.pop(labels)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    return ds


# %%
# extract cardinality
cardinalities = df.nunique().to_dict()

#%%
# prepare data to load into tf
train = DataLoader(
    train,
    embedding_cols=EMBEDDING_COLS,
    ohe_cols=OHE_COLS,
    continuous_cols=CONTINUOUS_COLS,
).load(training=True)

val = DataLoader(
    val,
    embedding_cols=EMBEDDING_COLS,
    ohe_cols=OHE_COLS,
    continuous_cols=CONTINUOUS_COLS,
).load(training=False)

test = DataLoader(
    test,
    embedding_cols=EMBEDDING_COLS,
    ohe_cols=OHE_COLS,
    continuous_cols=CONTINUOUS_COLS,
).load(training=False)

# %%
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


# %%
# split dataset into user specific and item specific
user_specific_train = train_ds.filter(lambda x, y: x["user_id"] in user_specific)


# %%
################## MODEL HERE ##########################

# %%
# initialize tracking
wandb.init(project="my-test-project", entity="annadai")
