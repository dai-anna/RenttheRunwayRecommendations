# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pickle


# %%
# load data parquet
df = pd.read_parquet("../artifacts/reduceddata.parquet")

# %%
tf.constant(df)
# %%
