# %%
import keras
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(1234)

# df_m = pd.read_parquet(
#     "../artifacts/reduceddata.parquet", engine="pyarrow"
# )  # this dataset should be a subset for each cluster

df_m = pd.read_csv("../artifacts/Clauster_0.csv")

# Processing the set for NN to train on
df_mn = df_m.dropna()
pairs = []

# creating indexes
item_index = {item: idx for idx, item in enumerate(df_mn.item_id.unique())}
index_item = {idx: item for item, idx in item_index.items()}
user_index = {user: idx for idx, user in enumerate(df_mn.user_id.unique())}
index_user = {idx: user for user, idx in user_index.items()}

# Iterate through each item
for item in df_mn["item_id"].unique():
    # Iterate through the users bought the item
    pairs.extend(
        (item_index[item], user_index[user])
        for user in df_mn[df_mn["item_id"] == item]["user_id"]
    )


def generate_batch(pairs, n_positive=50, negative_ratio=1.0, classification=False):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))

    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1

    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (item_id, user_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (item_id, user_id, 1)

        # Increment idx by 1
        idx += 1

        # Add negative examples until reach batch size
        while idx < batch_size:

            # random selection
            random_item = random.randrange(len(df_mn["item_id"].unique()))
            random_user = random.randrange(len(df_mn["user_id"].unique()))

            # Check to make sure this is not a positive example
            if (random_item, random_user) not in pairs_set:

                # Add to batch and increment index
                batch[idx, :] = (random_item, random_user, neg_label)
                idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {"item": batch[:, 0], "user": batch[:, 1]}, batch[:, 2]


# next(generate_batch(pairs, n_positive = 2, negative_ratio = 2)) #generates a batch

# NN embedding model


def item_embedding_model(embedding_size=50, classification=False):
    """Model to embed items and users using the functional API.
    Trained to discern if a link is present in a article"""

    # Both inputs are 1-dimensional
    item = Input(name="item", shape=[1])
    user = Input(name="user", shape=[1])

    # Embedding the item (shape will be (None, 1, 50))
    item_embedding = Embedding(
        name="item_embedding", input_dim=len(item_index), output_dim=embedding_size
    )(item)

    # Embedding the user (shape will be (None, 1, 50))
    user_embedding = Embedding(
        name="user_embedding", input_dim=len(user_index), output_dim=embedding_size
    )(user)

    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name="dot_product", normalize=True, axes=2)(
        [item_embedding, user_embedding]
    )

    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape=[1])(merged)

    # If classifcation, add extra layer and loss function is binary cross entropy
    if classification:
        merged = Dense(1, activation="sigmoid")(merged)
        model = Model(inputs=[item, user], outputs=merged)
        model.compile(
            optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs=[item, user], outputs=merged)
        model.compile(optimizer="Adam", loss="mse")

    return model


# Instantiate model and show parameters
model = item_embedding_model()
print(model.summary())


# train model
pairs_set = set(pairs)
n_positive = 512

gen = generate_batch(pairs, n_positive, negative_ratio=2)

# Train
h = model.fit_generator(
    gen,
    epochs=50,
    steps_per_epoch=len(pairs) // n_positive,
    verbose=2,
)
model.save("../models/cluster_0.h5")  # Save model here


# Extract Embeddings
item_layer = model.get_layer("item_embedding")
item_weights = item_layer.get_weights()[0]
item_weights = item_weights / np.linalg.norm(item_weights, axis=1).reshape((-1, 1))


### Apply the network here!
### Finding similar items

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
plt.rcParams["font.size"] = 15


def find_similar(
    name, weights, index_name="item", n=10, least=False, return_dist=False, plot=False
):
    """Find n most similar items (or least) to name based on embeddings. Option to also plot the results"""

    # Select index and reverse index
    if index_name == "item":
        index = item_index
        rindex = index_item
    elif index_name == "user":
        index = user_index
        rindex = index_user

    # Check to make sure `name` is in index
    try:
        # Calculate dot product between book and all others
        dists = np.dot(weights, weights[index[name]])
    except KeyError:
        print(f"{name} Not Found.")
        return

    # Sort distance indexes from smallest to largest
    sorted_dists = np.argsort(dists)

    # Plot results if specified
    if plot:

        # Find furthest and closest items
        furthest = sorted_dists[: (n // 2)]
        closest = sorted_dists[-n - 1 : len(dists) - 1]
        items = [rindex[c] for c in furthest]
        items.extend(rindex[c] for c in closest)

        # Find furthest and closets distances
        distances = [dists[c] for c in furthest]
        distances.extend(dists[c] for c in closest)

        colors = ["r" for _ in range(n // 2)]
        colors.extend("g" for _ in range(n))

        data = pd.DataFrame({"distance": distances}, index=items)

        # Horizontal bar chart
        data["distance"].plot.barh(
            color=colors, figsize=(10, 8), edgecolor="k", linewidth=2
        )
        plt.xlabel("Cosine Similarity")
        plt.axvline(x=0, color="k")

        # # Formatting for italicized title
        # name_str = f'{index_name.capitalize()}s Most and Least Similar to'
        # for word in name.split():
        #     # Title uses latex for italize
        #     name_str += ' $\it{' + word + '}$'
        # plt.title(name_str, x = 0.2, size = 28, y = 1.05)

        return None

    # If specified, find the least similar
    if least:
        # Take the first n from sorted distances
        closest = sorted_dists[:n]

        print(f"{index_name.capitalize()}s furthest from {name}.\n")

    # Otherwise find the most similar
    else:
        # Take the last n sorted distances
        closest = sorted_dists[-n:]

        # Need distances later on
        if return_dist:
            return dists, closest

        print(f"{index_name.capitalize()}s closest to {name}.\n")

    # Need distances later on
    if return_dist:
        return dists, closest

    # # Print formatting
    # max_width = max([len(rindex[c]) for c in closest])

    # Print the most similar and distances
    for c in reversed(closest):
        print(f"{index_name.capitalize()}: {rindex[c]} Similarity: {dists[c]:.{2}}")


## Example Usage
find_similar(1234, item_weights)  # 1234 is placeholder item_id

# %%
