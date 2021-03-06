# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

import pickle

mpl.rcParams["font.sans-serif"] = "Arial"
mpl.rcParams["font.family"] = "sans-serif"

DUKENAVY = "#012169"
DUKEROYAL = "#00539B"
PGREEN = "#ACCFBF"
PPINK = "#EEBBCC"
PORANGE = "#FCC9A5"
PBLUE = "#C8E4FF"
PYELLOW = "#F8F1AE"


# %%
# read in pickle file
with open(f"../artifacts/cluster_models_final.pkl", "rb") as f:
    cluster_models = pickle.load(f)

with open(f"../artifacts/model_all.pkl", "rb") as f:
    general_model = pickle.load(f)

# %%
models = [general_model] + cluster_models

# %%
# read in data ROC/PR curves
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
colors = [DUKENAVY, PPINK, PGREEN, PORANGE, PBLUE, PYELLOW]
cluster_labels = ["(L)", "(M)", "(S)"]

for idx, model in enumerate(models):

    # plot the ROC curve
    RocCurveDisplay.from_estimator(
        model["model"],
        model["X_test"],
        model["y_test"],
        lw=2,
        color=colors[idx],
        name=f'{model["name"]}',
        ax=ax[0],
    )

    # plot the PR curve
    ax[1].plot(
        model["recall"],
        model["precision"],
        lw=2,
        color=colors[idx],
        label=f'{model["name"]} (AP = {round(model["auc_score"], 2)})',
    )
    ax[1].hlines(
        model["baseline"],
        xmin=0,
        xmax=1,
        lw=2,
        linestyle="--",
        color="lightgray",
    )

# plot baselines
ax[1].hlines(
    model["baseline"],
    xmin=0,
    xmax=1,
    lw=2,
    linestyle="--",
    color="lightgray",
    label="Random Guess",
)
ax[0].plot(
    [0, 1], [0, 1], color="lightgray", lw=2, linestyle="--", label="Random Guess"
)

# set ax level dimensions
ax[0].set_title("ROC Curve", fontsize=16, fontweight="bold")
ax[0].set_xlabel("False Positive Rate", fontsize=14)
ax[0].set_ylabel("True Positive Rate", fontsize=14)
ax[0].legend(loc="lower right", fontsize=12)

ax[1].set_title("Precision-Recall Curves", fontsize=16, fontweight="bold")
ax[1].set_xlabel("Recall", fontsize=14)
ax[1].set_ylabel("Precision", fontsize=14)
ax[1].legend(loc="upper right", fontsize=12)

# set figure level title
fig.suptitle("Evaluation on Test Dataset", fontsize=16, fontweight="bold")
fig.tight_layout()
sns.despine()
# %%
