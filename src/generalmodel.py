################################
# Model Training
################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import train_test_split


import time

verbose = False
n_jobs = 4


df = pd.read_parquet(
    "imputeddta.parquet",
    engine="pyarrow",
)
df.head()
X = df.drop(["recommend"], axis=1)
y = df["recommend"]
X = X.to_numpy()
y = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1234
)


# Define our models to try
clf_lr = LogisticRegression(solver="saga", n_jobs=n_jobs, verbose=verbose)
clf_knn = KNeighborsClassifier(n_jobs=n_jobs, n_neighbors=5)
clf_rf = RandomForestClassifier(n_jobs=n_jobs, verbose=verbose)
clf_gbt = XGBClassifier()

names = ["LR", "KNN", "RF", "GBT"]
clfs = [clf_lr, clf_knn, clf_rf, clf_gbt]

fpr, tpr, precision, recall, auc_roc, ap = (
    dict(),
    dict(),
    dict(),
    dict(),
    dict(),
    dict(),
)
time_train, time_predict = dict(), dict()

for i, clf in enumerate(clfs):
    print(f"Training {names[i]}")
    t0 = time.time()
    clf.fit(X_train, y_train)
    time_train[i] = (time.time() - t0) / 60
    print(f"Trained {names[i]} in {time_train[i]} min")

    t0 = time.time()
    scores = clf.predict_proba(X_test)
    time_predict[i] = (time.time() - t0) / 60
    print(f"Predicted {names[i]} in {time_predict[i]} min")

    # Evaluate performance metrics for each model
    fpr[i], tpr[i], _ = roc_curve(y_test, scores[:, 1], pos_label=1)
    precision[i], recall[i], _ = precision_recall_curve(
        y_test, scores[:, 1], pos_label=1
    )
    auc_roc[i] = auc(fpr[i], tpr[i])
    ap[i] = average_precision_score(y_test, scores[:, 1])

################################
# Plot performance
################################
import matplotlib.pyplot as plt


def add_roc(ax, fpr, tpr, auc, name, **kwargs):
    ax.plot(fpr, tpr, label="{}, AUC={:0.3f}".format(name, auc), **kwargs)


def add_pr(ax, recall, precision, ap, name, **kwargs):
    ax.plot(recall, precision, label="{}, AP={:0.3f}".format(name, ap), **kwargs)


# Plot the ROC curves
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 6))

# Plot chancete
pos_ratio = sum(y_test) / len(y_test)
add_roc(ax_roc, [0, 1], [0, 1], 0.5, "Chance", color="grey", linestyle="--")
add_pr(
    ax_pr,
    [0, 1],
    [pos_ratio, pos_ratio],
    pos_ratio,
    "Chance",
    color="grey",
    linestyle="--",
)

# Plot each curve
for i, name in enumerate(names):
    add_roc(ax_roc, fpr[i], tpr[i], auc_roc[i], name)
    add_pr(ax_pr, recall[i], precision[i], ap[i], name)

# ROC Curve Formatting
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_xlim([0, 1])
ax_roc.set_ylim([0, 1])
ax_roc.grid(True)
ax_roc.legend()

# PR Curve Formatting
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_xlim([0, 1])
ax_pr.set_ylim([0, 1])
ax_pr.grid(True)
ax_pr.legend()

plt.tight_layout()
