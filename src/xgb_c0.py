# %%
# import libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)
from sklearn.model_selection import RandomizedSearchCV


# %%
# load data csv
df = pd.read_csv("../artifacts/Cluster_0.csv")

# %%

IWANTTORERUNMYMODEL = False

# set up hyperparameter grid
learning_rate = [0.01, 0.05, 0.1, 0.3]
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
num_leaves = [5, 10, 20, 50]
max_depth = [3, 5, 10]

if IWANTTORERUNMYMODEL:
    tac = time.time()
    fit_params = {
        "eval_metric": "auc",
        "eval_set": [(X_val_boost, y_val)],
        "eval_names": ["validation"],
        "callbacks": [
            log_evaluation(period=1),
            early_stopping(stopping_rounds=10),
        ],
    }

    lgbm_randomsearch = RandomizedSearchCV(
        estimator=LGBMClassifier(random_state=42, n_jobs=-1),
        param_distributions={
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
        },
        n_iter=50,
        verbose=0,
        scoring="roc_auc",
        n_jobs=-1,
    ).fit(X_train_boost, y_train, **fit_params)

    tic = time.time()
    lgbm_time_tune = tic - tac

    with open("lgbm_randomsearch.pkl", "wb") as file:
        pickle.dump([lgbm_randomsearch, lgbm_time_tune], file)
else:
    with open("lgbm_randomsearch.pkl", "rb") as file:
        lgbm_randomsearch, lgbm_time_tune = pickle.load(file)


verbose = False
n_jobs = 4

# Define our models to try
clf_gbt = XGBClassifier(
    colsample_bytree=0.5052384738649442,
    gamma=1.8432456479627346,
    max_depth=17,
    min_child_weight=8,
    reg_alpha=124,
    reg_lambda=0.22040047754189196,
)

names = ["LR", "KNN", "RF", "GBT", "NN"]
clfs = [clf_lr, clf_knn, clf_rf, clf_gbt, clf_nn]

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
    scores = clf.predict_proba(X_val)
    time_predict[i] = (time.time() - t0) / 60
    print(f"Predicted {names[i]} in {time_predict[i]} min")

    # Evaluate performance metrics for each model
    fpr[i], tpr[i], _ = roc_curve(y_val, scores[:, 1], pos_label=1)
    precision[i], recall[i], _ = precision_recall_curve(
        y_val, scores[:, 1], pos_label=1
    )
    auc_roc[i] = auc(fpr[i], tpr[i])
    ap[i] = average_precision_score(y_val, scores[:, 1])

# %%
