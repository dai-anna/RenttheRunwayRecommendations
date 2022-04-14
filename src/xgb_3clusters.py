# %%
# import libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.model_selection import cross_val_score
import pickle
from helperfunctions import prep_data_clf

# %%
# load data from csv
c0 = pd.read_csv("../artifacts/Clauster_0.csv")
c0users = c0.user_id.unique().astype(list)
c1 = pd.read_csv("../artifacts/Clauster_1.csv")
c1users = c1.user_id.unique().astype(list)
c2 = pd.read_csv("../artifacts/Clauster_2.csv")
c2users = c2.user_id.unique().astype(list)

df = pd.read_parquet("../artifacts/imputeddata.parquet")

# join in with the clusters
clustered_dfs = []
for idx in range(3):
    clustered_dfs.append(df[df.user_id.isin(eval("c{}users".format(idx)))])

# define hyperparameter search space
with open("../artifacts/best_params_all.pkl", "rb") as f:
    best_params = pickle.load(f)

# %%
models = []
# loop to train model
for idx, cluster in enumerate(clustered_dfs):
    print(f"Training model for cluster {idx}")
    # train model using optimal hyperparameters
    clf_tuned = XGBClassifier(
        learning_rate=best_params["params"]["learning_rate"],
        n_estimators=int(best_params["params"]["n_estimators"]),
        max_depth=int(best_params["params"]["max_depth"]),
        reg_alpha=best_params["params"]["reg_alpha"],
        reg_lambda=best_params["params"]["reg_lambda"],
        min_child_weight=best_params["params"]["min_child_weight"],
        gamma=best_params["params"]["gamma"],
        use_label_encoder=False,
        seed=1234,
        n_jobs=-1,
    )

    X_train, X_test, y_train, y_test = prep_data_clf(cluster, kfold=False)

    clf_tuned.fit(X_train, y_train)

    print(f"Saving metrics for cluster {idx}")
    # evaluate model
    preds = clf_tuned.predict(X_test)
    preds_proba = clf_tuned.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, preds_proba)
    acc_score = accuracy_score(y_test, preds)
    precision, recall, _ = precision_recall_curve(y_test, preds_proba)
    fpr, tpr, _ = roc_curve(y_test, preds)
    ap_score = auc(fpr, tpr)
    baseline = sum(y_test) / len(y_test)
    # save model
    models.append(
        {
            "model": clf_tuned,
            "name": f"Cluster {idx+1}",
            "auc_score": auc_score,
            "precision": precision,
            "recall": recall,
            "ap_score": ap_score,
            "acc_score": acc_score,
            "baseline": baseline,
            "X_test": X_test,
            "y_test": y_test,
        }
    )
    print(f"Done with cluster {idx} complete")


# %%
with open(f"../artifacts/cluster_models_final.pkl", "wb") as f:
    pickle.dump(models, f)
# %%
