#%%
# import libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
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
# load data from parquet

df = pd.read_parquet("../artifacts/imputeddata.parquet")

c0 = pd.read_csv("../artifacts/cluster_0.csv")
c1 = pd.read_csv("../artifacts/cluster_1.csv")
c2 = pd.read_csv("../artifacts/cluster_2.csv")

c0_users = c0.user_id.unique().astype(list)
c1_users = c1.user_id.unique().astype(list)
c2_users = c2.user_id.unique().astype(list)

clustered_dfs = []
for idx in range(3):
    split_df = df[df.user_id.isin(eval(f"c{idx}_users"))]
    clustered_dfs.append({"name": f"Cluster {idx}", "data": split_df})


# %%
# define hyperparameter search space
pbounds = {
    "learning_rate": (0.01, 1.0),
    "n_estimators": (100, 1000),
    "max_depth": (3, 20),
    "reg_alpha": (0, 1),
    "reg_lambda": (0, 1),
    "min_child_weight": (1, 6),
    "gamma": (0, 9),
}

# %%
best_params_clusters = []
# run loop for bayesian optimization
for idx, cluster in enumerate(clustered_dfs):

    # X_train, X_test, y_train, y_test = prep_data_clf(cluster["data"], kfold=False)
    X, y = prep_data_clf(cluster["data"], kfold=True)
    X_train = X
    y_train = y
    X_test = X
    y_test = y

    def xgboost_hyper_param(
        learning_rate,
        n_estimators,
        max_depth,
        reg_alpha,
        reg_lambda,
        min_child_weight,
        gamma,
    ):

        max_depth = int(max_depth)
        n_estimators = int(n_estimators)

        clf = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            use_label_encoder=False,
            seed=1234,
            n_jobs=-1,
        )
        return np.mean(cross_val_score(clf, X_train, y_train, cv=3, scoring="roc_auc"))

    # define optimizer for bayesian search
    optimizer = BayesianOptimization(
        f=xgboost_hyper_param,
        pbounds=pbounds,
        random_state=1234,
    )

    # tune hyperparameters
    IWANTTOWAITHOURSTOTUNETHESEPARAMS = False
    if IWANTTOWAITHOURSTOTUNETHESEPARAMS:
        logger = JSONLogger(path=f"../artifacts/logs/c_{idx+1}of3.json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        print(f"Tuning hyperparameters for cluster {idx+1} of 3")
        optimizer.maximize(
            init_points=3,
            n_iter=5,
        )
        # save the best hyperparameters to disk
        best_params = optimizer.max
        best_params_clusters.append(best_params)
        print(f"Finished {idx+1} of 3")
        with open(f"../models/params/best_params_{idx+1}of3.pkl", "wb") as f:
            pickle.dump(best_params, f)
    else:
        with open(f"../models/params/best_params_{idx+1}of3.pkl", "rb") as f:
            best_params = pickle.load(f)
            best_params_clusters.append(best_params)


# %%
# run loop to train model with best hyperparameters
IWANTTORETRAINMYMODELS = False
if IWANTTORETRAINMYMODELS:
    models = []

    for idx, best_params in enumerate(best_params_clusters):
        print(f"Starting cluster {idx+1}")

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
            verbosity=None,
        )

        X, y = prep_data_clf(clustered_dfs[idx]["data"], kfold=True)
        clf_tuned.fit(X_train, y_train)
        print(f"Finished training model for cluster {idx}")

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
                "name": f"Cluster {idx}",
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

        print(f"Done with cluster {idx+1}")

        with open(f"../models/cluster_models_final.pkl", "wb") as f:
            pickle.dump(models, f)
    else:
        with open(f"../models/cluster_models_final.pkl", "rb") as f:
            models = pickle.load(f)

# %%
models

#%%
# check lift
for model in models:
    print(f"{model['name']}")
    print(f"model['ap_score']")
    print(f"Lift: {model['ap_score']/model['baseline']}")

# %%
for cluster in clustered_dfs:
    print(sum(cluster["recommend"]) / len(cluster["recommend"]))

# %%
# check random chance
print(sum(c0["recommend"]) / len(c0["recommend"]))
print(sum(c1["recommend"]) / len(c1["recommend"]))
print(sum(c2["recommend"]) / len(c2["recommend"]))
# %%
