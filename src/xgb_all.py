#%%
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
# load/prep data from parquet
df = pd.read_parquet("../artifacts/imputeddata.parquet")


X_train, X_test, y_train, y_test = prep_data_clf(df, kfold=False)

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

#%%
# tune hyperparameters
IWANTTOWAITANOTHER20MINTOTUNETHESEPARAMS = False
if IWANTTOWAITANOTHER20MINTOTUNETHESEPARAMS:
    logger = JSONLogger(path="../artifacts/logs/c_all.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=5,
        n_iter=5,
    )
    # save the best hyperparameters to disk
    best_params = optimizer.max
    with open("../artifacts/best_params_all.pkl", "wb") as f:
        pickle.dump(best_params, f)
else:
    with open("../artifacts/best_params_all.pkl", "rb") as f:
        best_params = pickle.load(f)

# %%
print(best_params)

# %%
IWANTTORETRAINMYMODEL = True
if IWANTTORETRAINMYMODEL:

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

    clf_tuned.fit(X_train, y_train)

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
    model = {
        "model": clf_tuned,
        "name": "General Model",
        "auc_score": auc_score,
        "precision": precision,
        "recall": recall,
        "ap_score": ap_score,
        "acc_score": acc_score,
        "baseline": baseline,
        "X_test": X_test,
        "y_test": y_test,
    }

    with open("../artifacts/model_all_test.pkl", "wb") as f:
        pickle.dump(model, f)
else:
    with open("../artifacts/model_all.pkl", "rb") as f:
        model = pickle.load(f)


# %%
model
