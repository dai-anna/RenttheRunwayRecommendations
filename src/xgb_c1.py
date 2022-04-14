#%%
# import libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import pickle

# %%
# load data from parquet
df = pd.read_parquet("../artifacts/cluster_0.parquet")

# drop time features and rating
df = df.drop(
    [
        "review_month",
        "review_day_of_month",
        "rating",
        "review_year",
        "review_date",
    ],
    axis=1,
)
features = pd.get_dummies(df.drop("recommend", axis=1)).astype(int)

# %%
# for k-fold cross validation
X = features
y = df.recommend

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    features, df.recommend, test_size=0.3, random_state=1234
)

# %%
# define hyperparameter search space
pbounds = {
    "learning_rate": (0.01, 1.0),
    "n_estimators": (100, 1000),
    "max_depth": (3, 20),
    "reg_alpha": (0, 1),
    "reg_lambda": (0, 1),
    "min_child_weight": (1, 6),
    "subsample": (1.0, 1.0),  # Change for big datasets
    "colsample": (1.0, 1.0),  # Change for datasets with lots of features
    "gamma": (0, 9),
}


def xgboost_hyper_param(
    learning_rate,
    n_estimators,
    max_depth,
    reg_alpha,
    reg_lambda,
    min_child_weight,
    subsample,
    colsample,
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
    return np.mean(cross_val_score(clf, X, y, cv=3, scoring="roc_auc"))


optimizer = BayesianOptimization(
    f=xgboost_hyper_param,
    pbounds=pbounds,
    random_state=1234,
)

#%%
# tune hyperparameters
IWANTTOWAITANOTHER20MINTOTUNETHESEPARAMS = False
if IWANTTOWAITANOTHER20MINTOTUNETHESEPARAMS:
    logger = JSONLogger(path="../artifacts/logs/c0.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=5,
        n_iter=5,
    )
    # save the best hyperparameters to disk
    best_params = optimizer.max
    with open("../artifacts/best_params_c0.pkl", "wb") as f:
        pickle.dump(best_params, f)
else:
    with open("../artifacts/best_params_c0.pkl", "rb") as f:
        best_params = pickle.load(f)

# %%
print(best_params)

# %%
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

# %%
# evaluate model
preds = clf_tuned.predict(X)
roc_auc_score(y, preds)
accuracy_score(y, preds)

# %%
# check baseline accuracy

t_hat = 0
for t in y:
    t_hat += t

t_hat / len(y)  # 63.79% accurate

# %%
print("Baseline accuracy:", t_hat / len(y))
print("Model accuracy:", accuracy_score(y, preds))
print("Model AUC:", roc_auc_score(y, preds))

# %%
