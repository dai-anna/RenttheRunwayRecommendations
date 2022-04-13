#%%
# import libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score, roc_auc_score

# %%
# load data from parquet
df = pd.read_parquet("../artifacts/cluster_3.parquet")

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

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    features, df.recommend, test_size=0.3, random_state=1234
)

# %%
# define hyperparameter search space
space = {
    "n_estimators": hp.uniform("n_estimators", 100, 500),
    "learning_rate": hp.uniform("learning_rate", 0, 0.5),
    "max_depth": hp.quniform("max_depth", 3, 18, 1),
    "gamma": hp.uniform("gamma", 1, 9),
    "reg_alpha": hp.quniform("reg_alpha", 40, 180, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
}


# %%
# tune the model
def objective(space):
    clf = XGBClassifier(
        n_estimators=int(space["n_estimators"]),
        max_depth=int(space["max_depth"]),
        learning_rate=space["learning_rate"],
        gamma=space["gamma"],
        reg_alpha=space["reg_alpha"],
        min_child_weight=space["min_child_weight"],
        colsample_bytree=space["colsample_bytree"],
        use_label_encoder=False,
        seed=1234,
        n_jobs=-1,
    )

    evaluation = [(X_train, y_train), (X_test, y_test)]

    clf.fit(
        X_train,
        y_train,
        eval_set=evaluation,
        eval_metric="auc",
        early_stopping_rounds=10,
        verbose=False,
    )

    pred = clf.predict(X_test)
    score = roc_auc_score(y_test, pred)
    loss = 1 - score
    return {"loss": loss, "status": STATUS_OK}


best_hyperparams = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)


print("The best hyperparameters are : ", "\n")
print(best_hyperparams)

# %%

# retrain model
clf_tuned = XGBClassifier(
    n_estimators=int(best_hyperparams["n_estimators"]),
    max_depth=int(best_hyperparams["max_depth"]),
    gamma=best_hyperparams["gamma"],
    reg_alpha=best_hyperparams["reg_alpha"],
    min_child_weight=best_hyperparams["min_child_weight"],
    colsample_bytree=best_hyperparams["colsample_bytree"],
    use_label_encoder=False,
    seed=1234,
    n_jobs=-1,
)

evaluation = [(X_train, y_train), (X_test, y_test)]

clf_tuned.fit(
    X_train,
    y_train,
    eval_set=evaluation,
    eval_metric="auc",
    early_stopping_rounds=25,
    verbose=False,
)

# %%
pred = clf_tuned.predict(X_test)
roc_auc_score(y_test, pred)
# %%
# check baseline accuracy

t_hat = 0
for t in y_test:
    t_hat += t

t_hat / len(y_test)  # 62.34% accurate


# %%
