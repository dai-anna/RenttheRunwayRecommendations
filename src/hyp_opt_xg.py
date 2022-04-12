from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import xgboost as xgb
from sklearn.metrics import accuracy_score

space = {
    "max_depth": hp.quniform("max_depth", 3, 18, 1),
    "gamma": hp.uniform("gamma", 1, 9),
    "reg_alpha": hp.quniform("reg_alpha", 40, 180, 1),
    "reg_lambda": hp.uniform("reg_lambda", 0, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
    "n_estimators": 180,
    "seed": 1234,
}


def objective(space):
    clf = xgb.XGBClassifier(
        n_estimators=space["n_estimators"],
        max_depth=int(space["max_depth"]),
        gamma=space["gamma"],
        reg_alpha=int(space["reg_alpha"]),
        min_child_weight=int(space["min_child_weight"]),
        colsample_bytree=int(space["colsample_bytree"]),
    )

    evaluation = [(X_train, y_train), (X_val, y_val)]

    clf.fit(
        X_train,
        y_train,
        eval_set=evaluation,
        eval_metric="auc",
        early_stopping_rounds=10,
        verbose=False,
    )

    pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, pred > 0.5)
    print("SCORE:", accuracy)
    return {"loss": -accuracy, "status": STATUS_OK}


trials = Trials()

best_hyperparams = fmin(
    fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials
)


print("The best hyperparameters are : ", "\n")
print(best_hyperparams)
