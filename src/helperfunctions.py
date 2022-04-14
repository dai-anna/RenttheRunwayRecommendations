import pandas as pd
from sklearn.model_selection import train_test_split

# Add reusuable helpber functions here


def prep_data_clf(data: pd.DataFrame, kfold: bool = True):
    # drop rating for predictions because response variable is built from ratings
    try:
        data = data.copy().drop("rating", axis=1)
    except:
        pass

    if kfold:
        # for k-fold cross validation
        X = data.drop("recommend", axis=1)
        y = data.recommend
        return X, y

    else:
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop("recommend", axis=1),
            data.recommend,
            test_size=0.3,
            random_state=1234,
        )
        return X_train, X_test, y_train, y_test
