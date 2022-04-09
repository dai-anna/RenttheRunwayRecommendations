#helper functions
def accuracy(preds):
    correct = 0
    for pred in preds:
        p = 1 if pred.est > 0.5 else 0
        if p == pred.r_ui:
            correct += 1
    return correct / len(preds)


def ranking(user_id):
    recs = []
    rated = df.loc[df["user_id"] == user_id, "item_id"].unique()
    print(rated)
    for idx in range(5000):
        if idx in rated:
            continue
        p = mf.predict(uid=user_id, iid=idx)
        recs.append((idx, p.est))
    return recs