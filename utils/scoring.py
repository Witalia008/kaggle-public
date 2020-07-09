from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def score_model(model, X, y):
    y_pred = model.predict(X)
    score = accuracy_score(y, y_pred)
    print("Accuracy:", score)
    return score


def score_cross_val(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print("Scores:", scores)
    score = scores.mean()
    print("Avg score:", score)

    return score
