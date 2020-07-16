import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


# A custom transformer for age group-mean, so that mean is from train set always (the one fitted to).
class GroupbyMeanImputer(TransformerMixin):
    def __init__(self, group_by_labels, target_label):
        self.group_by_labels = group_by_labels
        self.target_label = target_label

    def fit(self, X, y=None):
        self.group_mean = self._get_grouped(X).mean()
        return self

    def transform(self, X):
        return self._get_grouped(X).transform(lambda x: x.fillna(self.group_mean[x.name])).to_numpy().reshape(-1, 1)

    def _get_grouped(self, X):
        return X.groupby(self.group_by_labels)[self.target_label]


def load_titanic_data():
    train_data = pd.read_csv("/kaggle/input/titanic/train.csv", index_col="PassengerId")
    test_data = pd.read_csv("/kaggle/input/titanic/test.csv", index_col="PassengerId")

    y_train = train_data["Survived"]
    X_train = train_data.drop("Survived", axis=1).copy()

    X_pred = test_data

    return (X_train, y_train, X_pred)


def split_data(X, y, test_size, random_state):
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_data_preprocessor(standardize=True):
    from sklearn.compose import ColumnTransformer

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler

    import re

    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    name_transformer = Pipeline(
        steps=[
            (
                "convert_to_title",
                FunctionTransformer(lambda df: df.applymap(lambda name: re.search(", ([\w ]+).", name).group(1))),
            ),
            (
                "rare_to_others",
                FunctionTransformer(
                    lambda df: df.applymap(
                        lambda title: title if title in ["Mr", "Mrs", "Miss", "Master"] else "Others"
                    )
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    fare_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")), ("log", FunctionTransformer(np.log1p))]
    )

    family_transformer = FunctionTransformer(lambda df: df.sum(axis=1).to_frame())

    remember_missing_transformer = FunctionTransformer(lambda df: df.apply(lambda col: np.where(col.isnull(), 1, 0)))

    age_transformer = Pipeline(
        steps=[
            ("imputer", GroupbyMeanImputer(group_by_labels="Pclass Sex".split(), target_label="Age")),
            (
                "bin",
                FunctionTransformer(
                    lambda df: pd.cut(df[:, 0], bins=[0, 16, 30, 50, 80], labels=False).reshape(-1, 1) + 1
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("keep", FunctionTransformer(), ["Pclass"]),
            ("onehot", categorical_transformer, "Sex Embarked".split()),
            ("name", name_transformer, ["Name"]),
            ("fare", fare_transformer, ["Fare"]),
            ("family", family_transformer, ["Parch", "SibSp"]),
            ("missing", remember_missing_transformer, ["Age"]),
            ("age", age_transformer, ["Age", "Pclass", "Sex"]),
        ]
    )

    if standardize:
        preprocessor = Pipeline(steps=[("transform", preprocessor), ("standardize", StandardScaler())])

    preprocessed_column_names = (
        ["Pclass"]
        + sorted("Female Male".split())
        + sorted("C Q S".split())
        + sorted("Master Miss Mr Mrs Others".split())
        + ["Fare", "Family"]
        + sorted("Age_Missing".split())
        + ["Age"]
    )

    return preprocessor, preprocessed_column_names
