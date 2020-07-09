from sklearn.pipeline import Pipeline


def fit_model(model, preprocessor, X, y, **fit_kwargs):
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X, y, **fit_kwargs)
    return pipeline
