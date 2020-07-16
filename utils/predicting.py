import pandas as pd
import numpy as np


def store_predictions(model, X_pred, index, submission_name):
    predictions = model.predict(X_pred)

    predictions = np.round(predictions).astype(np.uint8).reshape((-1))

    print(f"{submission_name}:\n{predictions[:100]}...")

    output = pd.DataFrame({"Survived": predictions}, index=index)
    output.to_csv(f"/kaggle/working/{submission_name}_submission.csv", index=True)
