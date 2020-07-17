# %% [markdown]
# # Tune hyperparameters for the DL model

# %%
import os

# %env PYTHONHASHSEED=0
os.environ["PYTHONHASHSEED"] = "0"

DEVMODE = os.getenv("KAGGLE_MODE") == "DEV"
print(f"DEV MODE: {DEVMODE}")

# Define seed for reproducibility of random generation
SEED = 42
DEV_SPLIT = 0.2


# %%
import numpy as np
import pandas as pd

# To display all the columns from left to right without breaking into next line.
pd.set_option("display.width", 1500)
pd.plotting.register_matplotlib_converters()

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)


# %%
import random as python_random

# Make sure Keras produces reproducible results.

np.random.seed(SEED)
python_random.seed(SEED)
tf.random.set_seed(SEED)


# %%
physical_devices = tf.config.experimental.list_physical_devices("GPU")
print(physical_devices)
for device in physical_devices or []:
    tf.config.experimental.set_memory_growth(device, True)

# %% [markdown]
# ## Load data and split into train/dev sets

# %%
from titanic.titanic_data import load_titanic_data, split_data, get_data_preprocessor

X_train_full, y_train_full, X_pred = load_titanic_data()
X_train, X_valid, y_train, y_valid = split_data(X_train_full, y_train_full, test_size=DEV_SPLIT, random_state=SEED)

# %% [markdown]
# ## Define pre-processing of the data

# %%
preprocessor, preprocessed_column_names = get_data_preprocessor()


# %%
X_train = pd.DataFrame(preprocessor.fit_transform(X_train), index=X_train.index, columns=preprocessed_column_names)
X_valid = pd.DataFrame(preprocessor.transform(X_valid), index=X_valid.index, columns=preprocessed_column_names)
X_pred = pd.DataFrame(preprocessor.transform(X_pred), index=X_pred.index, columns=preprocessed_column_names)

# %% [markdown]
# # DL model using Keras

# %%
METRICS = [
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
]


# %%
from titanic.titanic_data import get_class_weights

class_weights = get_class_weights(y_train)


# %%
import kerastuner as kt

# https://www.sicara.ai/blog/hyperparameter-tuning-keras-tuner
# https://www.curiousily.com/posts/hackers-guide-to-hyperparameter-tuning/


class TitanicHyperModel(kt.HyperModel):
    def __init__(self, input_size):
        self.input_shape = (input_size,)

    def build(self, hp):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, InputLayer
        from tensorflow.keras.regularizers import L1L2

        model = Sequential()
        model.add(InputLayer(input_shape=self.input_shape))

        for layer_n in range(hp.Int("num_layers", min_value=2, max_value=7, step=1, default=3)):
            units = hp.Int(f"dense_units_{layer_n}", min_value=8, max_value=64, step=8, default=64)
            activation = hp.Choice(f"dense_activation_{layer_n}", values=["relu", "tanh"], default="relu")
            regularizer_l1 = hp.Choice(f"l1_{layer_n}", values=[0.01, 0.001, 1e-4, 1e-5, 0.0], default=1e-2)
            regularizer_l2 = hp.Choice(f"l2_{layer_n}", values=[0.1, 0.01, 0.001], default=1e-2)
            initializer = "he_uniform" if activation == "relu" else "glorot_uniform"

            model.add(
                Dense(
                    units=units,
                    activation=activation,
                    kernel_regularizer=L1L2(l1=regularizer_l1, l2=regularizer_l2),
                    bias_regularizer=L1L2(l1=regularizer_l1, l2=regularizer_l2),
                    kernel_initializer=initializer,
                )
            )

            droupout_rate = hp.Float(f"dropout_{layer_n}", min_value=0.15, max_value=0.5, default=0.25, step=0.05,)

            model.add(Dropout(rate=droupout_rate))

        model.add(Dense(1, activation="sigmoid"))

        learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3], default=1e-3)  # , 5e-4, 1e-4],

        optimizer = hp.Choice("optimizer", values=["adam", "RMSprop"], default="adam")
        optimizer_type = {
            "adam": keras.optimizers.Adam,
            "RMSprop": keras.optimizers.RMSprop,
            "SGD": keras.optimizers.SGD,
        }

        model.compile(
            optimizer=optimizer_type[optimizer](learning_rate=learning_rate),
            metrics=METRICS,
            loss="binary_crossentropy",
        )

        return model


hypermodel = TitanicHyperModel(input_size=X_train.shape[1])


# %%
class MyTuner(kt.Tuner):
    def run_trial(self, trial, X_train, Y_train, validation_data):
        hp = trial.hyperparameters

        batch_size = hp.Int("batch_size", 32, 128, step=32, default=32)
        epoch_number = hp.Int("epoch_number", 400, 700, step=100, default=500)

        model = self.hypermodel.build(hp)

        history = model.fit(
            X_train,
            Y_train,
            epochs=epoch_number,
            batch_size=batch_size,
            validation_data=validation_data,
            class_weight=class_weights,
            verbose=1,
        )

        self.oracle.update_trial(trial.trial_id, {"val_accuracy": history.history["val_accuracy"][-1]})
        self.save_model(trial.trial_id, model)


# %%
DEVMODE = False


# %%
if DEVMODE:
    MAX_TRIALS = 5
else:
    MAX_TRIALS = 50


# %%
import time

hp = kt.HyperParameters()
if DEVMODE:
    hp.Fixed("epoch_number", 50)

# Populate some options that after tuning seem to always be the best.
hp.Fixed("batch_size", 32)
hp.Fixed("optimizer", "adam")
hp.Fixed("learning_rate", 1e-3)
hp.Fixed("epoch_number", 500)

tuner = MyTuner(
    oracle=kt.oracles.RandomSearch(
        objective="val_accuracy", seed=SEED, hyperparameters=hp, tune_new_entries=True, max_trials=MAX_TRIALS
    ),
    hypermodel=hypermodel,
    directory=f"/kaggle/tmp/hpsearch/{time.time()}",
    project_name="titanic",
)

if DEVMODE:
    tuner.search_space_summary()


# %%
tuner.search(X_train, y_train, validation_data=(X_valid, y_valid))

if DEVMODE:
    tuner.results_summary()

# %% [markdown]
# # Results of the tuning

# %%
NUM_TOP_MODELS = min(5, MAX_TRIALS)

top_models = tuner.get_best_models(num_models=NUM_TOP_MODELS)
top_hyperparameters = tuner.get_best_hyperparameters(num_trials=NUM_TOP_MODELS)

for exp_id in range(NUM_TOP_MODELS):
    print(f"\n\nTuned model {exp_id}\n\n")
    cur_model = top_models[exp_id]

    print("Tuned train:")
    evaluation_tuned_train = cur_model.evaluate(X_train, y_train)
    # draw_confusion_matrix(evaluation_tuned_train, f"tuned train {exp_id}")

    print("Tuned dev:")
    evaluation_tuned_dev = cur_model.evaluate(X_valid, y_valid)
    # draw_confusion_matrix(evaluation_tuned_dev, f"tuned dev {exp_id}")

    print("Model summary")
    cur_model.summary()

    print("Hyperparameters")
    print(top_hyperparameters[exp_id].values)

# %% [markdown]
# # Predict with the Tuned model(s)

# %%
from utils.predicting import store_predictions

for exp_id in range(NUM_TOP_MODELS):
    store_predictions(top_models[exp_id], X_pred, index=X_pred.index, submission_name=f"dl_tuned_{exp_id}")
