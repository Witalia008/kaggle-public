# import glob
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow_addons.callbacks import TimeStopping

print(tf.__version__)

DEVMODE = os.getenv("KAGGLE_MODE") == "DEV"
print(f"DEV MODE: {DEVMODE}")

if not DEVMODE:
    shutil.copyfile("/kaggle/input/googlebitemperedloss/bi_tempered_loss.py", "bi_tempered_loss.py")
from bi_tempered_loss import bi_tempered_logistic_loss

INPUT_FOLDER = "/kaggle/input/cassava-leaf-disease-classification/"
WORK_DIR = "/kaggle/working"

SEED = hash("cassava-leaf-disease-classification") % 100

DATASET_SIZE = 21367
VALID_DATASET_RATIO = 0.05
BATCH_SIZE = 16
IMAGE_SIZE = (512, 512)
N_CLASSES = 5

EPOCHS = 2 if DEVMODE else 17

T1 = 0.2
T2 = 1.2


def seed_everything(seed=SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def get_disease_map():
    with open(os.path.join(INPUT_FOLDER, "label_num_to_disease_map.json")) as f:
        disease_map: dict = json.load(f)

    disease_map = {int(disease_id): disease_name for (disease_id, disease_name) in disease_map.items()}
    assert len(disease_map) == N_CLASSES

    print(disease_map)
    return disease_map


def get_train_val_generators():
    train_data = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
    train_data.label = train_data.label.astype("str")
    print(train_data.head())

    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_v3.preprocess_input,
        validation_split=VALID_DATASET_RATIO,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,
        fill_mode="nearest",
    )

    train_dataset = train_datagen.flow_from_dataframe(
        train_data,
        subset="training",
        directory=os.path.join(INPUT_FOLDER, "train_images"),
        x_col="image_id",
        y_col="label",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        seed=SEED,
    )

    valid_dataset = train_datagen.flow_from_dataframe(
        train_data,
        subset="validation",
        directory=os.path.join(INPUT_FOLDER, "train_images"),
        x_col="image_id",
        y_col="label",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=False,
        seed=SEED,
    )

    return train_dataset, valid_dataset


def build_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def bi_tempered_loss(y_true, y_pred):
        y_true = K.cast(K.reshape(y_true, (-1,)), "int64")
        labels = K.one_hot(y_true, N_CLASSES)
        return bi_tempered_logistic_loss(y_pred, labels, T1, T2, label_smoothing=0.2)

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(loss=bi_tempered_loss, optimizer=optimizer, metrics=metrics)

    return model


def create_inceptionv3():
    backbone = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=IMAGE_SIZE + (3,))

    model = Sequential(
        [
            backbone,
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            Dropout(0.2),
            Dense(N_CLASSES, activation=None),  # Softmax is embedded in bi-tempered loss.
        ]
    )

    return model


def train_model(model: tf.keras.models.Model, train_dataset, valid_dataset):
    steps = 10 if DEVMODE else (train_dataset.n // train_dataset.batch_size)
    valid_steps = 5 if DEVMODE else (valid_dataset.n // valid_dataset.batch_size)

    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)
    checkpoint_cb = ModelCheckpoint(os.path.join(WORK_DIR, "cassava_best.h5"), monitor="val_loss", save_best_only=True)
    reduce_lr_on_plateau_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1)
    time_stopping_cb = TimeStopping(seconds=int(11.5 * 60 * 60), verbose=1)

    results = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps,
        validation_data=valid_dataset,
        validation_steps=valid_steps,
        shuffle=False,  # Datasets are already shuffled
        callbacks=[early_stopping_cb, checkpoint_cb, reduce_lr_on_plateau_cb, time_stopping_cb],
    )

    print(f"Train accuracy: {results.history['accuracy']}")
    print(f"Validation accuracy: {results.history['val_accuracy']}")

    model.save(os.path.join(WORK_DIR, "cassava.h5"))

    return results


def main():
    seed_everything()

    model = create_inceptionv3()
    print(model.summary())

    model = build_model(model)

    train_dataset, valid_dataset = get_train_val_generators()

    results = train_model(model, train_dataset, valid_dataset)

    return results, valid_dataset


if __name__ == "__main__":
    main()
