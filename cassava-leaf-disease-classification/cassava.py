# import glob
import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

print(tf.__version__)

DEVMODE = os.getenv("KAGGLE_MODE") == "DEV"
print(f"DEV MODE: {DEVMODE}")

INPUT_FOLDER = "/kaggle/input/cassava-leaf-disease-classification/"
WORK_DIR = "/kaggle/working"

SEED = hash("cassava-leaf-disease-classification") % 100

DATASET_SIZE = 21367
VALID_DATASET_RATIO = 0.05
BATCH_SIZE = 16
IMAGE_SIZE = (512, 512)
N_CLASSES = 5

EPOCHS = 1 if DEVMODE else 17


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


def _parse_tfrecord(example_proto):
    feature_descriptions = {
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }
    example = tf.io.parse_single_example(example_proto, feature_descriptions)

    image = tf.io.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, example["target"], example["image_name"]


def get_train_val_datasets():
    train_files = tf.data.Dataset.list_files(os.path.join(INPUT_FOLDER, "train_tfrecords/ld_train*.tfrec"))

    dataset = tf.data.TFRecordDataset(train_files, num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE).map(
        lambda x, y, _: (x, y), num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.shuffle(128)

    valid_dataset_size = int(DATASET_SIZE * VALID_DATASET_RATIO)

    valid_dataset = dataset.take(valid_dataset_size).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    train_dataset = dataset.skip(valid_dataset_size).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_dataset, valid_dataset


def get_train_val_generators():
    train_data = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
    train_data.label = train_data.label.astype("str")
    print(train_data.head())

    train_datagen = ImageDataGenerator(
        # rescale=1 / 255.0,
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


def create_model():
    model = Sequential(
        [
            Input(shape=IMAGE_SIZE + (3,)),
            Conv2D(filters=32, kernel_size=5, activation="relu"),
            MaxPooling2D((3, 3)),
            Conv2D(64, 5, activation="relu"),
            MaxPooling2D((3, 3)),
            Conv2D(128, 5, activation="relu"),
            MaxPooling2D((3, 3)),
            Flatten(),
            Dense(1024, activation="relu"),
            Dropout(0.2),
            Dense(N_CLASSES, activation="softmax"),
        ]
    )

    return model


def build_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=metrics)

    return model


def create_inceptionv3():
    backbone = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=IMAGE_SIZE + (3,))

    # Freeze the pre-trained backbone model.
    # for layer in backbone.layers:
    #     layer.trainable = False

    model = Sequential(
        [
            backbone,
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            Dropout(0.2),
            Dense(N_CLASSES, activation="softmax"),
        ]
    )

    return model


def train_model(model: tf.keras.models.Model, train_dataset, valid_dataset):
    steps = 30 if DEVMODE else (train_dataset.n // train_dataset.batch_size)
    valid_steps = 10 if DEVMODE else (valid_dataset.n // valid_dataset.batch_size)

    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)
    checkpoint_cb = ModelCheckpoint(os.path.join(WORK_DIR, "cassava_best.h5"), monitor="val_loss", save_best_only=True)
    reduce_lr_on_plateau_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1)

    results = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps,
        validation_data=valid_dataset,
        validation_steps=valid_steps,
        shuffle=False,  # Datasets are already shuffled
        callbacks=[early_stopping_cb, checkpoint_cb, reduce_lr_on_plateau_cb],
    )

    print(f"Train accuracy: {results.history['accuracy']}")
    print(f"Validation accuracy: {results.history['val_accuracy']}")

    model.save(os.path.join(WORK_DIR, "cassava.h5"))

    return results


if __name__ == "__main__":
    seed_everything()

    disease_map = get_disease_map()

    model = create_inceptionv3()
    print(model.summary())

    model = build_model(model)

    # train_dataset, valid_dataset = get_train_val_datasets()
    train_dataset, valid_dataset = get_train_val_generators()

    results = train_model(model, train_dataset, valid_dataset)
