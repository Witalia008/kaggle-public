import gc
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Average, Input
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model

DEVMODE = os.getenv("KAGGLE_MODE") == "DEV"
print(f"DEV MODE: {DEVMODE}")

INPUT_FOLDER = "/kaggle/input/cassava-leaf-disease-classification/"
WORK_DIR = "/kaggle/working"

IMAGE_SIZE = (512, 512)

N_FOLD = 3


def _parse_tfrecord(example_proto):
    feature_descriptions = {
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }
    example = tf.io.parse_single_example(example_proto, feature_descriptions)

    image = tf.io.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, example["image_name"]


def get_test_dataset():
    test_files = tf.data.Dataset.list_files(os.path.join(INPUT_FOLDER, "test_tfrecords/*.tfrec"))

    dataset = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTOTUNE)

    dataset = dataset.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(16).prefetch(AUTOTUNE)

    image_names = [bytes.decode(example[-1].numpy()[0]) for example in dataset]

    return dataset, image_names


def get_test_generator():
    test_datagen = ImageDataGenerator(rescale=1 / 255.0)

    # Copy into a temp folder so that test_images would have a subfolder to make flow_from_directory happy.
    test_dir = os.path.join(INPUT_FOLDER, "train_images")
    new_test_dir = os.path.join(WORK_DIR, "cassava-test")
    if not os.path.exists(new_test_dir):
        os.makedirs(new_test_dir, exist_ok=True)
        shutil.copytree(test_dir, os.path.join(new_test_dir, "test_images"))

    test_dataset = test_datagen.flow_from_directory(
        directory=new_test_dir,
        target_size=IMAGE_SIZE,
        class_mode=None,  # No labels, just images.
        shuffle=False,  # To preserve the order for prediction.
        batch_size=1,  # To sample all the images exactly once.
    )

    image_names = [os.path.basename(image_name) for image_name in test_dataset.filenames]

    return test_dataset, image_names


def store_predictions(predictions, image_names):
    submission_file = os.path.join(WORK_DIR, "submission.csv")
    pd.DataFrame({"image_id": image_names, "label": predictions}).to_csv(submission_file, index=False)
    print(pd.read_csv(submission_file).head())


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def run_predictions(model: Model):
    predictions = []
    test_image_names = []

    test_dir = os.path.join(INPUT_FOLDER, "test_images")

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=45,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
    )

    for image_idx, image_name in enumerate(os.listdir(test_dir)):
        image = tf.keras.preprocessing.image.load_img(os.path.join(test_dir, image_name), target_size=IMAGE_SIZE)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)

        images = np.broadcast_to(image, (N_FOLD,) + image.shape[1:])
        # Use predict_on_batch since looks like predict has a memory leak...
        prediction = model.predict_on_batch(next(datagen.flow(images)))
        prediction = softmax(prediction)
        prediction = np.mean(prediction, axis=0, keepdims=True)

        predictions.append(int(np.argmax(prediction, axis=-1).squeeze()))
        test_image_names.append(image_name)

        if image_idx % 100 == 0:  # Only run every so many images.
            gc.collect()

    return predictions, test_image_names


def define_ensemble_model(models):
    # Wrap each model so that names are different and do not collide.
    models = [
        Model(inputs=model.input, outputs=model.output, name=f"{model.name}_{model_idx}")
        for model_idx, model in enumerate(models)
    ]

    input = Input(IMAGE_SIZE + (3,))

    ensemble_outputs = [model(input) for model in models]

    avg = Average()(ensemble_outputs)

    model = Model(inputs=input, outputs=avg)

    return model


if __name__ == "__main__":
    models_location = "/kaggle/input/cassava-model"
    models_info = ["cassava_best.h5", "cassava_best_tempered.h5", "cassava_best_tempered_2.h5"]

    models = [
        load_model(
            os.path.join(models_location, model_name),
            custom_objects={
                "bi_tempered_loss": sparse_categorical_crossentropy
            },  # Loss is not used for inference, so assign to whatever.
        )
        for model_name in models_info
    ]

    model = define_ensemble_model(models)
    print(model.summary())

    # test_dataset, test_image_names = get_test_dataset()
    # test_dataset, test_image_names = get_test_generator()

    # test_dataset.reset()
    # predictions = model.predict(test_dataset)

    predictions, test_image_names = run_predictions(model)
    store_predictions(predictions, test_image_names)
