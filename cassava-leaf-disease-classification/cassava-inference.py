import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model

DEVMODE = os.getenv("KAGGLE_MODE") == "DEV"
print(f"DEV MODE: {DEVMODE}")

INPUT_FOLDER = "/kaggle/input/cassava-leaf-disease-classification/"
WORK_DIR = "/kaggle/working"

IMAGE_SIZE = (512, 512)


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
    predictions = np.argmax(predictions, axis=-1)

    submission_file = os.path.join(WORK_DIR, "submission.csv")
    pd.DataFrame({"image_id": image_names, "label": predictions}).to_csv(submission_file, index=False)
    print(pd.read_csv(submission_file).head())


def run_predictions(model):
    predictions = []
    test_image_names = []

    test_dir = os.path.join(INPUT_FOLDER, "test_images")

    for image_name in os.listdir(test_dir):
        image = tf.keras.preprocessing.image.load_img(os.path.join(test_dir, image_name), target_size=IMAGE_SIZE)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.inception_v3.preprocess_input(image)

        cur_prediction = model.predict(image)

        predictions.append(cur_prediction[0])
        test_image_names.append(image_name)

    return predictions, test_image_names


if __name__ == "__main__":
    model_location = "/kaggle/input/cassava-model"
    model = load_model(
        os.path.join(model_location, "cassava_best_tempered.h5"),
        custom_objects={
            "bi_tempered_loss": sparse_categorical_crossentropy
        },  # Loss is not used for inference, so assign to whatever.
    )

    # test_dataset, test_image_names = get_test_dataset()
    # test_dataset, test_image_names = get_test_generator()

    # test_dataset.reset()
    # predictions = model.predict(test_dataset)

    predictions, test_image_names = run_predictions(model)
    store_predictions(predictions, test_image_names)
