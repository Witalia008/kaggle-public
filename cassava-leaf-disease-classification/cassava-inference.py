import gc
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Average, Input
from tensorflow.keras.models import load_model

DEVMODE = os.getenv("KAGGLE_MODE") == "DEV"
print(f"DEV MODE: {DEVMODE}")

INPUT_FOLDER = "/kaggle/input/cassava-leaf-disease-classification/"
WORK_DIR = "/kaggle/working"

IMAGE_SIZE = (512, 512)

N_FOLD = 3


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
            custom_objects={"bi_tempered_loss": None},  # Loss is not used for inference, so assign to whatever.
        )
        for model_name in models_info
    ]

    model = define_ensemble_model(models)
    print(model.summary())

    predictions, test_image_names = run_predictions(model)
    store_predictions(predictions, test_image_names)
