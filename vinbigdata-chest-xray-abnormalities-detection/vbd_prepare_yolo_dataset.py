import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

DEVMODE = os.getenv("KAGGLE_MODE") == "DEV"
print(f"DEV MODE: {DEVMODE}")

INPUT_FOLDER = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
INPUT_FOLDER_PNG = "/kaggle/input/vinbigdata"
WORK_FOLDER = "/kaggle/working"


def convert_to_yolo(bbox_df):
    bbox_df["x_centre"] = (bbox_df["x_min"] + bbox_df["x_max"]) / 2
    bbox_df["y_centre"] = (bbox_df["y_min"] + bbox_df["y_max"]) / 2
    bbox_df["bw"] = bbox_df["x_max"] - bbox_df["x_min"]
    bbox_df["bh"] = bbox_df["y_max"] - bbox_df["y_min"]
    bbox_df[["x_centre", "bw"]] = bbox_df[["x_centre", "bw"]].div(bbox_df["dim0"], axis=0)
    bbox_df[["y_centre", "bh"]] = bbox_df[["y_centre", "bh"]].div(bbox_df["dim1"], axis=0)
    return bbox_df


def reduce_bboxes_random_rad(image_labels):
    random_rad = np.random.choice(image_labels["rad_id"].unique())
    return image_labels[image_labels["rad_id"] == random_rad]


def get_yolo_labels_txt(image_labels):
    # For now, select only labels from one random radiologist out of 3.
    image_labels = reduce_bboxes_random_rad(image_labels)
    return image_labels[["class_id", "x_centre", "y_centre", "bw", "bh"]].to_string(header=False, index=False)


def main():
    np.random.seed(0)

    train_data = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
    train_meta = pd.read_csv(os.path.join(INPUT_FOLDER_PNG, "train_meta.csv"))
    train_data = pd.merge(train_data, train_meta, left_on="image_id", right_on="image_id")
    train_data = convert_to_yolo(train_data)

    output_folder = os.path.join(WORK_FOLDER, "vbdyolo")
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder)
    os.makedirs(os.path.join(output_folder, "images", "train"))
    os.makedirs(os.path.join(output_folder, "labels", "train"))

    for image_id, image_grouped_labels in tqdm(train_data.groupby("image_id")):
        if image_grouped_labels["class_name"].iloc[0] == "No finding":
            # Only copy abnormal images (for now at least).
            continue

        image_file_name = f"{image_id}.png"
        shutil.copyfile(
            os.path.join(INPUT_FOLDER_PNG, "train", image_file_name),
            os.path.join(output_folder, "images", "train", image_file_name),
        )
        with open(os.path.join(output_folder, "labels", "train", f"{image_id}.txt"), "w") as f:
            f.write(get_yolo_labels_txt(image_grouped_labels))

    # Just copy all the test files as well.
    shutil.copytree(os.path.join(INPUT_FOLDER_PNG, "test"), os.path.join(output_folder, "images", "test"))


if __name__ == "__main__":
    main()
