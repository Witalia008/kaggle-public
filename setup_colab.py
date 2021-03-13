import json
from pathlib import Path
import shutil

INPUT_FOLDER = Path("/kaggle/input/")
OUTPUT_FOLDER = Path("/kaggle/output/")
WORK_FOLDER = Path("/kaggle/working/")


def dump_dataset_metadata(user_name, dataset_name, folder_path):
    with open(Path(folder_path) / "dataset-metadata.json", "w") as f:
        json.dump({
            "title": dataset_name,
            "id": f"{user_name}/{dataset_name}",
            "licenses": [{ "name": "CC0-1.0" }]
        }, f, indent=4)

def is_running_in_colab(check_env=True):
    if not check_env:
        return True
    running_in_colab = "google.colab" in str(get_ipython())
    print(f"Running in Colab: {running_in_colab}")
    return running_in_colab


def setup_colab_drive_for_kaggle(check_env=True):
    if not is_running_in_colab(check_env):
        return False

    from google.colab import drive
    drive.mount("/content/drive")

    return True  # Is Colab


def setup_colab_secrets_for_kaggle(check_env=True):
    if not is_running_in_colab(check_env):
        return False

    drive_sources_dir = Path("/content/drive/MyDrive/Colab Notebooks/kaggle")

    # Set up kaggle.json to access Kaggle data.
    if (drive_sources_dir / "kaggle.json").exists():
        kaggle_config = Path.home() / ".kaggle"
        if kaggle_config.exists():
            shutil.rmtree(kaggle_config)
        kaggle_config.mkdir()
        (kaggle_config / "kaggle.json").symlink_to(drive_sources_dir / "kaggle.json")
        print(f"Content of Kaggle config dir ({kaggle_config}): {list(map(str, kaggle_config.iterdir()))}")

    return True  # Is Colab


def setup_colab_directories_for_kaggle(check_env=True, local_working=False):
    if not is_running_in_colab(check_env):
        return False

    # Only add "working" directory if it was requested to be mapped in Drive, not in local env.
    target_content_dirs = ["input", "output"] + ([] if local_working else ["working"])

    drive_content_dir = Path("/content/drive/MyDrive/kaggle")
    # Make sure directories are present in Drive
    drive_content_dir.mkdir(exist_ok=True)
    for content_dir in target_content_dirs:
        (drive_content_dir / content_dir).mkdir(exist_ok=True)
    print(f"Content of Drive Kaggle data dir ({drive_content_dir}): {list(map(str, drive_content_dir.iterdir()))}")

    kaggle_dir = Path("/kaggle")
    if kaggle_dir.exists():
        shutil.rmtree(kaggle_dir)
    kaggle_dir.mkdir()

    for content_dir in target_content_dirs:
        (kaggle_dir / content_dir).symlink_to(drive_content_dir / content_dir)

    # It was requested not to map working to Drive, so create it locally.
    if local_working:
        (kaggle_dir / "working").mkdir()

    print(f"Content of Kaggle data dir ({kaggle_dir}): {list(map(str, kaggle_dir.iterdir()))}")
    for content_dir in target_content_dirs + (["working"] if local_working else []):
        print(f"Content of Kaggle data subdir ({kaggle_dir / content_dir}): {list(map(str, (kaggle_dir / content_dir).iterdir()))}")

    return True  # Is Colab


def setup_colab_for_kaggle(check_env=True, local_working=False):
    if not is_running_in_colab(check_env):
        return False

    setup_colab_drive_for_kaggle(check_env=False)
    setup_colab_directories_for_kaggle(check_env=False, local_working=local_working)
    setup_colab_secrets_for_kaggle(check_env=False)

    return True  # Is Colab