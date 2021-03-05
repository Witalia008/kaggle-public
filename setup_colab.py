from pathlib import Path
import shutil
import importlib


def is_running_in_colab(check_env=True):
    if not check_env:
        return True
    running_in_colab = "google.colab" in str(get_ipython())
    print(f"Running in Colab: {running_in_colab}")
    return running_in_colab


def setup_colab_directories_for_kaggle(check_env=True):
    if not is_running_in_colab(check_env):
        return False

    target_content_dirs = ["input", "working"]

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
    
    print(f"Content of Kaggle data dir ({kaggle_dir}): {list(map(str, kaggle_dir.iterdir()))}")
    for content_dir in target_content_dirs:
        print(f"Content of Kaggle data subdir ({kaggle_dir / content_dir}): {list(map(str, (kaggle_dir / content_dir).iterdir()))}")
    
    return True  # Is Colab


def setup_colab_drive_for_kaggle(check_env=True):
    if not is_running_in_colab(check_env):
        return False
    
    from google.colab import drive
    drive.mount("/content/drive")

    return True  # Is Colab


def setup_colab_for_kaggle(check_env=True):
    if not is_running_in_colab(check_env):
        return False

    setup_colab_drive_for_kaggle(check_env=False)
    setup_colab_directories_for_kaggle(check_env=False)

    return True  # Is Colab