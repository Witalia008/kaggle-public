from pathlib import Path
import shutil
import importlib


def is_running_in_colab(check_env=True):
    if not check_env:
        return True
    running_in_colab = "google.colab" in str(get_ipython())
    print(f"Running in Colab: {running_in_colab}")
    return running_in_colab


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


def setup_colab_drive_for_kaggle(check_env=True):
    if not is_running_in_colab(check_env):
        return False
    
    from google.colab import drive
    drive.mount("/content/drive")

    return True  # Is Colab


def setup_colab_for_kaggle(check_env=True, local_working=False):
    if not is_running_in_colab(check_env):
        return False

    setup_colab_drive_for_kaggle(check_env=False)
    setup_colab_directories_for_kaggle(check_env=False, local_working=local_working)

    return True  # Is Colab