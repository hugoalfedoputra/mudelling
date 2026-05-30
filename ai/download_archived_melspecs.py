import os
import time
import traceback
from dotenv import load_dotenv
from rclone_python import rclone
from dataclasses import dataclass

# import json

# ====================================================================================================
# Globals
#
# READ AND RECONFIGURE THE GLOBALS FIRST BEFORE SHIPPING CODE
# READ AND RECONFIGURE THE GLOBALS FIRST BEFORE SHIPPING CODE
# READ AND RECONFIGURE THE GLOBALS FIRST BEFORE SHIPPING CODE
# READ AND RECONFIGURE THE GLOBALS FIRST BEFORE SHIPPING CODE
# READ AND RECONFIGURE THE GLOBALS FIRST BEFORE SHIPPING CODE
# ====================================================================================================

# This is from ../prep/preprocess.py; hard-coded so as to not cause import issues during build
MELSPEC_PREFIXES = ["5_", "15_", "30_"]
MELSPEC_DIR_PREFIX = "melspec"
LOCAL_TEMP_FOLDER = "./temp"
LOCAL_MANIFEST_FOLDER = "manifest"
LOCAL_ARCHIVE_FOLDER = "archive"

RCLONE_ARGS = [
    "--retries",
    "10",
    "--low-level-retries",
    "10",
    "--contimeout",
    "120s",
    "--timeout",
    "300s",
    "--retries-sleep",
    "10s",
    "-vv",
]


@dataclass
class DownloadConfig:
    melspec_remote: str
    melspec_storage: str
    melspec_chunk_length: str
    start_subfolder: int
    end_subfolder: int
    split: str


def load_config() -> DownloadConfig:
    load_dotenv()
    try:
        melspec_remote = os.environ["RCLONE_REMOTE"]
        melspec_storage = os.environ["REMOTE_MELSPEC_FOLDER"]
        start_subfolder = int(os.environ["START_SUBFOLDER"])
        end_subfolder = int(os.environ["END_SUBFOLDER"])
        melspec_chunk_length = MELSPEC_PREFIXES[int(os.environ["MELSPEC_PREFIX_INDEX"])]

        # Determine the split to process (can also be driven by sys.argv)
        split = os.environ.get("SPLIT_TO_DOWNLOAD", "test")

    except KeyError as e:
        traceback.print_exc()
        raise Exception(f"Missing env var: {e.args[0]}")

    return DownloadConfig(
        melspec_remote=melspec_remote,
        melspec_storage=melspec_storage,
        melspec_chunk_length=melspec_chunk_length,
        start_subfolder=start_subfolder,
        end_subfolder=end_subfolder,
        split=split,
    )


def sys_prepare(split: str):
    archive_path = f"{LOCAL_TEMP_FOLDER}/{LOCAL_ARCHIVE_FOLDER}/{split}"
    manifest_path = f"{LOCAL_TEMP_FOLDER}/{LOCAL_MANIFEST_FOLDER}/{split}"

    os.makedirs(archive_path, exist_ok=True)
    os.makedirs(manifest_path, exist_ok=True)
    return archive_path, manifest_path


def download_archives(config: DownloadConfig):
    archive_path, manifest_out_path = sys_prepare(config.split)

    for i in range(config.start_subfolder, config.end_subfolder, 1):
        subfolder = str(i).zfill(2)

        # Paths
        base_name = f"{config.melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}"
        manifest_remote = f"{config.melspec_remote}:{config.melspec_storage}/{config.split}/{base_name}_manifest.json"
        tar_remote = f"{config.melspec_remote}:{config.melspec_storage}/{config.split}/{base_name}.tar"

        # Check manifest first
        rclone.copy(
            in_path=manifest_remote,
            out_path=manifest_out_path,
            ignore_existing=True,
            show_progress=False,
            args=RCLONE_ARGS,
        )

        manifest_local = f"{manifest_out_path}/{base_name}_manifest.json"
        if not os.path.exists(manifest_local):
            print(f"Manifest for {subfolder} missing on remote. Skipping.")
            continue

        print(f"Downloading archive {tar_remote} to {archive_path}...")
        start_time = time.time()
        rclone.copy(
            in_path=tar_remote,
            out_path=archive_path,
            ignore_existing=True,  # Skips if .tar is already fully downloaded
            show_progress=True,
            args=RCLONE_ARGS,
        )
        print(f"Finished downloading {subfolder} in {time.time() - start_time:.2f}s.\n")


if __name__ == "__main__":
    config = load_config()
    print(f"Starting manual archive download for split: {config.split}")
    download_archives(config)
