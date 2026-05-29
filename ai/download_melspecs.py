import os
import traceback
import time
import tarfile
import json
from dotenv import load_dotenv
from rclone_python import rclone

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
LOCAL_RAW_FOLDER = "raw"

# These are actually default as per: https://rclone.org/commands/rclone/
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
    # "--dry-run",
]


def load_config():
    """Load configuration from the .env file."""
    load_dotenv()

    # If one or more keys are empty, raise an error
    if not all(
        [
            os.environ["RCLONE_REMOTE"] != "",
            os.environ["REMOTE_MELSPEC_FOLDER"] != "",
            os.environ["START_SUBFOLDER"] != "",
            os.environ["END_SUBFOLDER"] != "",
            os.environ["MELSPEC_PREFIX_INDEX"] != "",
        ]
    ):
        raise Exception("One or more variables in the .env file is empty.")

    try:
        melspec_remote = os.environ["RCLONE_REMOTE"]
        melspec_storage = os.environ["REMOTE_MELSPEC_FOLDER"]
        start_subfolder = int(os.environ["START_SUBFOLDER"])
        end_subfolder = int(os.environ["END_SUBFOLDER"])
        melspec_chunk_length = MELSPEC_PREFIXES[int(os.environ["MELSPEC_PREFIX_INDEX"])]
    except KeyError as e:
        traceback.print_exc()
        raise Exception(f"Missing env var: {e.args[0]}")

    return (
        melspec_remote,
        melspec_storage,
        melspec_chunk_length,
        start_subfolder,
        end_subfolder,
    )


def sys_prepare(subfolder, melspec_chunk_length, split):
    if subfolder is None:
        raise Exception("Subfolder is required.")

    archive_path = f"{LOCAL_TEMP_FOLDER}/{LOCAL_ARCHIVE_FOLDER}/{split}"

    raw_path = f"{LOCAL_TEMP_FOLDER}/{LOCAL_RAW_FOLDER}/{split}/{melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}"

    if not os.path.exists(LOCAL_TEMP_FOLDER):
        os.mkdir(LOCAL_TEMP_FOLDER)

    if not os.path.exists(archive_path):
        os.makedirs(archive_path)

    if not os.path.exists(raw_path):
        os.makedirs(raw_path)


def is_local_complete(
    manifest_path, manifest_out_path, local_path, subfolder, melspec_chunk_length
):
    rclone.copy(
        in_path=manifest_path,
        out_path=manifest_out_path,
        ignore_existing=True,
        show_progress=False,
        args=RCLONE_ARGS,
    )

    with open(
        f"{manifest_out_path}/{melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}_manifest.json",
        "r",
    ) as f:
        content = json.loads(f.read())

        local = len(os.listdir(local_path))

        if int(content["total_files_in_tar"]) == local:
            print(
                f"Subfolder {subfolder} for {melspec_chunk_length} already has all melspecs locally. Skipped downloading melspecs for this subfolder..."
            )
            return True
        else:
            print(
                f"Subfolder {subfolder} for {melspec_chunk_length} is incomplete (remote: {int(content["total_files_in_tar"])} vs. local: {local}). Downloading..."
            )
            return False


def download_remote_melspecs(
    melspec_remote, melspec_storage, subfolder, melspec_chunk_length, split
):
    # Remote
    manifest_path = f"{melspec_remote}:{melspec_storage}/{split}/{melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}_manifest.json"
    manifest_out_path = f"{LOCAL_TEMP_FOLDER}/{LOCAL_MANIFEST_FOLDER}/{split}"

    in_path = f"{melspec_remote}:{melspec_storage}/{split}/{melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}.tar"

    # Local
    out_path = f"{LOCAL_TEMP_FOLDER}/{LOCAL_RAW_FOLDER}/{split}/{melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}"
    tar_path = f"{LOCAL_TEMP_FOLDER}/{LOCAL_ARCHIVE_FOLDER}/{split}"

    if is_local_complete(
        manifest_path, manifest_out_path, out_path, subfolder, melspec_chunk_length
    ):
        # To answer the question: was something downloaded?
        return False

    print(f"Copying {in_path} from remote to {tar_path}...")
    start_time = time.time()
    rclone.copy(
        in_path=in_path,
        out_path=tar_path,
        ignore_existing=False,
        show_progress=False,
        args=RCLONE_ARGS,
    )
    elapsed_time = time.time() - start_time
    print(f"Finished copying from remote in {elapsed_time}s. Continuing...\n")

    return True


def extract_remote_melspecs(subfolder, melspec_chunk_length, split):
    tar_loc = f"{LOCAL_TEMP_FOLDER}/{LOCAL_ARCHIVE_FOLDER}/{split}/{melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}.tar"
    raw_loc = f"{LOCAL_TEMP_FOLDER}/{LOCAL_RAW_FOLDER}/{split}/{melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}"

    with tarfile.open(tar_loc) as t:
        print(f"Extracting remote tar at {tar_loc} to {raw_loc}")
        t.extractall(path=raw_loc)

    try:
        os.remove(tar_loc)
        print(f"Removed original remote tar at {tar_loc}")
    except Exception:
        traceback.print_exc()
        print(f"Failed to remove original remote tar at {tar_loc}")


def main(split: str = "train"):
    (
        melspec_remote,
        melspec_storage,
        melspec_chunk_length,
        start_subfolder,
        end_subfolder,
    ) = load_config()

    # This is just to download training split. Val and test splits can be downloaded
    # in the training script instead because it is "more on-demand" than this.
    for i in range(start_subfolder, end_subfolder, 1):
        subfolder = str(i).zfill(2)

        sys_prepare(
            subfolder=subfolder,
            melspec_chunk_length=melspec_chunk_length,
            split=split,
        )

        was_something_downloaded = download_remote_melspecs(
            melspec_remote=melspec_remote,
            melspec_storage=melspec_storage,
            subfolder=subfolder,
            melspec_chunk_length=melspec_chunk_length,
            split=split,
        )

        if was_something_downloaded:
            extract_remote_melspecs(
                subfolder=subfolder,
                melspec_chunk_length=melspec_chunk_length,
                split=split,
            )


if __name__ == "__main__":
    main()
