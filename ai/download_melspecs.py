import os
import traceback
import time
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


def sys_prepare(subfolder, melspec_chunk_length):
    if subfolder is None:
        raise Exception("Subfolder is required.")

    dir_path = (
        f"{LOCAL_TEMP_FOLDER}/{melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}"
    )

    if not os.path.exists(LOCAL_TEMP_FOLDER):
        os.mkdir(LOCAL_TEMP_FOLDER)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def is_local_complete(in_path, out_path, subfolder, melspec_chunk_length):
    remote = int(rclone.size(in_path)["count"])

    local = int(rclone.size(out_path)["count"])

    if remote == local:
        print(
            f"Subfolder {subfolder} for {melspec_chunk_length} already has all melspecs locally. Skipped downloading melspecs for this subfolder..."
        )
        return True
    else:
        return False


def download_remote_melspecs(
    melspec_remote, melspec_storage, subfolder, melspec_chunk_length
):
    # Remote
    in_path = f"{melspec_remote}:{melspec_storage}/{melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}"

    # Local
    out_path = (
        f"{LOCAL_TEMP_FOLDER}/{melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}"
    )

    if is_local_complete(in_path, out_path, subfolder, melspec_chunk_length):
        return

    print(f"Copying {in_path} from remote to {out_path}...")
    start_time = time.time()
    rclone.copy(
        in_path=in_path,
        out_path=out_path,
        ignore_existing=False,
        show_progress=False,
        args=RCLONE_ARGS,
    )
    elapsed_time = time.time() - start_time
    print(f"Finished copying from remote in {elapsed_time}s. Continuing...\n")


def main():
    (
        melspec_remote,
        melspec_storage,
        melspec_chunk_length,
        start_subfolder,
        end_subfolder,
    ) = load_config()

    for i in range(start_subfolder, end_subfolder, 1):
        subfolder = str(i).zfill(2)

        sys_prepare(subfolder, melspec_chunk_length)

        download_remote_melspecs(
            melspec_remote=melspec_remote,
            melspec_storage=melspec_storage,
            subfolder=subfolder,
            melspec_chunk_length=melspec_chunk_length,
        )


if __name__ == "__main__":
    main()
