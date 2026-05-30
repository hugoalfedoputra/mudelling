import os
import tarfile
import traceback
from dotenv import load_dotenv
from dataclasses import dataclass

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
LOCAL_ARCHIVE_FOLDER = "archive"
LOCAL_RAW_FOLDER = "raw"


@dataclass
class ExtractConfig:
    melspec_chunk_length: str
    start_subfolder: int
    end_subfolder: int
    split: str


def load_config() -> ExtractConfig:
    load_dotenv()
    try:
        start_subfolder = int(os.environ["START_SUBFOLDER"])
        end_subfolder = int(os.environ["END_SUBFOLDER"])
        melspec_chunk_length = MELSPEC_PREFIXES[int(os.environ["MELSPEC_PREFIX_INDEX"])]
        split = os.environ.get("SPLIT_TO_EXTRACT", "test")
    except KeyError as e:
        traceback.print_exc()
        raise Exception(f"Missing env var: {e.args[0]}")

    return ExtractConfig(
        melspec_chunk_length=melspec_chunk_length,
        start_subfolder=start_subfolder,
        end_subfolder=end_subfolder,
        split=split,
    )


def extract_archives(config: ExtractConfig):
    for i in range(config.start_subfolder, config.end_subfolder, 1):
        subfolder = str(i).zfill(2)
        base_name = f"{config.melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}"

        tar_loc = (
            f"{LOCAL_TEMP_FOLDER}/{LOCAL_ARCHIVE_FOLDER}/{config.split}/{base_name}.tar"
        )
        raw_loc = f"{LOCAL_TEMP_FOLDER}/{LOCAL_RAW_FOLDER}/{config.split}/{base_name}"

        if not os.path.exists(tar_loc):
            print(f"Archive missing at {tar_loc}. Skipping...")
            continue

        os.makedirs(raw_loc, exist_ok=True)

        try:
            print(f"Extracting {tar_loc} -> {raw_loc}")
            with tarfile.open(tar_loc) as t:
                t.extractall(path=raw_loc)

            # Remove .tar file to save space after extraction
            # os.remove(tar_loc)
            # print(f"Successfully extracted and removed local tar for {subfolder}.\n")
            print(f"Successfully extracted local tar for {subfolder}.\n")
        except Exception:
            traceback.print_exc()
            print(f"Failed to extract {tar_loc}")


if __name__ == "__main__":
    config = load_config()
    print(f"Starting extraction for split: {config.split}")
    extract_archives(config)
