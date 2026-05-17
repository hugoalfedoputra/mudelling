import os
import traceback
from dotenv import load_dotenv
from rclone_python import rclone

# ====================================================================================================
#
# THIS FILE WAS TO BE RUN ONCE AND IT HAS BEEN RUN
# IT IS THEORETICALLY IDEMPOTENT BUT IT HAS NOT BEEN TESTED YET
# ====================================================================================================

# This is from preprocess.py; hard-coded so as to not cause import issues during build
MELSPEC_PREFIXES = ["5_", "15_", "30_"]
MELSPEC_DIR_PREFIX = "melspec"


def load_config():
    """Load configuration from the .env file."""
    load_dotenv()
    try:
        melspec_remote = os.environ["RCLONE_REMOTE"]
        melspec_storage = os.environ["REMOTE_MELSPEC_FOLDER"]
        start_subfolder = int(os.environ["START_SUBFOLDER"])
        end_subfolder = int(os.environ["END_SUBFOLDER"])
    except KeyError as e:
        raise Exception(f"Missing env var: {e.args[0]}")

    return melspec_remote, melspec_storage, start_subfolder, end_subfolder


def migrate_remote_folders():
    melspec_remote, melspec_storage, start_subfolder, end_subfolder = load_config()

    for i in range(start_subfolder, end_subfolder):
        subfolder = str(i).zfill(2)
        old_folder_path = (
            f"{melspec_remote}:{melspec_storage}/{MELSPEC_DIR_PREFIX}{subfolder}"
        )

        print(f"\n========================================================")
        print(f"Processing Subfolder: {subfolder}")
        print(f"Source: {old_folder_path}")
        print(f"========================================================")

        # Quick check if the old directory exists to prevent rclone from throwing errors
        try:
            rclone.ls(old_folder_path)
        except Exception:
            print(f"Directory {old_folder_path} not found. Skipping...")
            continue

        for mpref in MELSPEC_PREFIXES:
            new_folder_path = f"{melspec_remote}:{melspec_storage}/{mpref}{MELSPEC_DIR_PREFIX}{subfolder}"

            print(f"\n[+] Creating destination directory: {new_folder_path}")
            try:
                # rclone_args = ["--dry-run"]
                # rclone.mkdir(new_folder_path, args=rclone_args)
                rclone.mkdir(new_folder_path)
            except Exception as e:
                print(f"Warning during mkdir for {new_folder_path}: {e}")

            print(f"[*] Moving '{mpref}' files from old folder to new folder...")
            try:
                rclone_args = [
                    # "--dry-run",
                    "--include",
                    f"{mpref}*.npy",
                    "--transfers",
                    "16",
                    "--retries",
                    "10",
                ]

                rclone.move(
                    in_path=old_folder_path,
                    out_path=new_folder_path,
                    args=rclone_args,
                    show_progress=True,
                )
            except Exception:
                traceback.print_exc()
                print(f"[-] Error moving {mpref} files for subfolder {subfolder}.")


if __name__ == "__main__":
    print("Starting melspec folder structure migration at remote...")

    migrate_remote_folders()

    print("\nMigration finished.")
