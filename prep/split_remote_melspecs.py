import os
import multiprocessing as mp
import numpy as np
import traceback
import time
from datetime import datetime
from dotenv import load_dotenv
from rclone_python import rclone
from dataclasses import dataclass

# import subprocess

# ====================================================================================================
#
# THIS FILE WAS TO BE RUN ONCE AND IT HAS BEEN RUN
# IT IS THEORETICALLY IDEMPOTENT BUT IT HAS NOT BEEN TESTED YET
# ====================================================================================================

# ====================================================================================================
#
# Globals
# ====================================================================================================

# This is from preprocess.py; hard-coded so as to not cause import issues during build
MELSPEC_PREFIXES = ["5_", "15_", "30_"]
MELSPEC_DIR_PREFIX = "melspec"
LOCAL_TEMP_FOLDER = "./temp"
N_DELETION_WINDOW = 2

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


@dataclass
class PreprocessorConfig:
    melspec_remote: str
    melspec_storage: str
    start_subfolder: int
    end_subfolder: int
    num_processes: int


def load_config() -> PreprocessorConfig:
    load_dotenv()
    try:
        return PreprocessorConfig(
            melspec_remote=os.environ["RCLONE_REMOTE"],
            melspec_storage=os.environ["REMOTE_MELSPEC_FOLDER"],
            start_subfolder=int(os.environ["START_SUBFOLDER"]),
            end_subfolder=int(os.environ["END_SUBFOLDER"]),
            num_processes=int(os.environ["NUM_PROCESSES"]),
        )
    except KeyError as e:
        raise Exception(f"Missing env var: {e.args[0]}")


def sys_cleanup(subfolder_path: str):
    """Deletes a local directory and its contents."""
    dir_path = f"{LOCAL_TEMP_FOLDER}/{subfolder_path}"
    if not os.path.exists(dir_path):
        return
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    try:
        os.rmdir(dir_path)
    except OSError as error:
        print(f"Error removing {dir_path}: {error}")


def split_worker(config: PreprocessorConfig, mpref: str, subfolders_chunk: list):
    """Worker function to download, split, and upload files."""
    for i, subfolder in enumerate(subfolders_chunk):
        remote_path = f"{config.melspec_remote}:{config.melspec_storage}/{mpref}{MELSPEC_DIR_PREFIX}{subfolder}"
        local_temp = f"{LOCAL_TEMP_FOLDER}/{mpref}{MELSPEC_DIR_PREFIX}{subfolder}"
        os.makedirs(local_temp, exist_ok=True)

        print(f"\n[{mpref}{subfolder}] Checking remote files...")
        try:
            print(
                ">>>",
                datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss"),
            )

            files_info = rclone.ls(remote_path)
            file_names = [f["Name"] for f in files_info]

            # Original combined files have exactly 1 underscore (e.g., '5_12345.npy')
            combined_files = [
                f
                for f in file_names
                if str(f).count("_") == 1 and str(f).endswith(".npy")
            ]
            # Split files have 2 underscores (e.g., '5_12345_0.npy')
            split_files = set(
                [
                    f
                    for f in file_names
                    if str(f).count("_") >= 2 and str(f).endswith(".npy")
                ]
            )

            for cf in combined_files:
                # cf = "5_12345.npy" -> raw_name = "12345"
                raw_name = str(cf).replace(mpref, "").replace(".npy", "")

                # IDEMPOTENCY CHECK: If the first chunk is already on remote, skip
                first_chunk_name = f"{mpref}{raw_name}_0.npy"
                if first_chunk_name in split_files:
                    print(
                        f"[{mpref}{subfolder}] {cf} is already split remotely. Skipping..."
                    )
                    continue

                print(f"[{mpref}{subfolder}] Downloading {cf}...")
                local_cf = f"{local_temp}/{cf}"

                try:
                    start_time = time.time()
                    rclone.copy(
                        in_path=f"{remote_path}/{cf}",
                        out_path=local_temp,
                        args=RCLONE_ARGS,
                        show_progress=False,
                    )
                    elapsed_time = time.time() - start_time
                    print(
                        f"Finished copying from remote in {elapsed_time}s. Continuing...\n"
                    )

                    if not os.path.exists(local_cf):
                        print(f"[{mpref}{subfolder}] Failed to download {cf}.")
                        continue

                    # Load combined array: shape (chunks, time, mels)
                    arr = np.load(local_cf)
                    num_chunks = arr.shape[0]

                    print(
                        f"[{mpref}{subfolder}] Splitting {cf} into {num_chunks} chunks..."
                    )
                    for idx in range(num_chunks):
                        chunk_path = f"{local_temp}/{mpref}{raw_name}_{idx}.npy"
                        np.save(chunk_path, arr[idx])

                    # Delete local combined file BEFORE uploading so it isn't copied back to remote
                    os.remove(local_cf)

                except Exception as e:
                    print(f"[{mpref}{subfolder}] Error processing {cf}: {e}")
                    traceback.print_exc()
                    continue

            # Upload all newly created split files in this temp folder back to remote
            print(f"[{mpref}{subfolder}] Uploading split files to remote...")

            start_time = time.time()
            rclone.copy(
                in_path=local_temp,
                out_path=remote_path,
                args=RCLONE_ARGS,
                show_progress=False,
            )
            elapsed_time = time.time() - start_time
            print(f"Finished copying to remote in {elapsed_time}s. Continuing...\n")

        except Exception as e:
            print(
                ">>>",
                datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss"),
            )
            print(f"[{mpref}{subfolder}] Fatal error handling directory: {e}")
            traceback.print_exc()

        # Local temp deletion window
        target_idx = i - N_DELETION_WINDOW
        if target_idx >= 0:
            sys_cleanup(f"{mpref}{MELSPEC_DIR_PREFIX}{subfolders_chunk[target_idx]}")


def cleanup_remote_sources(config: PreprocessorConfig, mpref: str, subfolder: str):
    """Deletes the original combined files from remote AFTER all splits are securely uploaded."""
    remote_path = f"{config.melspec_remote}:{config.melspec_storage}/{mpref}{MELSPEC_DIR_PREFIX}{subfolder}"

    try:
        print(
            ">>>",
            datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss"),
        )

        files_info = rclone.ls(remote_path)
        file_names = [f["Name"] for f in files_info]
        combined_files = [
            f for f in file_names if str(f).count("_") == 1 and str(f).endswith(".npy")
        ]

        if not combined_files:
            return

        print(
            f"[{mpref}{subfolder}] Cleaning up {len(combined_files)} original source files from remote..."
        )

        # Write filenames to a text file for rclone to process natively
        os.makedirs(LOCAL_TEMP_FOLDER, exist_ok=True)
        txt_path = f"{LOCAL_TEMP_FOLDER}/to_delete_{mpref}{subfolder}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(combined_files))  # type: ignore

        # Use subprocess directly to use the --files-from command
        # subprocess.run(
        #     ["rclone", "delete", remote_path, "--files-from", txt_path] + RCLONE_ARGS,
        #     check=True,
        # )

        delete_args = ["--files-from", txt_path]

        rclone.delete(remote_path, args=RCLONE_ARGS + delete_args)

        os.remove(txt_path)
        print(f"[{mpref}{subfolder}] Remote cleanup successful.")

    except Exception as e:
        print(f"[{mpref}{subfolder}] Error cleaning up remote sources: {e}")


def main():
    config = load_config()
    all_subfolders = [
        str(i).zfill(2) for i in range(config.start_subfolder, config.end_subfolder)
    ]
    num_processes = config.num_processes

    print(
        ">>>",
        datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss"),
    )

    # 1. Outer loop: Chunk length prefixes
    for mpref in MELSPEC_PREFIXES:
        print(
            f"\n\n{'='*60}\nSTARTING SPLIT PROCESS FOR CHUNK LENGTH: {mpref}\n{'='*60}"
        )

        # Split subfolders among processes
        chunks = [all_subfolders[i::num_processes] for i in range(num_processes)]
        processes: list[mp.Process] = []

        # 2. Fire up workers for splitting/uploading
        for chunk in chunks:
            if not chunk:
                continue
            p = mp.Process(target=split_worker, args=(config, mpref, chunk))
            p.start()
            processes.append(p)

        # 3. Wait for all workers in this prefix to completely finish
        for p in processes:
            p.join()

        # 4. Perform Remote Cleanup ONLY after the whole prefix is done
        print(
            f"\n\n{'='*60}\nWORKERS FINISHED FOR {mpref}. INITIATING REMOTE CLEANUP...\n{'='*60}"
        )
        for subfolder in all_subfolders:
            cleanup_remote_sources(config, mpref, subfolder)

    print("\n\nAll tasks completed successfully!")
    print(
        ">>>",
        datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss"),
    )


if __name__ == "__main__":
    main()
