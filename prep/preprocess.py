import numpy as np
import os
import librosa
import sys
import traceback
import multiprocessing as mp
from datetime import datetime
from dataclasses import dataclass
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
# Implicit: n_fft = 2048, hop_length = 512
SAMPLE_RATE = 16000
N_MEL_BANDS = 96
EPSILON = 10e-12  # For mel-band-wise z-score normalisation to avoid division by 0
INTENSITY_REF = 10e-12
N_MELSPEC_VARIANTS = 3  # Number of melspec chunk length variants (quasi-hardcoded)
N_MELSPEC_LENGTHS: list[int] = [
    157,
    469,
    938,
]  # Melspec output lengths (for these globals; hardcoded)
N_DELETION_WINDOW = 2

LOCAL_TEMP_FOLDER = "./temp"
MELSPEC_DIR_PREFIX = "melspec"
MELSPEC_5S_PREFIX = "5_"
MELSPEC_15S_PREFIX = "15_"
MELSPEC_30S_PREFIX = "30_"

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
]

# ====================================================================================================


@dataclass
class PreprocessorConfig:
    overwrite_remote_copy: bool
    redo_calculation: bool
    storage_base_url: str
    storage_base_folder: str
    melspec_remote: str
    melspec_storage: str
    start_subfolder: int
    end_subfolder: int


def load_config() -> PreprocessorConfig:
    """Load CLI arguments and environment variables into a typed object."""

    overwrite_all = False
    redo_calculation = False

    if len(sys.argv) == 3:
        overwrite_all = bool(sys.argv[1])
        redo_calculation = bool(sys.argv[2])
    elif len(sys.argv) == 2:
        overwrite_all = bool(sys.argv[1])

    try:
        # If the key is missing, raise KeyError
        storage_base_url = os.environ["REMOTE_BASE_URL"]
        storage_base_folder = storage_base_url + ":" + os.environ["REMOTE_BASE_FOLDER"]
        melspec_remote = os.environ["RCLONE_REMOTE"]
        melspec_storage = os.environ["REMOTE_MELSPEC_FOLDER"]

        start_subfolder = int(os.environ["START_SUBFOLDER"])
        end_subfolder = int(os.environ["END_SUBFOLDER"])

    except KeyError as e:
        # args[0] tells which variable is missing
        raise Exception(f".env file is incomplete! Missing: {e.args[0]}") from e

    # If one or more keys are empty, raise an error
    if not all(
        [
            os.environ["REMOTE_BASE_URL"] != "",
            os.environ["REMOTE_BASE_FOLDER"] != "",
            os.environ["RCLONE_REMOTE"] != "",
            os.environ["REMOTE_MELSPEC_FOLDER"] != "",
            os.environ["START_SUBFOLDER"] != "",
            os.environ["END_SUBFOLDER"] != "",
        ]
    ):
        raise Exception("One or more variables in the .env file is empty.")

    return PreprocessorConfig(
        overwrite_remote_copy=overwrite_all,
        redo_calculation=redo_calculation,
        storage_base_url=storage_base_url,
        storage_base_folder=storage_base_folder,
        melspec_remote=melspec_remote,
        melspec_storage=melspec_storage,
        start_subfolder=start_subfolder,
        end_subfolder=end_subfolder,
    )


def remote_melspec_subfolder_builder(
    config: PreprocessorConfig, subfolder=None, tcon=False
):
    if subfolder is None and tcon:
        return config.melspec_remote + ":" + config.melspec_storage
    elif subfolder is not None and not tcon:
        return (
            config.melspec_remote
            + ":"
            + config.melspec_storage
            + "/"
            + MELSPEC_DIR_PREFIX
            + subfolder
        )
    else:
        raise Exception("Subfolder can not be empty.")


def test_connections(config: PreprocessorConfig):
    print("Testing rclone remote connection...")
    if not rclone.check_remote_existing(config.melspec_remote):
        raise Exception(
            f"MELSPEC_REMOTE '{config.melspec_remote}' is incorrect, missing, or empty in rclone.conf."
        )
    print("Connection test passed.")


def sys_prepare(subfolder):
    if subfolder is None:
        raise Exception("Subfolder is required.")

    dir_path = LOCAL_TEMP_FOLDER + "/" + subfolder
    melspec_dir_path = LOCAL_TEMP_FOLDER + "/" + MELSPEC_DIR_PREFIX + subfolder

    if not os.path.exists(LOCAL_TEMP_FOLDER):
        os.mkdir(LOCAL_TEMP_FOLDER)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    if not os.path.exists(melspec_dir_path):
        os.mkdir(melspec_dir_path)


def local_melspec_path_builder(subfolder, item_in_question, melspec_chunk_length):
    return (
        LOCAL_TEMP_FOLDER
        + "/"
        + MELSPEC_DIR_PREFIX
        + subfolder
        + "/"
        + melspec_chunk_length
        + item_in_question.split(".")[0]
        + ".npy"
    )


def remote_melspec_path_builder(
    config: PreprocessorConfig, subfolder, item_in_question, melspec_chunk_length
):
    return (
        config.melspec_remote
        + ":"
        + config.melspec_storage
        + "/"
        + MELSPEC_DIR_PREFIX
        + subfolder
        + "/"
        + melspec_chunk_length
        + item_in_question.split(".")[0]
        + ".npy"
    )


def music_file_exists(subfolder, item_in_question):
    return os.path.exists(LOCAL_TEMP_FOLDER + "/" + subfolder + "/" + item_in_question)


def melspec_exists_locally(subfolder, item_in_question):
    return all(
        [
            os.path.exists(
                local_melspec_path_builder(
                    subfolder, item_in_question, MELSPEC_5S_PREFIX
                )
            ),
            os.path.exists(
                local_melspec_path_builder(
                    subfolder, item_in_question, MELSPEC_15S_PREFIX
                )
            ),
            os.path.exists(
                local_melspec_path_builder(
                    subfolder, item_in_question, MELSPEC_30S_PREFIX
                )
            ),
        ]
    )


def melspec_exists_remotely_uncorrupted(
    config: PreprocessorConfig, subfolder, item_in_question
):
    conns = []
    melspec_prefixes = [MELSPEC_5S_PREFIX, MELSPEC_15S_PREFIX, MELSPEC_30S_PREFIX]
    try:
        for i, mpref in enumerate(melspec_prefixes):
            remote_npy_path = remote_melspec_path_builder(
                config, subfolder, item_in_question, mpref
            )
            local_npy_path = local_melspec_path_builder(
                subfolder, item_in_question, mpref
            )

            try:
                out_dir = f"{LOCAL_TEMP_FOLDER}/{MELSPEC_DIR_PREFIX}{subfolder}"

                rclone.copy(
                    in_path=remote_npy_path,
                    out_path=out_dir,
                    ignore_existing=False,
                    args=RCLONE_ARGS,
                )

                if not os.path.exists(local_npy_path):
                    res = False
                else:
                    remote_res = np.load(local_npy_path)
                    if int(remote_res.shape[2]) != N_MELSPEC_LENGTHS[i]:
                        res = False  # Corrupted
                    else:
                        res = True
            except Exception:
                res = False
            finally:
                conns.append(res)

        return all(conns)
    except Exception:
        traceback.print_exc()
        print("Error checking remote melspec corruption. Assuming corrupted/missing.")
        return False


def copy_melspec_to_remote(config: PreprocessorConfig, subfolder):
    in_path = LOCAL_TEMP_FOLDER + "/" + MELSPEC_DIR_PREFIX + subfolder
    out_path = remote_melspec_subfolder_builder(config, subfolder=subfolder)
    ignore_existing = not config.overwrite_remote_copy

    # rclone.copy(
    #     in_path=in_path,
    #     out_path=out_path,
    #     ignore_existing=ignore_existing,
    #     show_progress=True,
    # )
    rclone.copy(
        in_path=in_path,
        out_path=out_path,
        ignore_existing=ignore_existing,
        show_progress=False,
        args=RCLONE_ARGS,
    )


def sys_cleanup(subfolder):
    # This function is needed because all outputs are going to be copied to a remote storage

    if subfolder is None:
        raise Exception("Subfolder is required.")

    dir_path = LOCAL_TEMP_FOLDER + "/" + subfolder

    # List all files in the directory
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)

        # Check if it is a file (not a subdirectory)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {filename}")

    try:
        os.rmdir(dir_path)
        print(f"Directory '{dir_path}' has been removed successfully.")
    except OSError as error:
        print(error)
        print(f"Directory '{dir_path}' could not be removed.")


def znorm(col):
    return (col - np.average(col)) / (np.sqrt(np.var(col) + EPSILON))


def calc_melspec(subfolder, item_in_question, naive_interval=15):
    if item_in_question is None:
        raise Exception("The item in question does not exist.")

    interval = SAMPLE_RATE * naive_interval

    # Load file locally from temp folder
    y, sr = librosa.load(
        LOCAL_TEMP_FOLDER + "/" + subfolder + "/" + item_in_question, sr=SAMPLE_RATE
    )

    # rawmels = []
    melspecs: list[np.ndarray] = []

    for i in range(len(y) // interval):
        lower = i * interval
        upper = (i + 1) * interval

        if upper > len(y):
            upper = -1

        si = librosa.feature.melspectrogram(y=y[lower:upper], sr=sr, n_mels=N_MEL_BANDS)
        s_dbi = librosa.amplitude_to_db(si, ref=INTENSITY_REF)

        s_dbi_norm = np.apply_along_axis(
            znorm, 1, s_dbi
        )  # apply across mel bands ie. rows

        # rawmels.append(s_dbi)
        melspecs.append(s_dbi_norm)

    return np.array(melspecs)


def load_remote_dir(config: PreprocessorConfig, subfolder):
    if subfolder is None:
        raise Exception("Subfolder is required")

    remote_audio_path = f"{config.storage_base_folder}/{subfolder}"

    try:
        return rclone.ls(remote_audio_path)
    except Exception as e:
        raise Exception(
            f"Failed to list remote directory {remote_audio_path}. Error: {e}"
        )


def calc_remote_dir(
    config: PreprocessorConfig, children: list[dict[str, int | str]], subfolder: str
):
    if len(children) == 0:
        raise Exception("Input length is zero.")
    if subfolder is None:
        raise Exception("Subfolder is required.")

    len_children = len(children)

    for i, child in enumerate(children):
        item_in_question = child["Name"]

        print(
            ">>>",
            datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss"),
            subfolder,
            item_in_question,
            "\t",
            i + 1,
            "out of",
            len_children,
        )

        if (
            melspec_exists_locally(subfolder, item_in_question)
            and not config.redo_calculation
        ):
            print(
                f"All melspecs for {item_in_question} already exist locally. Skipping calculation..."
            )
            continue

        music_file_path = f"{LOCAL_TEMP_FOLDER}/{subfolder}/{item_in_question}"

        if music_file_exists(subfolder, item_in_question):
            print(f"{item_in_question} already exist locally. Skipping download...")
        else:
            # Download raw audio via rclone instead of requests
            remote_audio_file = (
                f"{config.storage_base_folder}/{subfolder}/{item_in_question}"
            )
            local_audio_dir = f"{LOCAL_TEMP_FOLDER}/{subfolder}"

            try:
                rclone.copy(
                    in_path=remote_audio_file,
                    out_path=local_audio_dir,
                    ignore_existing=True,
                    args=RCLONE_ARGS,
                )
            except Exception as e:
                print(
                    f"Rclone failed to download {item_in_question}: {e}. Skipping this file..."
                )
                continue  # Skip to the next file, idempotency will catch it eventually :)

            # Failsafe check
            if not os.path.exists(music_file_path):
                print(
                    f"File {item_in_question} not found locally after rclone download. Skipping..."
                )
                continue

        try:
            with open(
                local_melspec_path_builder(
                    subfolder, item_in_question, MELSPEC_5S_PREFIX
                ),
                "wb",
            ) as f:
                np.save(f, calc_melspec(subfolder, item_in_question, naive_interval=5))

            with open(
                local_melspec_path_builder(
                    subfolder, item_in_question, MELSPEC_15S_PREFIX
                ),
                "wb",
            ) as f:
                np.save(f, calc_melspec(subfolder, item_in_question, naive_interval=15))

            with open(
                local_melspec_path_builder(
                    subfolder, item_in_question, MELSPEC_30S_PREFIX
                ),
                "wb",
            ) as f:
                np.save(f, calc_melspec(subfolder, item_in_question, naive_interval=30))

        except Exception:
            traceback.print_exc()
            print(
                f"Error calculating and saving melspec locally for {item_in_question}. Skipping..."
            )


def is_subfolder_complete(config: PreprocessorConfig, subfolder):
    if (
        rclone.size(remote_melspec_subfolder_builder(config, subfolder))["count"]
        == int(
            rclone.size(
                config.melspec_remote + ":" + config.melspec_storage + "/" + subfolder
            )["count"]
        )
        * N_MELSPEC_VARIANTS
    ):
        print(
            f"Subfolder {subfolder} has complete melspecs. Skipping this subfolder..."
        )
        return True
    else:
        return False


def preprocess_all(config: PreprocessorConfig):
    subfolders = []
    for i in range(config.start_subfolder, config.end_subfolder, 1):
        subfolders.append(str(i).zfill(2))

    for i, subfolder in enumerate(subfolders):
        try:
            if is_subfolder_complete(config, subfolder) and not config.redo_calculation:
                print("Subfolder is complete. Skipping...")
                continue
        except Exception as e:
            print(
                f"Subfolder doesn't exist yet. Calculating... Original error is ignored: {e.args}"
            )

        sys_prepare(subfolder)

        try:
            children = load_remote_dir(config, subfolder)
            calc_remote_dir(config, children, subfolder)
        except Exception:
            traceback.print_exc()
            print("Error(s) occured during preprocessing.")
        finally:
            copy_melspec_to_remote(config, subfolder)

            # Deletion window just in case of corruption in tail-end subfolders
            if i - N_DELETION_WINDOW >= 0:
                sys_cleanup(subfolders[i - N_DELETION_WINDOW])
                sys_cleanup(MELSPEC_DIR_PREFIX + subfolders[i - N_DELETION_WINDOW])

            # Make function continue anyway when error
            continue


def preprocess_worker(config: PreprocessorConfig, subfolders_chunk: list):
    """Worker function for multiprocessing."""

    # Run the exact same logic as preprocess_all but only on specific chunks separately
    for i, subfolder in enumerate(subfolders_chunk):
        try:
            if is_subfolder_complete(config, subfolder) and not config.redo_calculation:
                # If subfolder is complete then error check it. Though this can only happen on the second run
                children = load_remote_dir(config, subfolder)

                if len(children) == 0:
                    raise Exception("Input length is zero.")

                len_children = len(children)

                for i, child in enumerate(children):
                    item_in_question = child["Name"]

                    print(
                        ">>>",
                        datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss"),
                        subfolder,
                        item_in_question,
                        "\t",
                        i + 1,
                        "out of",
                        len_children,
                    )

                    if melspec_exists_remotely_uncorrupted(
                        config, subfolder, item_in_question
                    ):
                        print(
                            f"All melspecs for {item_in_question} already exist remotely. Skipping..."
                        )
                        continue

                continue
            else:
                print(f"Subfolder {subfolder} is incomplete at remote. Calculating...")
        except Exception as e:
            print(
                f"Subfolder {subfolder} doesn't exist remotely. Calculating... Error ignored: {e.args}"
            )

        sys_prepare(subfolder)

        try:
            children = load_remote_dir(config, subfolder)
            calc_remote_dir(config, children, subfolder)

            copy_melspec_to_remote(config, subfolder)

            # Deletion window just in case of corruption in tail-end subfolders
            if i - N_DELETION_WINDOW >= 0:
                sys_cleanup(subfolders_chunk[i - N_DELETION_WINDOW])
                sys_cleanup(
                    MELSPEC_DIR_PREFIX + subfolders_chunk[i - N_DELETION_WINDOW]
                )
        except Exception:
            traceback.print_exc()
            print(f"Error(s) occured during preprocessing of subfolder {subfolder}.")
        finally:
            # Continue regardless of errors
            continue


def preprocess_all_mp(
    config: PreprocessorConfig, start_subfolder=0, end_subfolder=100, num_processes=4
):
    """
    Splits total subfolders into interleaved lists and assigns them to independent processes.
    """
    # Generate the total list of subfolders (e.g. "00" to "03", or "00" to "99")
    all_subfolders = [str(i).zfill(2) for i in range(start_subfolder, end_subfolder)]

    # Split into chunks. Given num_processes=4:
    # chunk 0: 00, 04, 08...
    # chunk 1: 01, 05, 09...
    # etc.
    chunks = [all_subfolders[i::num_processes] for i in range(num_processes)]

    processes: list[mp.Process] = []

    # Spin up child processes
    for chunk in chunks:
        if not chunk:
            continue  # Skip empty chunks

        p = mp.Process(target=preprocess_worker, args=(config, chunk))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()


def main():
    from dotenv import load_dotenv

    load_dotenv()

    config = load_config()

    test_connections(config)

    preprocess_all_mp(
        config,
        start_subfolder=config.start_subfolder,
        end_subfolder=config.end_subfolder,
        num_processes=4,
    )


if __name__ == "__main__":
    main()
