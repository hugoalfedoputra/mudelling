import requests
import pandas as pd
import numpy as np
import os
import librosa
import sys
import traceback
import multiprocessing as mp
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
from dataclasses import dataclass
from requests.auth import HTTPBasicAuth
from rclone_python import rclone

# from tqdm import tqdm

# Globals
SAMPLE_RATE = 16000
N_MEL_BANDS = 96
EPSILON = 10e-12  # For mel-band-wise z-score normalisation to avoid division by 0
INTENSITY_REF = 10e-12
N_MELSPEC_VARIANTS = 3  # Number of melspec chunk length variants (quasi-hardcoded)
N_DELETION_WINDOW = 2

LOCAL_TEMP_FOLDER = "./temp"
MELSPEC_DIR_PREFIX = "melspec"
MELSPEC_5S_PREFIX = "5_"
MELSPEC_15S_PREFIX = "15_"
MELSPEC_30S_PREFIX = "30_"


@dataclass
class PreprocessorConfig:
    overwrite_remote_copy: bool
    redo_calculation: bool
    storage_base_url: str
    storage_base_folder: str
    melspec_remote: str
    melspec_storage: str
    storage_auth: HTTPBasicAuth
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
        storage_base_folder = storage_base_url + os.environ["REMOTE_BASE_FOLDER"]
        melspec_remote = os.environ["RCLONE_REMOTE"]
        melspec_storage = os.environ["REMOTE_MELSPEC_FOLDER"]

        storage_auth = HTTPBasicAuth(
            username=os.environ["REMOTE_USERNAME"],
            password=os.environ["REMOTE_PASSWORD"],
        )

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
            os.environ["REMOTE_USERNAME"] != "",
            os.environ["REMOTE_PASSWORD"] != "",
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
        storage_auth=storage_auth,
        start_subfolder=start_subfolder,
        end_subfolder=end_subfolder,
    )


def remote_melspec_path_builder(config: PreprocessorConfig, subfolder=None, tcon=False):
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


def test_connections(config: PreprocessorConfig, session: requests.Session):
    # Test rclone
    conns = [
        rclone.check_remote_existing(config.melspec_remote),
    ]
    if not all(conns):
        raise Exception(
            conns,
            "MELSPEC_REMOTE and/or MELSPEC_STORAGE is incorrect, missing, or empty.",
        )

    # Test remote storage
    # try:
    #     res = session.get(config.storage_base_folder)

    #     if res.status_code != 200:
    #         raise Exception(
    #             "Remote storage configuration incorrect. Got status code:",
    #             res.status_code,
    #             res.json(),
    #         )
    # except Exception as e:
    #     raise Exception(f"Connection failed with: {e.args}") from e


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


def melspec_path_builder(subfolder, item_in_question, melspec_chunk_length):
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


def music_file_exists(subfolder, item_in_question):
    return os.path.exists(LOCAL_TEMP_FOLDER + "/" + subfolder + "/" + item_in_question)


def melspec_exists(subfolder, item_in_question):
    return all(
        [
            os.path.exists(
                melspec_path_builder(subfolder, item_in_question, MELSPEC_5S_PREFIX)
            ),
            os.path.exists(
                melspec_path_builder(subfolder, item_in_question, MELSPEC_15S_PREFIX)
            ),
            os.path.exists(
                melspec_path_builder(subfolder, item_in_question, MELSPEC_30S_PREFIX)
            ),
        ]
    )


def copy_melspec_to_remote(config: PreprocessorConfig, subfolder):
    in_path = LOCAL_TEMP_FOLDER + "/" + MELSPEC_DIR_PREFIX + subfolder
    out_path = remote_melspec_path_builder(config, subfolder=subfolder)
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

    # Load file locally
    y, sr = librosa.load(item_in_question, sr=SAMPLE_RATE)

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


def load_remote_dir(config: PreprocessorConfig, session: requests.Session, subfolder):
    if subfolder is None:
        raise Exception("Subfolder is required")

    # Loads dir where audio files are (can only be from remote)
    # res = session.get(config.storage_base_folder + subfolder)

    # if res.status_code == 200:
    #     return res
    # else:
    #     raise Exception(
    #         "Error when getting remote subfolder with status code:",
    #         res.status_code,
    #         res.json(),
    #     )


def calc_remote_dir(
    config: PreprocessorConfig, session: requests.Session, dir, subfolder
):
    if dir is None:
        raise Exception("Input is empty.")

    if len(dir.json()) <= 0:
        raise Exception("Input length is zero.")

    if subfolder is None:
        raise Exception("Subfolder is required.")

    children = dir.json()["children"]
    # for child in tqdm(children):
    # for child in children:
    for filename in os.listdir(os.path.join(config.storage_base_folder, subfolder)):
        print(">>>", subfolder, filename)
        item_in_question = filename
        path = os.path.join(config.storage_base_folder, subfolder, filename)

        if melspec_exists(subfolder, item_in_question) and not config.redo_calculation:
            print(
                f"All melspecs for {item_in_question} already exist. Skipping calculation..."
            )
            continue

        # if music_file_exists(subfolder, item_in_question):
        #     print(f"{item_in_question} already exist. Skipping download...")
        # else:
        # Loads raw audio file from remote location (can only be from remote)
        # music_file = session.get(config.storage_base_url + path)

        # if music_file.status_code == 200:
        #     music_file_path = (
        #         LOCAL_TEMP_FOLDER + "/" + subfolder + "/" + item_in_question
        #     )

        #     if not os.path.exists(music_file_path):
        #         try:
        #             f = open(
        #                 music_file_path,
        #                 mode="xb",
        #             )
        #             f.write(music_file.content)
        #             f.close()
        #         except Exception:
        #             raise Exception(
        #                 "Error when saving music file to local temp folder."
        #             )
        # else:
        #     raise Exception("Error at music file download:", music_file.status_code)

        try:
            with open(
                melspec_path_builder(subfolder, item_in_question, MELSPEC_5S_PREFIX),
                "wb",
            ) as f:
                np.save(
                    f,
                    calc_melspec(subfolder, path, naive_interval=5),
                )

            with open(
                melspec_path_builder(subfolder, item_in_question, MELSPEC_15S_PREFIX),
                "wb",
            ) as f:
                np.save(
                    f,
                    calc_melspec(subfolder, path, naive_interval=15),
                )

            with open(
                melspec_path_builder(subfolder, item_in_question, MELSPEC_30S_PREFIX),
                "wb",
            ) as f:
                np.save(
                    f,
                    calc_melspec(subfolder, path, naive_interval=30),
                )
        except Exception:
            raise Exception(
                "There was an error at:",
                subfolder,
                "especially at:",
                item_in_question,
            )


def is_subfolder_complete(config: PreprocessorConfig, subfolder):
    if (
        rclone.size(remote_melspec_path_builder(config, subfolder))["count"]
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


def preprocess_all(config: PreprocessorConfig, s: requests.Session):
    subfolders = []
    for i in range(0, 4, 1):
        subfolders.append(str(i).zfill(2))

    for i, subfolder in enumerate(subfolders):
        try:
            if is_subfolder_complete(config, subfolder) and not config.redo_calculation:
                continue
        except Exception as e:
            print(
                f"Subfolder doesn't exist yet. Calculating... Original error is ignored: {e.args}"
            )

        sys_prepare(subfolder)

        try:
            dir = load_remote_dir(config, s, subfolder)
            calc_remote_dir(config, s, dir, subfolder)
        except Exception:
            traceback.print_exc()
            print("Error(s) occured during preprocessing.")
        finally:
            copy_melspec_to_remote(config, subfolder)

            # Deletion window just in case of corruption in a certain subfolder
            if i - N_DELETION_WINDOW >= 0:
                sys_cleanup(subfolders[i - N_DELETION_WINDOW])
                sys_cleanup(MELSPEC_DIR_PREFIX + subfolders[i - N_DELETION_WINDOW])

            # Make function continue anyway when error
            continue


def preprocess_worker(config: PreprocessorConfig, subfolders_chunk: list):
    """
    Worker function for multiprocessing.
    Each process MUST create its own requests.Session() because open sockets cannot be pickled.
    """
    from urllib3.util import Retry
    from requests.adapters import HTTPAdapter
    import requests

    # Initialize process-specific session
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.2,
        backoff_max=30,
        backoff_jitter=2,
        status_forcelist=[404, 429, 500, 502, 503, 504],
        allowed_methods={"GET"},
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0"
        }
    )
    s.auth = config.storage_auth

    # Run the exact same logic as preprocess_all, but ONLY on this specific chunk
    for i, subfolder in enumerate(subfolders_chunk):
        try:
            if is_subfolder_complete(config, subfolder) and not config.redo_calculation:
                continue
        except Exception as e:
            print(
                f"Subfolder {subfolder} doesn't exist yet. Calculating... Original error ignored: {e.args}"
            )

        sys_prepare(subfolder)

        try:
            dir = load_remote_dir(config, s, subfolder)
            calc_remote_dir(config, s, dir, subfolder)
        except Exception:
            traceback.print_exc()
            print(f"Error(s) occured during preprocessing of subfolder {subfolder}.")
        finally:
            copy_melspec_to_remote(config, subfolder)

            # Deletion window to clean up inside this process's specific chunk
            if i - N_DELETION_WINDOW >= 0:
                sys_cleanup(subfolders_chunk[i - N_DELETION_WINDOW])
                sys_cleanup(
                    MELSPEC_DIR_PREFIX + subfolders_chunk[i - N_DELETION_WINDOW]
                )


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

    processes = []

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

    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.2,
        backoff_max=30,
        backoff_jitter=2,
        status_forcelist=[404, 429, 500, 502, 503, 504],
        allowed_methods={"GET"},
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0"
        }
    )
    s.auth = config.storage_auth

    test_connections(config, s)

    # preprocess_all(config, s)
    preprocess_all_mp(
        config,
        start_subfolder=config.start_subfolder,
        end_subfolder=config.end_subfolder,
        num_processes=4,
    )


if __name__ == "__main__":
    main()
