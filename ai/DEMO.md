# Demo

## This demo assumes that you have deployed your own MLflow instance, be it locally or publicly with a domain name.

For MLflow instance settings, please fill these variables in the `.env` in this `/ai` dir:

```
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
MLFLOW_TRACKING_URI
```

# Steps

1. Copy `prepare_demo.example.py` and paste as `prepare_demo.py` in this `/ai` dir
2. Scroll all the way down and change `source_audio_dir="/path/to/files",` in the try catch calling the `create_mini_dataset()` function to the actual source file dirs. The source file in question is the raw audio files from MTG-Jamendo "mood/theme" subset
3. Run `uv sync --extra rocm72` if you have not
4. Run `uv run prepare_demo.py`. If you need to check whether the files that are randomly sampled are "good enough", you can set `export_selection` to `False` under the `CHANGE THESE` section of the `prepare_demo.py` file
5. Change the `.env` file **in `/prep`** to:
    ```
    REMOTE_BASE_URL=LOCAL
    REMOTE_BASE_FOLDER=../ai/tempdemo/raw

    RCLONE_REMOTE=LOCAL
    REMOTE_MELSPEC_FOLDER=../ai/tempdemo/archive

    START_SUBFOLDER=0
    END_SUBFOLDER=1

    NUM_PROCESSES=4

    TRAIN_CSV=../ai/tempdemo/csv/mini_dataset_train.csv
    VAL_CSV=../ai/tempdemo/csv/mini_dataset_val.csv
    TEST_CSV=../ai/tempdemo/csv/mini_dataset_test.csv
    ```
6. Run `uv run preprocess.py` **in the `/prep` dir**.
7. Open the file called `extract_archived_melspecs.py` and change some global variables:
    ```
    LOCAL_TEMP_FOLDER = "./tempdemo"
    LOCAL_RAW_FOLDER = "melspec"
    ```
8. Change only *some* of the variables in the `.env` file **in `/ai`** with:

    ```
    START_SUBFOLDER=0
    END_SUBFOLDER=1

    MELSPEC_PREFIX_INDEX=1

    TRAINING_METADATA_CSV=./tempdemo/csv/mini_dataset_train.csv
    VALIDATION_METADATA_CSV=./tempdemo/csv/mini_dataset_val.csv

    TESTING_METADATA_CSV=./tempdemo/csv/mini_dataset_test.csv

    SPLIT_TO_EXTRACT=train
    ```
9. Run `uv run extract_archived_melspecs.py`. This will extract the melspecs *only* for the training split and *only* for the 15 second spectrogram chunks because we set `MELSPEC_PREFIX_INDEX=1` in the `.env` at step 8.
10. Change a variable in the `.env` file **in `/ai`** to:
    ```
    SPLIT_TO_EXTRACT=val
    ```
11. Repeat step 9. That time will extract the melspecs *only* for the validation split. Then repeat step 10 but change `val` to `test` then repeat step 9 one more time which will extract the melspecs *only* for the testing split.
12. Copy `params.example.json` three times and paste as: `params_cnn_demo.json`, `params_gru_demo.json`, and `params_attn_demo.json`.
13. Fill the three files respectively with:
    ```
    {
        "backend_dropout": 0.1,
        "backend_type": "CNN",
        "batch_size": 32,
        "classifier_dropout": 0.0,
        "epochs": 3,
        "learning_rate": 0.001,
        "post_global_dropout": 0.5
    }
    ```
    ```
    {
        "backend_dropout": 0.5,
        "backend_type": "GRU",
        "batch_size": 32,
        "classifier_dropout": 0.0,
        "epochs": 3,
        "learning_rate": 0.0001,
        "post_global_dropout": 0.5
    }
    ```
    ```
    {
        "backend_dropout": 0.0,
        "backend_type": "ATTN",
        "batch_size": 32,
        "classifier_dropout": 0.0,
        "epochs": 3,
        "learning_rate": 0.0001,
        "post_global_dropout": 0.5
    }
    ```
14. Copy `train.py` and paste as `traindemo.py` in this `/ai` dir
15. Set global variables:
    ```
    IS_TESTING = False
    EXPERIMENT_IDENT = "demo_1"
    JSON_CONFIG_PATH = "./params_cnn_demo.json"

    LOCAL_TEMP_FOLDER = "tempdemo"
    LOCAL_RAW_FOLDER = "melspec"
    ```
    Note: ignore the rest, it is fine. \
    Second note: notice `EXPERIMENT_IDENT` is set to `"demo_1"`, if you had done the first demo beforehand, then increment it to `"demo_2"` or use whatever name/naming scheme you wish.
16. Scroll all the way to the bottom of the `traindemo.py` file. Set the functions to run in the `main()` function (by commenting or uncommenting some of the code lines to be the same as follows):
    ```
    config = load_config()

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    check_system(config=config)

    sys_prepare()

    # download_all_melspecs_first()

    # dummy_run(config=config)

    # run_hyperparameter_search(config=config)

    run_training(config=config, json_config_path=JSON_CONFIG_PATH)

    # run_retraining(
    #     config=config,
    #     json_config_path=JSON_CONFIG_PATH,
    #     previous_run_id=RUN_ID,
    # )
    ```
17. Run `uv run traindemo.py` or `uv run traindemo.py > training_logs_cnn_demo_1.txt` if you so wish to log to a `.txt` file.
18. Repeat step 15; this time, change `JSON_CONFIG_PATH` to be equal to `"./params_gru_demo.json"` then repeat step 17. If you so choose to  log to a .txt file, pipe it to `training_logs_gru_demo_1.txt`
19. Repeat step 15 again; this time, change `JSON_CONFIG_PATH` to be equal to `"./params_attn_demo.json"` then repeat step 17 again. If you so choose to  log to a .txt file, pipe it to `training_logs_attn_demo_1.txt`
20. Go to your MLflow tracking server UI, you will see three experiments for each of the model variants: CNN, CNN-GRU, and CNN-ATTN. Open each of them and get the `Run ID` of each run of each experiment *manually*. I do not provide a script to do this automatically.
21. Make sure `SEARCH_EXPERIMENT_NAME=` in `.env` in this `/ai` dir is empty or equates to nothing; not even an empty string.
22. Copy `testing_all_metrics.py` and paste as `testing_all_metrics_demo.py` in this `/ai` dir
23. Set global variables:
    ```
    EXPERIMENT_IDENT = "demo_test_1"
    SKIP_TESTING = False
    BACKEND_TYPES = ["CNN", "GRU", "ATTN"]
    RUN_IDS = [
        "whatever Run ID you got here",  # CNN
        "whatever Run ID you got here",  # GRU
        "whatever Run ID you got here",  # ATTN
    ]
    PTH_PATHS = [
        "training_checkpoints/BEST_CNN_model_epoch_X.pth",
        "training_checkpoints/BEST_GRU_model_epoch_Y.pth",
        "training_checkpoints/BEST_ATTN_model_epoch_Z.pth",
    ]

    # Add these two below the `from train import (...)` code block
    LOCAL_TEMP_FOLDER = "tempdemo"
    LOCAL_RAW_FOLDER = "melspec"
    ```
    Note: substitute the Run IDs and the PTH paths with whatever you have. The PTH paths is under the "artifacts" tab when you open a run and the files are under the `training_checkpoints` folder. For X, Y, Z it is also whatever you got in the artifact files under the aforementioned folder. \
    Second note: use whatever name/naming scheme you want for `EXPERIMENT_IDENT`
24. Run `uv run testing_all_metrics_demo.py > testing_logs_demo_1.txt`