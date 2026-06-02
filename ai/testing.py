import os
import time
import torch
import numpy as np
import pandas as pd
import traceback
import mlflow
import mlflow.artifacts
import re
import download_melspecs as dl
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, average_precision_score
from dataclasses import dataclass

# import mlflow.pytorch
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

EXPERIMENT_IDENT = "v3_TEST_15s"
SKIP_TESTING = True  # to only show the stats
# SKIP_TESTING = False
BACKEND_TYPES = ["CNN", "GRU", "ATTN"]
RUN_IDS = [
    "0c0d66e93e524c959f92f82156a97011",  # CNN
    "0b159a90989548dba96a7176e0721050",  # GRU
    "91c6d578c3224667a208a9ce24612db6",  # ATTN
]
PTH_PATHS = [
    "training_checkpoints/CNN_model_epoch_0.pth",
    "training_checkpoints/BEST_GRU_model_epoch_7.pth",
    "training_checkpoints/BEST_ATTN_model_epoch_8.pth",
]
# BACKEND_TYPES = ["ATTN"]

# Import from existing train.py
from train import (
    MusicModel,
    Frontend,
    CNNBackend,
    GRUBackend,
    AttentionBackend,
    Classifier,
    build_vocabulary,
    bour_weighted_bce_loss,
    _calculate_dataset_relative_frequencies,
    load_config,
    LOCAL_MODEL_FOLDER,
    N_MEL_BANDS,
    CNN_N_FILTERS,
    GRU_HIDDEN_SIZE,
    ATTN_LIN_PROJ,
    ATTN_N_HEADS,
    HIDDEN_UNITS,
    N_MELSPEC_LENGTHS,
    MELSPEC_PREFIXES,
    MELSPEC_DIR_PREFIX,
    LOCAL_TEMP_FOLDER,
    LOCAL_RAW_FOLDER,
)


@dataclass
class TestingConfig:
    test_metadata_path: str
    training_metadata_path: str
    melspec_chunk_length: str
    melspec_time_step: int
    mlflow_tracking_uri: str
    target_run_id: str
    search_experiment_name: str
    top_n_models: int
    dataset_relative_frequencies: np.ndarray


def load_testing_config() -> TestingConfig:
    load_dotenv()
    try:
        test_csv = os.environ["TESTING_METADATA_CSV"]
        train_csv = os.environ["TRAINING_METADATA_CSV"]
        melspec_chunk_length = MELSPEC_PREFIXES[int(os.environ["MELSPEC_PREFIX_INDEX"])]
        melspec_time_step = N_MELSPEC_LENGTHS[int(os.environ["MELSPEC_PREFIX_INDEX"])]

        mlflow_uri = os.environ["MLFLOW_TRACKING_URI"]

        # Can either use a specific RUN ID, or search programmatically
        target_run_id = os.environ.get("TEST_RUN_ID", "").strip()
        search_experiment_name = os.environ.get("SEARCH_EXPERIMENT_NAME", "").strip()
        top_n_models = int(os.environ.get("TOP_N_MODELS", "3"))

        if not target_run_id and not search_experiment_name:
            raise Exception(
                "You must provide either TEST_RUN_ID or SEARCH_EXPERIMENT_NAME in .env"
            )

        dataset_relative_frequencies = _calculate_dataset_relative_frequencies(
            metadata_path=test_csv
        )
    except KeyError as e:
        traceback.print_exc()
        raise Exception(f"Missing env var: {e.args[0]}")

    return TestingConfig(
        test_metadata_path=test_csv,
        training_metadata_path=train_csv,
        melspec_chunk_length=melspec_chunk_length,
        melspec_time_step=melspec_time_step,
        mlflow_tracking_uri=mlflow_uri,
        target_run_id=target_run_id,
        search_experiment_name=search_experiment_name,
        top_n_models=top_n_models,
        dataset_relative_frequencies=dataset_relative_frequencies,
    )


def instantiate_model_from_run(
    config: TestingConfig, run, num_classes: int, device: torch.device
):
    """Fetches MLflow run metadata, deduces backend, and initializes model architecture"""
    backend_type = run.data.params.get("backend_type", "CNN")
    print(f"Detected backend type '{backend_type}' from MLflow Run {run.info.run_id}")

    # Build schema
    training_config = load_config()  # load from train script because it's easiest
    frontend = Frontend(config=training_config)
    dummy_input = torch.zeros(1, config.melspec_time_step, N_MEL_BANDS)
    frontend_out = frontend(dummy_input)

    if backend_type == "CNN":
        backend = CNNBackend(input_shape=frontend_out.shape)
        backend_out_features = CNN_N_FILTERS * 2
    elif backend_type == "GRU":
        backend = GRUBackend(input_shape=frontend_out.shape)
        backend_out_features = GRU_HIDDEN_SIZE * 2
    elif backend_type == "ATTN":
        backend = AttentionBackend(
            input_shape=frontend_out.shape,
            project=True,
            d_model=ATTN_LIN_PROJ,
            n_heads=ATTN_N_HEADS,
        )
        backend_out_features = ATTN_LIN_PROJ * 2
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    classifier = Classifier(
        in_features=backend_out_features,
        hidden_units=HIDDEN_UNITS,
        out_features=num_classes,
    )
    return MusicModel(frontend, backend, classifier).to(device)


def extract_epoch_from_filename(filename: str) -> int:
    """Helper to find the highest epoch if multiple checkpoints exist."""
    match = re.search(r"epoch_(\d+)", filename)
    return int(match.group(1)) if match else -1


def evaluate_run(
    config: TestingConfig, run, tag2idx, vocab, num_classes, device, pth_path=None
):
    """Encapsulated testing logic for a single MLflow Run"""
    run_id = run.info.run_id
    print(f"\n{'='*80}\n>>> Testing Run ID: {run_id}\n{'='*80}")

    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)

    # Gather all .pth files to find the latest epoch
    pth_candidates = []

    def search_artifacts(art_list, path=""):
        for art in art_list:
            if art.is_dir:
                sub_arts = client.list_artifacts(run_id, path=art.path)
                search_artifacts(sub_arts, art.path)
            elif art.path.endswith(".pth"):
                pth_candidates.append(art.path)

    search_artifacts(artifacts)

    if not pth_candidates:
        print(
            f"Skipping Run {run_id}: Could not find a .pth checkpoint in MLflow artifacts!"
        )
        return

    # Sort candidates by epoch (highest epoch last) or fallback to first found if no epoch in string
    pth_candidates.sort(key=extract_epoch_from_filename)
    best_pth_artifact_path = pth_candidates[-1]

    if pth_path is not None:
        best_pth_artifact_path = pth_path

    print(f"Downloading checkpoint: {best_pth_artifact_path}")
    local_checkpoint_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=best_pth_artifact_path,
        dst_path=f"{LOCAL_TEMP_FOLDER}/{LOCAL_MODEL_FOLDER}/{run_id}/downloaded_test_checkpoints",
    )

    # 1. Instantiate model and load weights
    model = instantiate_model_from_run(config, run, num_classes, device)
    checkpoint = torch.load(local_checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Handle legacy keys in attention back-end
    legacy_keys = []
    for k in state_dict.keys():
        if "backend.pos_encoder.penc." in k:
            legacy_keys.append(k)

    for old_k in legacy_keys:
        new_k = old_k.replace(
            "backend.pos_encoder.penc.", "backend.pos_encoder.posenc."
        )
        state_dict[new_k] = state_dict.pop(old_k)
        print(f"Migrated legacy state_dict key: '{old_k}' -> '{new_k}'")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Iterate test dataset sequentially (no shuffling and no batching)
    df = pd.read_csv(config.test_metadata_path)
    all_song_outputs = []
    all_song_labels = []

    print(f"Starting unweighted chunk-averaged inference on {len(df)} tracks...")
    start_time = time.time()

    with torch.no_grad():
        for _, row in df.iterrows():
            path_val = row["PATH"]
            subfolder = path_val.split("/")[0]
            song_id = path_val.split("/")[1].split(".")[0]

            chunk_dir = Path(
                f"{LOCAL_TEMP_FOLDER}/{LOCAL_RAW_FOLDER}/test/{config.melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}"
            )
            pattern = f"{config.melspec_chunk_length}{song_id}_*.npy"
            chunk_files = sorted(list(chunk_dir.glob(pattern)))

            if not chunk_files:
                # Silently skip missing test chunks to avoid log spam
                continue

            chunks_tensors = [
                torch.from_numpy(np.load(cf).transpose()).float() for cf in chunk_files
            ]
            batch_melspecs = torch.stack(chunks_tensors).to(device)

            # Target Labels
            tags = row["TAGS"].split("+")
            label_tensor = torch.zeros(num_classes, dtype=torch.float32)
            for t in tags:
                if t in tag2idx:
                    label_tensor[tag2idx[t]] = 1.0

            # Forward pass: shape [num_chunks, 56]
            outputs = model(batch_melspecs)

            # Average inference outputs over chunks to get a song-level prediction [56]
            song_output = torch.mean(outputs, dim=0)

            all_song_outputs.append(song_output.cpu().numpy())
            all_song_labels.append(label_tensor.numpy())

    if len(all_song_outputs) == 0:
        print("No valid test files were evaluated. Skipping metrics generation.")
        return

    # 3. Format overall results to specified sizes (N_SONGS, 56)
    all_song_outputs = np.vstack(all_song_outputs)
    all_song_labels = np.vstack(all_song_labels)

    # Calculate testing split Loss using Bour's logic
    outputs_t = torch.from_numpy(all_song_outputs).to(device)
    labels_t = torch.from_numpy(all_song_labels).to(device)

    overall_test_loss = bour_weighted_bce_loss(
        outputs_t, labels_t, config.dataset_relative_frequencies
    )
    overall_test_loss_val = overall_test_loss.item()

    # Std dev of testing loss
    pt = torch.from_numpy(config.dataset_relative_frequencies).to(device)
    w_pos, w_neg = 2.0 / (1.0 + pt), (2.0 * pt) / (1.0 + pt)
    eps = 10e-12
    out_clamp = torch.clamp(outputs_t, min=eps, max=1.0 - eps)
    t1 = w_pos * labels_t * torch.clamp(torch.log(out_clamp), min=-100)
    t2 = w_neg * (1.0 - labels_t) * torch.clamp(torch.log(1.0 - out_clamp), min=-100)
    per_song_losses = -torch.sum(t1 + t2, dim=1) / num_classes  # [N_SONGS]
    std_test_loss = float(torch.std(per_song_losses).item())

    # 4. AUC
    pr_aucs = []
    roc_aucs = []
    test_metrics = {
        "test_loss_avg": overall_test_loss_val,
        "test_loss_std": std_test_loss,
    }

    # Evaluate each class individually (OvR)
    for i, label_name in enumerate(vocab):
        y_true = all_song_labels[:, i]
        y_score = all_song_outputs[:, i]

        if np.isnan(y_score).any() or len(np.unique(y_true)) <= 1:
            pr_auc, roc_auc = 0.0, 0.5
        else:
            pr_auc = average_precision_score(y_true, y_score)
            roc_auc = roc_auc_score(y_true, y_score)

        pr_aucs.append(pr_auc)
        roc_aucs.append(roc_auc)

        test_metrics[f"test_pr_auc_{label_name}"] = float(pr_auc)
        test_metrics[f"test_roc_auc_{label_name}"] = float(roc_auc)

    # Macro aggregates
    test_metrics["test_macro_pr_auc"] = float(np.mean(pr_aucs))
    test_metrics["test_macro_roc_auc"] = float(np.mean(roc_aucs))

    print(f"Testing finished in {time.time() - start_time:.2f}s.")
    print(f"Overall test avg loss: {overall_test_loss_val:.8f}")
    print(f"Test macro PR-AUC: {test_metrics['test_macro_pr_auc']:.8f}")
    print(f"Test macro ROC-AUC: {test_metrics['test_macro_roc_auc']:.8f}")

    # 5. Log to MLflow
    testing_exp_name = f"{EXPERIMENT_IDENT}_chunk_music_model"
    mlflow.set_experiment(testing_exp_name)
    dynamic_run_name = f"eval_of_run_{run_id[:8]}"

    with mlflow.start_run(run_name=dynamic_run_name):
        mlflow.log_param("evaluated_run_id", run_id)
        mlflow.log_param("checkpoint_epoch", checkpoint.get("epoch", "Unknown"))
        mlflow.log_param("original_experiment", config.search_experiment_name)
        mlflow.log_metrics(test_metrics)
        print("Logged all testing metrics to MLflow.")


def main():
    config = load_testing_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Environment initialized. System will use device: {device}")

    tag2idx, vocab = build_vocabulary(config.training_metadata_path)
    num_classes = len(vocab)

    runs_to_evaluate = []

    # Programmatic search based on metric
    # TODO: metric filter is hardcoded
    if config.target_run_id:
        print(f"Specific TEST_RUN_ID provided. Fetching Run: {config.target_run_id}")
        run = client.get_run(config.target_run_id)
        runs_to_evaluate.append(run)
    elif config.search_experiment_name == "":
        print(f"Fetching models from hardcoded globals..")
        for run_id in RUN_IDS:
            run = client.get_run(run_id=run_id)
            runs_to_evaluate.append(run)
    else:
        print(f"Programmatic search for experiment: '{config.search_experiment_name}'")
        experiment = client.get_experiment_by_name(config.search_experiment_name)
        if not experiment:
            raise ValueError(
                f"Could not find experiment '{config.search_experiment_name}' in MLflow!"
            )

        print(
            f"Fetching Top {config.top_n_models} models ordered by hardcoded filter..."
        )
        for bt in BACKEND_TYPES:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f'params.backend_type = "{bt}" and params.classifier_dropout = "0.0"',
                order_by=["metrics.val_macro_pr_auc DESC"],
                max_results=config.top_n_models,
            )

            if not runs:
                print("No runs found for this experiment.")
                return

            runs_to_evaluate.extend(runs)

    # Quick printout of what models were picked up before running the real code
    for i, r in enumerate(runs_to_evaluate):
        pr_auc = r.data.metrics.get("val_macro_pr_auc", "N/A")
        backend_type = r.data.params.get("backend_type", "N/A")

        backend_dropout = float(r.data.params.get("backend_dropout", 0.0))
        learning_rate = float(r.data.params.get("learning_rate", 0.0))
        epochs = int(r.data.params.get("epochs", 0))

        print(
            f"\t{i+1}. Run ID: {r.info.run_id} | backend type: {backend_type} | val macro PR-AUC: {pr_auc}"
        )
        print(
            f"\t   Run name: {r.data.tags.get("mlflow.runName", "N/A")} | BE dropout: {backend_dropout} | LR: {learning_rate} | epoch: {epochs}"
        )

    # Evaluate each collected run sequentially
    if SKIP_TESTING:
        print(">>> SKIP_TESTING is True.")
        return
    else:
        # Download the test split
        # dl.main(split="test")
        pass

    if len(PTH_PATHS) > 0:
        for i, run in enumerate(runs_to_evaluate):
            evaluate_run(
                config=config,
                run=run,
                tag2idx=tag2idx,
                vocab=vocab,
                num_classes=num_classes,
                device=device,
                pth_path=PTH_PATHS[i],
            )
    else:
        for run in runs_to_evaluate:
            evaluate_run(
                config=config,
                run=run,
                tag2idx=tag2idx,
                vocab=vocab,
                num_classes=num_classes,
                device=device,
            )


if __name__ == "__main__":
    main()
