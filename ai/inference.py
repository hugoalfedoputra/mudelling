import time
import torch
import numpy as np
import traceback
import time
import torch
import json
import mlflow
import mlflow.artifacts
from pathlib import Path
from train import (
    TrainingConfig,
    MusicModel,
    Frontend,
    CNNBackend,
    GRUBackend,
    AttentionBackend,
    Classifier,
    build_vocabulary,
    load_config,
    N_MEL_BANDS,
    CNN_N_FILTERS,
    GRU_HIDDEN_SIZE,
    HIDDEN_UNITS,
    ATTN_N_HEADS,
    ATTN_LIN_PROJ,
    MELSPEC_DIR_PREFIX,
    LOCAL_TEMP_FOLDER,
    LOCAL_RAW_FOLDER,
    LOCAL_MODEL_FOLDER,
)


def _infer(
    path_val: str,
    config: TrainingConfig,
    model: MusicModel,
    device,
    tag2idx,
    num_classes,
    split: str = "test",
):
    model.eval()

    manifest: dict = {}

    subfolder = path_val.split("/")[0]
    song_id = path_val.split("/")[1].split(".")[0]

    chunk_dir = Path(
        f"{LOCAL_TEMP_FOLDER}/{LOCAL_RAW_FOLDER}/{split}/{config.melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}"
    )
    pattern = f"{config.melspec_chunk_length}{song_id}_*.npy"
    chunk_files = sorted(list(chunk_dir.glob(pattern)))

    with open(f"{chunk_dir}/{config.melspec_chunk_length}{song_id}_0.json", "r") as f:
        manifest = json.loads(f.read())

    print(">>> MANIFEST", manifest)

    print(f"Starting unweighted chunk-averaged inference on {path_val}...")
    start_time = time.time()

    all_song_outputs = []
    all_song_labels = []

    with torch.no_grad():
        chunks_tensors = [
            torch.from_numpy(np.load(cf).transpose()).float() for cf in chunk_files
        ]
        batch_melspecs = torch.stack(chunks_tensors).to(device)

        # Target labels
        tags: list[str] = manifest["tags"]
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
        traceback.print_exc()
        raise Exception("No file was evaluated. Skipping metrics generation...")

    all_song_outputs = np.vstack(all_song_outputs)
    all_song_labels = np.vstack(all_song_labels)

    print(f"Inference finished in {time.time() - start_time:.2f}s.")

    return all_song_outputs, all_song_labels


def main(
    json_config_path: str, previous_run_id: str, artifact_file_name: str, path_val: str
):
    config = load_config()

    print(f"Downloading checkpoint artifact from Run ID: {previous_run_id}...")

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    # This downloads to a temporary path managed by MLflow/local temp.
    # From the way this is, there can only be one checkpoint at a time and a new checkpoint
    # download will override the old one
    local_checkpoint_path = mlflow.artifacts.download_artifacts(
        run_id=previous_run_id,
        artifact_path=f"training_checkpoints/{artifact_file_name}.pth",
        dst_path=f"{LOCAL_TEMP_FOLDER}/{LOCAL_MODEL_FOLDER}/inference/{previous_run_id}/{artifact_file_name}.pth",
    )

    print(
        f"After download: loading local PyTorch checkpoint from {local_checkpoint_path}..."
    )

    # map_location='cpu' to load to CPU first before being loaded to device in the end so that
    # PyTorch doesn't "assume" that the model is loaded to the correct device from the getgo
    checkpoint = torch.load(local_checkpoint_path, map_location="cpu")

    # Make model
    params: dict = {}

    with open(json_config_path, "r") as f:
        params = json.loads(f.read())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inferencing on: {device}")

    tag2idx, vocab = build_vocabulary(config.training_metadata_path)
    num_classes = len(vocab)
    frontend = Frontend(config=config)

    # Dummy tensor to dynamically get frontend output shape
    dummy_input = torch.zeros(1, config.melspec_time_step, N_MEL_BANDS)
    frontend_out = frontend(dummy_input)

    if params["backend_type"] == "CNN":
        backend = CNNBackend(input_shape=frontend_out.shape)
        backend_out_features = CNN_N_FILTERS * 2
    elif params["backend_type"] == "GRU":
        backend = GRUBackend(input_shape=frontend_out.shape)
        backend_out_features = GRU_HIDDEN_SIZE * 2
    elif params["backend_type"] == "ATTN":
        backend = AttentionBackend(
            input_shape=frontend_out.shape,
            project=True,
            d_model=ATTN_LIN_PROJ,
            n_heads=ATTN_N_HEADS,
        )
        backend_out_features = ATTN_LIN_PROJ * 2

    classifier = Classifier(
        in_features=backend_out_features,
        hidden_units=HIDDEN_UNITS,
        out_features=num_classes,
    )

    model = MusicModel(frontend, backend, classifier).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # type: ignore
    print(f"Successfully restored model and optimizer states from prior epoch: {checkpoint.get('epoch', 'Unknown')}\n")  # type: ignore

    aso, asl = _infer(
        path_val=path_val,
        config=config,
        model=model,
        device=device,
        tag2idx=tag2idx,
        num_classes=num_classes,
    )

    print(aso, end="\n")
    print(asl, end="\n")


if __name__ == "__main__":
    # =================================================================================
    #
    # CHANGE FN PARAMS BELOW
    # CHANGE FN PARAMS BELOW
    # CHANGE FN PARAMS BELOW
    # CHANGE FN PARAMS BELOW
    # CHANGE FN PARAMS BELOW
    # =================================================================================

    main(
        json_config_path="./params_cnn_f2.json",
        previous_run_id="0d0be9c5ee5e44b8a2fab97582a6ff7b",
        artifact_file_name="BEST_CNN_model_epoch_17",
        path_val="00/13400.mp3",  # Music file to inference
    )
