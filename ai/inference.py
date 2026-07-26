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
    input_path: str,
    config: TrainingConfig,
    model: MusicModel,
    device,
    tag2idx,
    num_classes,
    split: str = "test",
    custom_input=False,
):
    model.eval()

    manifest: dict = {}

    subfolder = input_path.split("/")[0]
    song_id = input_path.split("/")[1].split(".")[0]

    chunk_dir = Path(
        f"{LOCAL_TEMP_FOLDER}/{LOCAL_RAW_FOLDER}/{split}/{config.melspec_chunk_length}{MELSPEC_DIR_PREFIX}{subfolder}"
    )

    if custom_input:
        pattern = f"{config.melspec_chunk_length}*_{song_id}.npy"
        custom_dir = Path(f"./temp/mock")
        chunk_files = sorted(list(custom_dir.glob(pattern)))
        with open(
            f"./temp/mock/{config.melspec_chunk_length}{song_id}_0.json", "r"
        ) as f:
            manifest = json.loads(f.read())
    else:
        pattern = f"{config.melspec_chunk_length}{song_id}_*.npy"
        chunk_files = sorted(list(chunk_dir.glob(pattern)))
        # Only need to read the 0th JSON because the rest of the chunks have the same information
        with open(
            f"{chunk_dir}/{config.melspec_chunk_length}{song_id}_0.json", "r"
        ) as f:
            manifest = json.loads(f.read())

    # print(">>> MANIFEST", manifest)

    print(f"Inferensi pada berkas musik: {input_path}...", end="\n")
    # start_time = time.time()

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

    # print(f"Inference finished in {time.time() - start_time:.2f}s.")

    return all_song_outputs, all_song_labels


def main(
    json_config_path: str,
    previous_run_id: str,
    artifact_file_name: str,
    input_path: str,
    show_result=True,
    first_layer_only=False,
    skip_global_pooling=False,
    classify=True,
    custom_input=False,
):
    config = load_config()

    # print(f"Unduh checkpoint pada model terbaik: {previous_run_id}")

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    # This downloads to a temporary path managed by MLflow/local temp.
    # From the way this is, there can only be one checkpoint at a time and a new checkpoint
    # download will override the old one
    local_checkpoint_path = mlflow.artifacts.download_artifacts(
        run_id=previous_run_id,
        artifact_path=f"training_checkpoints/{artifact_file_name}.pth",
        dst_path=f"{LOCAL_TEMP_FOLDER}/{LOCAL_MODEL_FOLDER}/inference/{previous_run_id}/{artifact_file_name}.pth",
    )

    print(f"Muat checkpoint pada model terbaik: {local_checkpoint_path}...")

    # map_location='cpu' to load to CPU first before being loaded to device in the end so that
    # PyTorch doesn't "assume" that the model is loaded to the correct device from the getgo
    checkpoint = torch.load(local_checkpoint_path, map_location="cpu")

    # Make model
    params: dict = {}

    with open(json_config_path, "r") as f:
        params = json.loads(f.read())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inferensi pada: {device}")

    tag2idx, vocab = build_vocabulary(config.training_metadata_path)

    print("Pelabelan:")
    vocab_print = []
    for v in vocab:
        vocab_print.append(v.replace("mood/theme---", ""))
    print(vocab_print)
    print()

    num_classes = len(vocab)
    frontend = Frontend(config=config)

    # Dummy tensor to dynamically get frontend output shape
    dummy_input = torch.zeros(1, config.melspec_time_step, N_MEL_BANDS)
    frontend_out = frontend(dummy_input)

    if params["backend_type"] == "CNN":
        backend = CNNBackend(
            input_shape=frontend_out.shape,
            first_layer_only=first_layer_only,
            skip_global_pooling=skip_global_pooling,
        )
        backend_out_features = CNN_N_FILTERS * 2
    elif params["backend_type"] == "GRU":
        backend = GRUBackend(
            input_shape=frontend_out.shape,
            first_layer_only=first_layer_only,
            skip_global_pooling=skip_global_pooling,
        )
        backend_out_features = GRU_HIDDEN_SIZE * 2
    elif params["backend_type"] == "ATTN":
        backend = AttentionBackend(
            input_shape=frontend_out.shape,
            project=True,
            d_model=ATTN_LIN_PROJ,
            n_heads=ATTN_N_HEADS,
            first_layer_only=first_layer_only,
            skip_global_pooling=skip_global_pooling,
        )
        backend_out_features = ATTN_LIN_PROJ * 2

    classifier = Classifier(
        in_features=backend_out_features,
        hidden_units=HIDDEN_UNITS,
        out_features=num_classes,
    )

    model = MusicModel(frontend, backend, classifier, classify=classify).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # type: ignore
    print(f"Pemuatan model berhasil. Epoch: {checkpoint.get('epoch', 'Unknown')}\n")  # type: ignore

    aso, asl = _infer(
        input_path=input_path,
        config=config,
        model=model,
        device=device,
        tag2idx=tag2idx,
        num_classes=num_classes,
        custom_input=custom_input,
    )

    if show_result:
        print(aso, end="\n")
        print(asl, end="\n")

        for i, a in enumerate(asl[0]):
            if a > 0:
                print(
                    f"Label {vocab[i].replace("mood/theme---", "")} diprediksi: {aso[0][i]}, sesungguhnya: {a}"
                )

    return aso, asl


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
        # json_config_path="./params_gru_f2.json",
        # json_config_path="./params_attn_f2.json",
        previous_run_id="0d0be9c5ee5e44b8a2fab97582a6ff7b",  # CNN
        # previous_run_id="17e64b9ee037445d82075c4abd671dec",  # GRU
        # previous_run_id="4e7c7f09ba9b4490bd4c8008ffead321",  # ATNN
        artifact_file_name="BEST_CNN_model_epoch_17",
        # artifact_file_name="BEST_GRU_model_epoch_7",
        # artifact_file_name="BEST_ATTN_model_epoch_13",
        # path_val="00/13400.mp3",
        # path_val="74/1158174.mp3",
        input_path="94/1353294.mp3",
    )
