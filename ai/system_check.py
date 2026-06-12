import train
import os


def check_system(config: train.TrainingConfig):
    import torch

    torch.backends.cudnn.enabled = config.torch_backends_cudnn_enabled
    torch.backends.cudnn.benchmark = config.torch_backends_cudnn_benchmark

    # load_dotenv()

    # Only for AMD GPUs on ROCm, NVIDIA should just work out of the box
    _ = os.environ["HSA_OVERRIDE_GFX_VERSION"]

    # Source of checks: https://medium.com/@yulin_li/commands-for-the-cross-validation-of-pytorch-and-cuda-installation-235c8003b446
    # Will deliberately throw an error so the program is going to terminate early if any of these checks do not pass.

    print("Is CUDA available?", torch.cuda.is_available())
    print("PyTorch version:", torch.__version__)

    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU index: {torch.cuda.current_device()}")
    print(
        f"Name of current GPU {torch.cuda.get_device_name(torch.cuda.current_device())}"
    )

    # Check if CUDA is available, and if so, create a tensor on GPU
    if torch.cuda.is_available():
        x = torch.randn(3, 3)
        x = x.to("cuda")
        print(x)
    else:
        print("CUDA is not available.")

    print("CUDA version (shows 'None' if using ROCm):", torch.version.cuda)

    print("CUDNN enabled:", torch.backends.cudnn.enabled)  # Check if cuDNN is enabled
    print(
        "CUDNN version:", torch.backends.cudnn.version()
    )  # Print the cuDNN version used by PyTorch


def main():

    config = train.load_config()

    check_system(config=config)


if __name__ == "__main__":
    main()
