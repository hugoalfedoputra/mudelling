import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import download_melspecs as dl
from torchinfo import summary
from dotenv import load_dotenv

# =================================================================================================
#
# region Globals
# GLOBALS
# =================================================================================================
CNN_N_FILTERS = 64
GRU_HIDDEN_SIZE = 92
ATTN_LIN_PROJ = 92
ATTN_N_HEADS = 4
N_LAYERS = 3
HIDDEN_UNITS = 200
BATCH_SIZE = 32  # Follow TensorFlow's default


# region Global helpers
def check_system():
    import torch

    load_dotenv()

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


def print_summary(model, input_data):
    summary(
        model,
        input_data=input_data,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )


def download_melspecs():
    dl.main()


# endregion


# region Model helpers
# Helper to initialize weights similarly to tf.contrib.layers.variance_scaling_initializer()
def init_conv_linear_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def get_emb(sin_inp):
    """
    Positional encoding code from tatp22/multidim-positional-encoding)

    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


def count_parameters(model):
    """Helper function to count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dummy_run():
    # Declare config and dummy data
    config = {"setup_params": {"yInput": 96}}

    # Let's pretend this is transposed from (BATCH_SIZE, 96, 469) i.e. actual data shape from the dataloader
    dummy_input = torch.randn(BATCH_SIZE, 469, 96)
    dummy_targets = torch.rand(BATCH_SIZE, 56)  # Random continuous targets for BCE Loss

    # Instantiate shared frontend
    _ = Frontend(config, num_filt=16)

    # The frontend outputs [Batch, Time, 464, 1]
    # The concatenated output of the 10 parallel branches results in exactly
    # 16+32+64+16+32+64+16+32+64+128 = 464 channels (hard-coded for now)

    # Pass input_shape=[16, 469, 464, 1] into the backends
    dummy_frontend_shape = [BATCH_SIZE, 469, 464, 1]

    # Instantiate the 3 model variants
    model_cnn = MusicModel(
        frontend=Frontend(config, num_filt=16),
        backend=CNNBackend(dummy_frontend_shape),
        classifier=Classifier(
            in_features=2 * CNN_N_FILTERS, hidden_units=HIDDEN_UNITS, out_features=56
        ),
    )

    model_gru = MusicModel(
        frontend=Frontend(config, num_filt=16),
        backend=GRUBackend(dummy_frontend_shape, hidden_size=GRU_HIDDEN_SIZE),
        classifier=Classifier(
            in_features=2 * GRU_HIDDEN_SIZE, hidden_units=HIDDEN_UNITS, out_features=56
        ),
    )

    model_attn = MusicModel(
        frontend=Frontend(config, num_filt=16),
        # backend=AttentionBackend(dummy_frontend_shape, nhead=4, num_layers=2),
        backend=AttentionBackend(
            dummy_frontend_shape,
            project=True,
            d_model=ATTN_LIN_PROJ,
            n_heads=ATTN_N_HEADS,
            num_layers=3,
        ),
        # 928 becaues attention takes the 464 channels as is. Though, can be projected to
        # larger # of features if needed
        # classifier=Classifier(in_features=928, hidden_units=HIDDEN_UNITS, out_features=56)
        classifier=Classifier(
            in_features=2 * ATTN_LIN_PROJ, hidden_units=HIDDEN_UNITS, out_features=56
        ),
    )

    # Setup optimization (Adam + BCELoss)
    criterion = nn.BCELoss()
    # Example for CNN only
    optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)

    # Print architecture and parameter counts
    print("=" * 120)
    print(f"CNN model parameters: {count_parameters(model_cnn):,}")
    print(f"RNN model parameters: {count_parameters(model_gru):,}")
    print(f"Self-Attention model parameters: {count_parameters(model_attn):,}")
    print("=" * 120)

    # Perform a dummy forward pass and loss calculation (only for CNN as example)
    model_cnn.train()
    optimizer_cnn.zero_grad()

    outputs = model_cnn(dummy_input)
    loss = criterion(outputs, dummy_targets)
    loss.backward()
    optimizer_cnn.step()

    print(f"Output shape: {outputs.shape}")

    print("\n" + "=" * 120)
    print("Model Summaries")
    print("=" * 120)

    print("\nCNN")
    print_summary(model_cnn, input_data=dummy_input)

    print("\nGRU")
    print_summary(model_gru, input_data=dummy_input)

    print("\nSelf-Attention")
    print_summary(model_attn, input_data=dummy_input)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels

        self.inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        )

        try:
            self.register_buffer("inv_freq", self.inv_freq)
        except Exception as e:
            print(e.args)

        try:
            self.register_buffer("cached_penc", None, persistent=False)
        except Exception as e:
            print(e.args)

        self.dtype_override = dtype_override

    def forward(self, tensor):
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros(
            (x, self.channels),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class Summer(nn.Module):
    def __init__(self, penc):
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor):
        penc = self.penc(tensor)
        assert (
            tensor.size() == penc.size()
        ), f"The original tensor size {tensor.size()} and the positional encoding tensor size {penc.size()} must match!"
        return tensor + penc.to(tensor.device)


# endregion


# region Frontend
class Frontend(nn.Module):
    """
    Paper: Pons, J., Nieto, O., Prockup, M., Schmidt, E., Ehmann, A. and Serra, X., 2018. End-to-end
    learning for music audio tagging at scale. https://doi.org/10.48550/arXiv.1711.02520.

    Source: https://github.com/jordipons/music-audio-tagging-at-scale-models/blob/master/models.py is
    referred to as "source" code. Any other mentions of "source" code will specify which code it is
    possessed by.

    Converted to PyTorch.

    - 'num_filt': multiplicative factor that controls the number of filters for every filter shape.
    The paper's source code uses 16 (see: footnote no.5 and 6 on p.4 Pons et al., 2018).

    Some differences:
    - Uses He's normal initialisation, which is different than the truncated normal VarianceScaling
    initialisation used in the source code. The difference is that in this implementation the std dev
    is not truncated (or have std dev divided furthermore by ~.879 in TensorFlow's source code)
    - BN is calculated BEFORE the activation function, while in the source code it is activated then
    pushed to BN. My thesis follows the original author of BN's intention to use it before the activation
    function. This might change if aiming for direct conversion from source code.
    """

    def __init__(self, config, num_filt=16):
        super(Frontend, self).__init__()

        self.num_filt = num_filt
        y_input = config["setup_params"]["yInput"]

        # [TIMBRE] filter shape 1: 7x0.9f
        # Input padding in TF was [[0,0], [3,3], [0,0], [0,0]] (Time padding).
        # Handle this via the padding argument in Conv2d (pad_h, pad_w).
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filt,
            kernel_size=(7, int(0.9 * y_input)),
            padding=(3, 0),
        )  # Padding time by 3 (same), freq by 0 (valid)
        self.bn_conv1 = nn.BatchNorm2d(num_filt)

        # [TIMBRE] filter shape 2: 3x0.9f
        # Input padding in TF was [[0,0], [1,1]...]
        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filt * 2,
            kernel_size=(3, int(0.9 * y_input)),
            padding=(1, 0),
        )
        self.bn_conv2 = nn.BatchNorm2d(num_filt * 2)

        # [TIMBRE] filter shape 3: 1x0.9f
        # No padding needed for 1x... kernel to match TF logic
        self.conv3 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filt * 4,
            kernel_size=(1, int(0.9 * y_input)),
            padding=(0, 0),
        )
        self.bn_conv3 = nn.BatchNorm2d(num_filt * 4)

        # [TIMBRE] filter shape 4: 7x0.4f
        self.conv4 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filt,
            kernel_size=(7, int(0.4 * y_input)),
            padding=(3, 0),
        )
        self.bn_conv4 = nn.BatchNorm2d(num_filt)

        # [TIMBRE] filter shape 5: 3x0.4f
        self.conv5 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filt * 2,
            kernel_size=(3, int(0.4 * y_input)),
            padding=(1, 0),
        )
        self.bn_conv5 = nn.BatchNorm2d(num_filt * 2)

        # [TIMBRE] filter shape 6: 1x0.4f
        self.conv6 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filt * 4,
            kernel_size=(1, int(0.4 * y_input)),
            padding=(0, 0),
        )
        self.bn_conv6 = nn.BatchNorm2d(num_filt * 4)

        # [TEMPORAL-FEATURES]

        # From PyTorch docs: padding='same' pads the input so the output has the
        # shape as the input. However, this mode doesn’t support any stride values
        # other than 1.

        # shape 7: 165x1
        self.conv7 = nn.Conv1d(
            in_channels=1, out_channels=num_filt, kernel_size=165, padding="same"
        )
        self.bn_conv7 = nn.BatchNorm1d(num_filt)

        # shape 8: 128x1
        self.conv8 = nn.Conv1d(
            in_channels=1, out_channels=num_filt * 2, kernel_size=128, padding="same"
        )
        self.bn_conv8 = nn.BatchNorm1d(num_filt * 2)

        # shape 9: 64x1
        self.conv9 = nn.Conv1d(
            in_channels=1, out_channels=num_filt * 4, kernel_size=64, padding="same"
        )
        self.bn_conv9 = nn.BatchNorm1d(num_filt * 4)

        # shape 10: 32x1
        self.conv10 = nn.Conv1d(
            in_channels=1, out_channels=num_filt * 8, kernel_size=32, padding="same"
        )
        self.bn_conv10 = nn.BatchNorm1d(num_filt * 8)

        # Initialize weights
        self.apply(init_conv_linear_weights)

    def forward(self, x):
        """
        - 'x': Input tensor. Expected shape in TF was [Batch, Time, Freq].
               PyTorch will treat this as [Batch, 1, Time, Freq].
        - 'is_training': In PyTorch, uses model.train() or model.eval() instead of passing a boolean.
        """

        # input_layer = tf.expand_dims(x, 3) -> [Batch, Time, Freq, 1]
        # PyTorch Conv2D expects [Batch, Channels, Height(Time), Width(Freq)]
        # Input x is [Batch, Time, Freq]
        input_layer = x.unsqueeze(1)  # [Batch, 1, Time, Freq]

        # y_input for dynamic pooling
        y_input = input_layer.shape[3]

        # Helper for Max Pooling
        def run_2d_branch(conv_layer, bn_layer, input):
            # Conv -> BN -> ReLU
            out = F.relu(bn_layer(conv_layer(input)))

            # pool1 = tf.layers.max_pooling2d(inputs=bn_conv1, pool_size=[1, bn_conv1.shape[2]], ...)
            # PyTorch: Pool over the entire remaining Frequency dimension (dim 3)
            # Pool size: (1, current_freq_width)
            freq_dim = out.shape[3]
            out = F.max_pool2d(out, kernel_size=(1, freq_dim), stride=(1, freq_dim))

            # TF max_pool2d on [B, H, W, C]. pool_size=[1, W]. Result [B, H, 1, C].
            # TF squeeze [2] removes the dimension of size 1. Result [B, H, C].
            # PyTorch shape is [B, C, Time, Freq=1]. Need: [B, C, Time].
            return out.squeeze(3)

        # [TIMBRE] filter shape 1: 7x0.9f
        # Padding is handled in __init__
        p1 = run_2d_branch(self.conv1, self.bn_conv1, input_layer)

        # [TIMBRE] filter shape 2: 3x0.9f
        p2 = run_2d_branch(self.conv2, self.bn_conv2, input_layer)

        # [TIMBRE] filter shape 3: 1x0.9f
        p3 = run_2d_branch(self.conv3, self.bn_conv3, input_layer)

        # [TIMBRE] filter shape 3 (sic - meant 4): 7x0.4f
        p4 = run_2d_branch(self.conv4, self.bn_conv4, input_layer)

        # [TIMBRE] filter shape 5: 3x0.4f
        p5 = run_2d_branch(self.conv5, self.bn_conv5, input_layer)

        # [TIMBRE] filter shape 6: 1x0.4f
        p6 = run_2d_branch(self.conv6, self.bn_conv6, input_layer)

        # [TEMPORAL-FEATURES]
        # Average pooling over all frequency bins
        # pool7 = tf.layers.average_pooling2d(...) -> [B, T, 1, 1] (TF)
        # PyTorch: [B, 1, T, F] -> AvgPool((1, F)) -> [B, 1, T, 1]
        pool_avg = F.avg_pool2d(
            input_layer, kernel_size=(1, y_input), stride=(1, y_input)
        )

        # pool7_rs = tf.squeeze(pool7, [3]) -> TF [B, T, 1]
        # PyTorch needs [B, Channels, Length] for Conv1D.
        # Current [B, 1, T, 1]. Squeeze last dim -> [B, 1, T].
        pool_rs = pool_avg.squeeze(3)

        # Helper for padding in 1D
        def run_1d_branch(conv_layer, bn_layer, input):
            out = conv_layer(input)
            out = F.relu(bn_layer(out))
            return out

        # [TEMPORAL-FEATURES] - filter shape 7: 165x1
        out7 = run_1d_branch(self.conv7, self.bn_conv7, pool_rs)

        # [TEMPORAL-FEATURES] - filter shape 8: 128x1
        out8 = run_1d_branch(self.conv8, self.bn_conv8, pool_rs)

        # [TEMPORAL-FEATURES] - filter shape 9: 64x1
        out9 = run_1d_branch(self.conv9, self.bn_conv9, pool_rs)

        # [TEMPORAL-FEATURES] - filter shape 10: 32x1
        out10 = run_1d_branch(self.conv10, self.bn_conv10, pool_rs)

        # Concatenate all feature maps
        # TF Concat dim 2: [B, T, Channels]
        # PyTorch Current Shapes: [B, C, T]
        # Concatenate on dim 1 (Channels)
        pool = torch.cat([p1, p2, p3, p4, p5, p6, out7, out8, out9, out10], dim=1)

        # Return format
        # TF: return tf.expand_dims(pool, 3) -> [B, T, C, 1]
        # PyTorch: pool is [B, C, T].
        # To strictly match the requested TF shape logic:
        # [B, C, T] -> permute to [B, T, C] -> unsqueeze to [B, T, C, 1]
        return pool.permute(0, 2, 1).unsqueeze(3)


# endregion
# region Backends
class CNNBackend(nn.Module):
    def __init__(self, inp_shape):
        super(CNNBackend, self).__init__()
        # inp_shape from Frontend is [Batch, Time, Channels, 1]
        C = inp_shape[2]

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=CNN_N_FILTERS,
            kernel_size=(7, C),
            padding=(0, 0),
        )
        self.bn_conv1 = nn.BatchNorm2d(CNN_N_FILTERS)

        self.conv2 = nn.Conv2d(
            in_channels=CNN_N_FILTERS,
            out_channels=CNN_N_FILTERS,
            kernel_size=(7, 1),
            padding=(3, 0),
        )
        self.bn_conv2 = nn.BatchNorm2d(CNN_N_FILTERS)

        self.conv3 = nn.Conv2d(
            in_channels=CNN_N_FILTERS,
            out_channels=CNN_N_FILTERS,
            kernel_size=(7, 1),
            padding=(3, 0),
        )
        self.bn_conv3 = nn.BatchNorm2d(CNN_N_FILTERS)

        self.apply(init_conv_linear_weights)

    def forward(self, x):
        # x shape is [B, T, C, 1] (Mimicking TF's NHWC)
        # PyTorch Conv2D expects [B, in_channels, Height, Width]
        x = x.permute(0, 3, 1, 2)  # -> [B, 1, T, C]

        # 1. First CNN Layer
        out = F.relu(self.bn_conv1(self.conv1(x)))  # -> [B, 64, T-6, 1]

        # 2. Second CNN Layer (Time padding preserves T-6)
        out = F.relu(self.bn_conv2(self.conv2(out)))  # -> [B, 64, T-6, 1]

        # 3. Max Pooling (Downsamples Time dimension by half)
        # Matches TF's pool_size=[2, 1], strides=[2, 1]
        out = F.max_pool2d(
            out, kernel_size=(2, 1), stride=(2, 1)
        )  # -> [B, 64, (T-6)//2, 1]

        # 4. Third CNN Layer
        out = F.relu(self.bn_conv3(self.conv3(out)))  # -> [B, 64, (T-6)//2, 1]

        # 5. Global Pooling prep
        # Squeeze the trailing Width dimension of 1 to perform 1D Adaptive Pooling over time
        out = out.squeeze(3)  # -> [B, 64, Time]

        out_mean = F.adaptive_avg_pool1d(out, 1).squeeze(2)  # -> [B, 64]
        out_max = F.adaptive_max_pool1d(out, 1).squeeze(2)  # -> [B, 64]

        # Concatenate on the feature dimension (yielding a flat [B, 128] vector)
        out_cat = torch.cat([out_mean, out_max], dim=1)

        return out_cat


class GRUBackend(nn.Module):
    def __init__(self, input_shape, hidden_size=GRU_HIDDEN_SIZE):
        super(GRUBackend, self).__init__()
        # inp_shape from Frontend is [Batch, Time, Channels, 1]
        C = input_shape[2]

        self.gru1 = nn.GRU(
            input_size=C,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.ln1 = nn.LayerNorm(normalized_shape=hidden_size)

        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.ln2 = nn.LayerNorm(normalized_shape=hidden_size)

        self.gru3 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.ln3 = nn.LayerNorm(normalized_shape=hidden_size)

        # Weight initialisation uses defalt uniform dist. but bias set to 0
        nn.init.constant_(torch.Tensor(self.gru1.bias_ih_l0), 0)
        nn.init.constant_(torch.Tensor(self.gru1.bias_hh_l0), 0)

    def forward(self, x):
        # x is[B, T, C, 1]. Squeeze to [B, T, C] to feed as sequence to GRU as it
        # expects input to be 2D or 3D and not 4D (as per error msg during testing)
        x = x.squeeze(3)

        out, _ = self.gru1(x)  # out shape:[B, T, N_FILTERS]
        out = self.ln1(out)

        out, _ = self.gru2(out)
        out = self.ln2(out)

        out, _ = self.gru3(out)
        out = self.ln3(out)

        # To perform 1D Adaptive Pooling over time, PyTorch expects[B, Features, Time]
        out = out.permute(0, 2, 1)  # ->[B, N_FILTERS, T]

        out_mean = F.adaptive_avg_pool1d(out, 1).squeeze(2)  # -> [B, N_FILTERS]
        out_max = F.adaptive_max_pool1d(out, 1).squeeze(2)  # -> [B, N_FILTERS]

        out_cat = torch.cat([out_mean, out_max], dim=1)  # -> [B, 1024]
        return out_cat


class AttentionBackend(nn.Module):
    def __init__(
        self,
        inp_shape,
        project=False,
        d_model=ATTN_LIN_PROJ,
        n_heads=ATTN_N_HEADS,
        num_layers=N_LAYERS,
    ):
        super(AttentionBackend, self).__init__()
        # inp_shape from Frontend is[Batch, Time, Channels, 1]
        c = inp_shape[2]

        # Linear Projection:
        # Project the large concatenated channel size (c) down to N_FILTERS (d_model)
        # This keeps the parameter count roughly equivalent to the CNN/GRU backends
        self.project = project
        self.projection = nn.Linear(c, d_model)

        # Apply 1D positional encoding to the [Batch, Time, Features] sequence
        if self.project:
            self.pos_encoder = Summer(PositionalEncoding1D(channels=d_model))
        else:
            self.pos_encoder = Summer(PositionalEncoding1D(channels=c))

        # PyTorch TransformerEncoderLayer already includes Layer Normalization (LN)
        if self.project:
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model,
                batch_first=True,
                dropout=0.0,
            )
        else:
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=c,
                nhead=n_heads,
                dim_feedforward=c,
                batch_first=True,
                dropout=0.0,
            )

        self.transformer = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        # Weight initialisation uses default Xavier uniform
        nn.init.constant_(self.encoder_layer.self_attn.in_proj_bias.data, 0)
        nn.init.constant_(self.encoder_layer.self_attn.out_proj.bias.data, 0)

        # self.apply(init_conv_linear_weights)

    def forward(self, x):
        # x is[B, T, C, 1]. Squeeze to [B, T, C]
        x = x.squeeze(3)

        # Project features to dimension of ATTN_LIN_PROJ
        if self.project:
            x = self.projection(x)  # ->[B, T, ATTN_LIN_PROJ]

        # Apply sinusoidal Positional Encoding
        x = self.pos_encoder(
            x
        )  # -> [B, T, 464]; 464 is depending if it's projected or not

        # Run through Self-Attention layers
        out = self.transformer(x)  # ->[B, T, 464]

        # Pool across Time. Transform [B, T, 464] ->[B, 464, T]
        out = out.permute(0, 2, 1)

        out_mean = F.adaptive_avg_pool1d(out, 1).squeeze(2)  # -> [B, 464]
        out_max = F.adaptive_max_pool1d(out, 1).squeeze(2)  # ->[B, 464]

        out_cat = torch.cat([out_mean, out_max], dim=1)  # -> [B, 928]
        return out_cat


# endregion


# region Classifier
class Classifier(nn.Module):
    def __init__(self, in_features=128, hidden_units=200, out_features=56):
        super(Classifier, self).__init__()

        # 1. First Fully Connected layer
        self.fc1 = nn.Linear(in_features, hidden_units)

        # 2. Add the missing Batch Normalization layer
        # Use BatchNorm1d because it operates on feature vectors [Batch, Features]
        self.bn1 = nn.BatchNorm1d(hidden_units)

        # 3. Output layer
        self.out = nn.Linear(hidden_units, out_features)

        # Initialize weights for the Linear layers
        self.apply(init_conv_linear_weights)

    def forward(self, x):
        # The input 'x' comes from the backend, with shape [Batch, in_features]

        # Pass through the first linear layer
        x = self.fc1(x)

        # Apply ReLU activation, then Batch Norm, to exactly match the TF code's order
        # TF sequence: dense(activation='relu') -> batch_normalization
        x = F.relu(x)
        x = self.bn1(x)

        # Pass through the final output layer
        x = self.out(x)

        # Apply sigmoid to get final probabilities
        return torch.sigmoid(x)


# endregion


# region Model
class MusicModel(nn.Module):
    """
    Wrapper class to chain Frontend -> Backend -> Classifier
    """

    def __init__(self, frontend, backend, classifier):
        super(MusicModel, self).__init__()
        self.frontend = frontend
        self.backend = backend
        self.classifier = classifier

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.classifier(x)
        return x


# endregion


# region Dataloader
# endregion


# region Training
# endregion


# region Validation
# endregion


# region Testing
# endregion


# region Main
def main():
    check_system()

    download_melspecs()

    dummy_run()


if __name__ == "__main__":
    main()


# endregion
