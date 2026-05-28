import torch
import torch.nn as nn
import torch.nn.functional as F


class Frontend(nn.Module):
    def __init__(self, num_filt=16):
        super(Frontend, self).__init__()

        self.num_filt = num_filt
        y_input = 96

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filt,
            kernel_size=(7, int(0.9 * y_input)),
            padding=(3, 0),
        )
        self.bn_conv1 = nn.BatchNorm2d(num_filt)

        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filt * 2,
            kernel_size=(3, int(0.9 * y_input)),
            padding=(1, 0),
        )
        self.bn_conv2 = nn.BatchNorm2d(num_filt * 2)

        self.conv3 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filt * 4,
            kernel_size=(1, int(0.9 * y_input)),
            padding=(0, 0),
        )
        self.bn_conv3 = nn.BatchNorm2d(num_filt * 4)

        self.conv4 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filt,
            kernel_size=(7, int(0.4 * y_input)),
            padding=(3, 0),
        )
        self.bn_conv4 = nn.BatchNorm2d(num_filt)

        self.conv5 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filt * 2,
            kernel_size=(3, int(0.4 * y_input)),
            padding=(1, 0),
        )
        self.bn_conv5 = nn.BatchNorm2d(num_filt * 2)

        self.conv6 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filt * 4,
            kernel_size=(1, int(0.4 * y_input)),
            padding=(0, 0),
        )
        self.bn_conv6 = nn.BatchNorm2d(num_filt * 4)

        self.conv7 = nn.Conv1d(
            in_channels=1, out_channels=num_filt, kernel_size=165, padding="same"
        )
        self.bn_conv7 = nn.BatchNorm1d(num_filt)

        self.conv8 = nn.Conv1d(
            in_channels=1, out_channels=num_filt * 2, kernel_size=128, padding="same"
        )
        self.bn_conv8 = nn.BatchNorm1d(num_filt * 2)

        self.conv9 = nn.Conv1d(
            in_channels=1, out_channels=num_filt * 4, kernel_size=64, padding="same"
        )
        self.bn_conv9 = nn.BatchNorm1d(num_filt * 4)

        self.conv10 = nn.Conv1d(
            in_channels=1, out_channels=num_filt * 8, kernel_size=32, padding="same"
        )
        self.bn_conv10 = nn.BatchNorm1d(num_filt * 8)

    def forward(self, x):
        input_layer = x.unsqueeze(1)

        y_input = input_layer.shape[3]

        def run_2d_branch(conv_layer, bn_layer, input):
            out = F.relu(bn_layer(conv_layer(input)))
            freq_dim = out.shape[3]
            out = F.max_pool2d(out, kernel_size=(1, freq_dim), stride=(1, freq_dim))
            return out.squeeze(3)

        p1 = run_2d_branch(self.conv1, self.bn_conv1, input_layer)
        p2 = run_2d_branch(self.conv2, self.bn_conv2, input_layer)
        p3 = run_2d_branch(self.conv3, self.bn_conv3, input_layer)
        p4 = run_2d_branch(self.conv4, self.bn_conv4, input_layer)
        p5 = run_2d_branch(self.conv5, self.bn_conv5, input_layer)
        p6 = run_2d_branch(self.conv6, self.bn_conv6, input_layer)

        pool_avg = F.avg_pool2d(
            input_layer, kernel_size=(1, y_input), stride=(1, y_input)
        )

        pool_rs = pool_avg.squeeze(3)

        def run_1d_branch(conv_layer, bn_layer, input):
            out = conv_layer(input)
            out = F.relu(bn_layer(out))
            return out

        out7 = run_1d_branch(self.conv7, self.bn_conv7, pool_rs)
        out8 = run_1d_branch(self.conv8, self.bn_conv8, pool_rs)
        out9 = run_1d_branch(self.conv9, self.bn_conv9, pool_rs)
        out10 = run_1d_branch(self.conv10, self.bn_conv10, pool_rs)

        pool = torch.cat([p1, p2, p3, p4, p5, p6, out7, out8, out9, out10], dim=1)

        return pool.permute(0, 2, 1).unsqueeze(3)


class CNNBackend(nn.Module):
    def __init__(self, input_shape, p_dropout=0.1):
        super(CNNBackend, self).__init__()
        C = input_shape[2]

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, C),
            padding=(0, 0),
        )
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p=p_dropout)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(7, 1),
            padding=(3, 0),
        )
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(p=p_dropout)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(7, 1),
            padding=(3, 0),
        )
        self.bn_conv3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        out = F.relu(self.bn_conv1(self.conv1(x)))
        out = self.dropout1(out)

        out = F.relu(self.bn_conv2(self.conv2(out)))
        out = self.dropout2(out)

        out = F.max_pool2d(out, kernel_size=(2, 1), stride=(2, 1))

        out = F.relu(self.bn_conv3(self.conv3(out)))

        out = out.squeeze(3)

        out_mean = F.adaptive_avg_pool1d(out, 1).squeeze(2)
        out_max = F.adaptive_max_pool1d(out, 1).squeeze(2)

        out_cat = torch.cat([out_mean, out_max], dim=1)
        out_cat = self.dropout3(out_cat)

        return out_cat


class GRUBackend(nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_size=GRU_HIDDEN_SIZE,
        p_dropout=0.1,
        is_bidirectional=False,
    ):
        super(GRUBackend, self).__init__()
        # inp_shape from Frontend is [Batch, Time, Channels, 1]
        C = input_shape[2]

        if is_bidirectional:
            D = 2
        else:
            D = 1

        self.gru1 = nn.GRU(
            input_size=C,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=is_bidirectional,
        )

        self.ln1 = nn.LayerNorm(normalized_shape=D * hidden_size)
        self.dropout1 = nn.Dropout(p=p_dropout)

        self.gru2 = nn.GRU(
            input_size=D * hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=is_bidirectional,
        )

        self.ln2 = nn.LayerNorm(normalized_shape=D * hidden_size)
        self.dropout2 = nn.Dropout(p=p_dropout)

        self.gru3 = nn.GRU(
            input_size=D * hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=is_bidirectional,
        )

        self.ln3 = nn.LayerNorm(normalized_shape=D * hidden_size)
        self.dropout3 = nn.Dropout(p=p_dropout)

        # Weight initialisation uses defalt uniform dist. but bias set to 0
        nn.init.constant_(torch.Tensor(self.gru1.bias_ih_l0), 0)
        nn.init.constant_(torch.Tensor(self.gru1.bias_hh_l0), 0)

    def forward(self, x):
        # x is[B, T, C, 1]. Squeeze to [B, T, C] to feed as sequence to GRU as it
        # expects input to be 2D or 3D and not 4D (as per error msg during testing)
        x = x.squeeze(3)

        out, _ = self.gru1(x)  # out shape:[B, T, N_FILTERS]
        out = self.ln1(out)
        out = self.dropout1(out)

        out, _ = self.gru2(out)
        out = self.ln2(out)
        out = self.dropout2(out)

        out, _ = self.gru3(out)
        out = self.ln3(out)

        # To perform 1D Adaptive Pooling over time, PyTorch expects[B, Features, Time]
        out = out.permute(0, 2, 1)  # ->[B, N_FILTERS, T]

        out_mean = F.adaptive_avg_pool1d(out, 1).squeeze(2)
        out_max = F.adaptive_max_pool1d(out, 1).squeeze(2)

        out_cat = torch.cat([out_mean, out_max], dim=1)
        out_cat = self.dropout3(out_cat)

        return out_cat
