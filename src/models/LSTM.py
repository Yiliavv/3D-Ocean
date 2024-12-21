from typing import Any

import torch
from torch import nn, manual_seed, optim, tensor, mean
from lightning import LightningModule

from src.models.model import ssim
from src.utils.log import Log


class ConvLSTMCell(LightningModule):
    """
    Initialize ConvLSTM cell.

    Parameters
    ----------
    input_dim: int
        Number of channels of input tensor.
    hidden_dim: int
        Number of channels of hidden state.
    kernel_size: (int, int)
        Size of the convolutional kernel.
    bias: bool
        Whether to add the bias.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        manual_seed(1)

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,  #
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding='same',
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(LightningModule):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        bias: Bias or no bias in Convolution
        # Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers.
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> conv_lstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = conv_lstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        self.input_dim = input_dim
        self.hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        self.kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        if not len(self.kernel_size) == len(self.hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.num_layers = num_layers
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        self.fc = nn.Sequential(nn.Sigmoid(), nn.Linear(hidden_dim * 20 * 20, 1 * 20 * 20, bias=True))
        self.nor = nn.BatchNorm2d(20)

    def forward(self, x):
        Log.d(x.shape)

        # b - batch_size: Number of images in each batch
        # t - seq_len: Number of images in each sequence
        # c - Number of channels in the input
        # h - Height of the image
        # w - Width of the image
        b, t, c, h, w = x.shape
        Log.d(f"b: {b}, t: {t}, c: {c}, h: {h}, w: {w}")
        hidden_state = self._init_hidden(batch_size=b,
                                         image_size=(h, w))

        # 所有层的输出
        layer_output_list = []
        # 通过最后一个神经元的状态
        last_state_list = []

        seq_len = x.size(1)
        # 当前层的输入，从前一层继承输出
        cur_layer_input = x

        # 每一层进行计算
        for layer_idx in range(self.num_layers):

            # 保存每一层的输出
            h, c = hidden_state[layer_idx]
            Log.d(f"layer: {layer_idx}, h: {h.shape}, c: {c.shape}")
            output_inner = []
            # 每一个时间步进行计算
            for t in range(seq_len):
                cell = self.cell_list[layer_idx]
                h, c = cell(input_tensor=cur_layer_input[:, t, :, :, :],
                            cur_state=[h, c])
                output_inner.append(h)

            Log.d(f"out_h: {h.shape}, out_c: {c.shape}")

            # 一个层输出的时间序列
            layer_output = torch.stack(output_inner, dim=1)
            Log.d(f"layer_output: {layer_output.shape}")

            # 更新当前层的输入
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # 获取最后的输出
        output_seq = layer_output_list[-1]
        output_h = last_state_list[-1][0]
        output_c = last_state_list[-1][1]

        output = output_c
        output = self.nor(output)
        output = output.view(b, -1)
        output = self.fc(output)
        output = output.view(b, 1, 20, 20)

        Log.d(f"output_seq: {output_seq.shape}, output_h: {output_h.shape}, output_c: {output_c.shape}")

        # 返回最后一个时间步的输出
        return output

    def training_step(self, batch, batch_index):
        # 训练循环
        # x - 输入的时间序列
        # y - 输出的时间序列
        x, y = batch

        # b - batch_size: Number of images in each batch
        # t - seq_len: Number of images in each sequence
        # c - Number of channels in the input
        # h - Height of the image
        # w - Width of the image
        b, t, c, h, w = x.shape

        Log.d(batch_index)

        output = self(x)

        loss = nn.functional.mse_loss(output, y)
        self.log('loss', loss)
        print(f" --- loss: {loss}")

        return loss

    def validation_step(self, batch, batch_index):
        return self.training_step(batch, batch_index)

    def configure_optimizers(self):
        a_opt = optim.Adam(self.parameters(), lr=1e-4)
        return a_opt

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
