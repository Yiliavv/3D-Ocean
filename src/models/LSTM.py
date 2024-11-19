from torch import nn, tensor, cat, split, relu, sigmoid, tanh, zeros, stack, unsqueeze, optim
from torch.utils.data import DataLoader

from src.utils.log import Log


class ConvLSTCell(nn.Module):
    """
    2D ConvLSTM for Ocean SST
    """

    def __init__(self, kernel_size: tuple[int, int], bias: bool):
        super(ConvLSTCell, self).__init__()
        # 海洋数据的通道始终为1
        self.input_channels = 1
        self.hidden_channels = 1

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias).to('cuda')

    def forward(self, inputs, state):
        inputs = unsqueeze(inputs, dim=1)  # 增加一个通道维度

        h_cur, c_cur = state

        conv_inputs = cat([inputs, h_cur], dim=1)
        conv_outputs = self.conv(conv_inputs)

        co_i, co_f, co_g, co_o = split(conv_outputs, self.hidden_channels, dim=1)

        i = relu(co_i)
        f = relu(co_f)
        g = relu(co_g)
        o = relu(co_o)

        next_state = f * c_cur + i * g
        next_hidden_state = o * tanh(next_state)

        return next_hidden_state, next_state

    def init(self, batch_size, shape):
        height, width = shape

        return (zeros(batch_size, self.hidden_channels, height, width).to("cuda"),
                zeros(batch_size, self.hidden_channels, height, width).to("cuda"))


class ConvLSTMNetwork(nn.Module):
    """
    定义2D卷积长短期神经网络
    """

    def __init__(self, kernel_size: tuple[int, int],
                 bias: bool = True, num_layers: int = 2):
        super(ConvLSTMNetwork, self).__init__()

        self.kernel_size = kernel_size
        self.bias = bias  # 是否使用学习偏差
        self.num_layers = num_layers  # 神经元数量

        # 初始化神经元
        cells = nn.ModuleList()

        for i in range(self.num_layers):
            cell = ConvLSTCell(kernel_size=self.kernel_size, bias=self.bias)
            cells.append(cell)

        self.cells = cells

    def forward(self, inputs: tensor, states: tensor = None):
        # 默认 batch_first
        batch_size = inputs.shape[0]
        time_seq_length = inputs.shape[1]
        shape = inputs.shape[2:]

        if states is None:
            # 初始化状态
            states = [cell.init(batch_size, shape) for cell in self.cells]

        layers_outputs = []
        history_states = []

        for index in range(self.num_layers):

            h_state, c_state = states[index]
            # 当前层的输出
            outputs = []

            for time in range(time_seq_length):
                cell = self.cells[index]
                cell_inputs = inputs[:, time, ...]  # 3D
                h_state, c_state = cell(cell_inputs, (h_state, c_state))
                outputs.append(h_state)

            outputs = stack(outputs, dim=1)
            layers_outputs.append(outputs)
            history_states.append([h_state, c_state])

        return layers_outputs[-1].squeeze(dim=2), [state.squeeze(dim=1) for state in history_states[-1]]

    def fit(self, inputs: tensor or DataLoader, labels: tensor = None,
            epochs: int = 10, loss_function=None, optimizer=None):
        """
        训练模型
        """
        if loss_function is None:
            loss_function = nn.MSELoss()
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=0.01)

        use_loader = isinstance(inputs, DataLoader)

        print(optimizer)

        if use_loader:
            x, label = next(iter(inputs))
            batch_size = x.shape[0]

            for epoch in range(epochs):
                self.train(True)
                Log.i(f"Epoch {epoch + 1}\n---------------------------------")
                for sample_index in range(0, batch_size, 2):
                    sample_x = x[sample_index:sample_index + 1, ...].clone().detach().to('cuda:0')
                    sample_label = label[sample_index:sample_index + 1, ...].clone().detach().to('cuda:0')

                    output, (h_state, c_state) = self.forward(sample_x)
                    loss = loss_function(c_state, sample_label)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    Log.i(f" loss: is {loss.item()}")

        return self


# Keras Model --------------------------------------------------------------

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras import Model, Sequential, Input, layers, saving

@saving.register_keras_serializable(package="Custom", name="Conv2DLSTMNetwork")
class Conv2DLSTMNetwork(Model):
    def __init__(self, shape: tuple[int, int, int], **kwargs):
        super(Conv2DLSTMNetwork, self).__init__(**kwargs)

        print("Conv2DLSTMNetwork init shape: " + str(shape))

        self.model = Sequential([
            Input(batch_shape=shape),
            layers.ConvLSTM2D(
                filters=64,
                kernel_size=(5, 5),
                padding="same",
                activation="sigmoid",
                return_sequences=True
            ),  # output = (100, 14, 80, 80, 64)
            layers.ConvLSTM2D(
                filters=64,
                kernel_size=(5, 5),
                padding="same",
                activation="sigmoid",
                return_sequences=True
            ),  # output = (100, 14, 80, 80, 64)
            layers.ConvLSTM2D(
                filters=64,
                kernel_size=(5, 5),
                padding="same",
                activation="sigmoid",
                return_sequences=True
            ),  # output = (100, 14, 80, 80, 64)
            layers.ConvLSTM2D(
                filters=64,
                kernel_size=(5, 5),
                padding="same",
                activation="sigmoid",
                return_sequences=False
            ),  # output = (100, 14, 80, 80, 64)
            layers.BatchNormalization(),
            layers.Conv2D(
                filters=1,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ),  # output = (100, 80, 80, 1)
        ])

    def get_config(self):
        return {"shape": self.model.input_shape}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    def call(self, *args, **kwargs):
        return self.model(*args, **kwargs)
