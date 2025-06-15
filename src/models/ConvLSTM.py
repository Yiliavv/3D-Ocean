import torch
from torch import nn, manual_seed, optim
from lightning import LightningModule

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

        super().__init__()

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
        super().__init__()

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
        
        # 修改全连接层的输入和输出维度，保持与输入相同的空间维度
        self.fc = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim[-1] * 80 * 180, 1 * 80 * 180, bias=True)
        )
        self.nor = nn.BatchNorm2d(self.hidden_dim[-1])
        
        # 训练损失
        self.train_loss = []
        # 验证损失
        self.val_loss = []

    def forward(self, x):
        # b - batch_size: Number of images in each batch
        # t - seq_len: Number of images in each sequence
        # c - Number of channels in the input
        # h - Height of the image
        # w - Width of the image
        
        x_processed = self.__normalize__(x)
        
        b, t, c, h, w = x_processed.shape
        
        # 将张量尺寸转换为整数
        h_int = h.item() if isinstance(h, torch.Tensor) else h
        w_int = w.item() if isinstance(w, torch.Tensor) else w
        
        hidden_state = self._init_hidden(batch_size=b,
                                         image_size=(h_int, w_int))
        # 所有层的输出
        layer_output_list = []
        # 通过最后一个神经元的状态
        last_state_list = []

        seq_len = x_processed.size(1)
        # 当前层的输入，从前一层继承输出
        cur_layer_input = x_processed

        # 每一层进行计算
        for layer_idx in range(self.num_layers):
            # 保存每一层的输出
            h, c = hidden_state[layer_idx]
            output_inner = []
            # 每一个时间步进行计算
            for t in range(seq_len):
                cell = self.cell_list[layer_idx]
                h, c = cell(input_tensor=cur_layer_input[:, t, :, :, :],
                            cur_state=[h, c])
                output_inner.append(h)

            # 一个层输出的时间序列
            layer_output = torch.stack(output_inner, dim=1)

            # 更新当前层的输入
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # 获取最后的输出
        output_c = last_state_list[-1][1]

        output = output_c
        output = self.nor(output)
        # 调整 view 操作以匹配实际的输入维度
        output = output.view(b, -1)  # 将所有维度展平
        output = self.fc(output)
        output = output.view(b, 1, h_int, w_int)  # 使用整数尺寸

        return output

    def training_step(self, batch):
        x, y = batch

        output = self(x)

        loss = self.custom_mse_loss(output, y)
        self.train_loss.append(loss.item())
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch):
        x, y = batch
        
        output = self(x)

        loss = self.custom_mse_loss(output, y)
        self.val_loss.append(loss.item())
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        a_opt = optim.Adam(self.parameters(), lr=1e-4)
        return a_opt
    
    def custom_mse_loss(self, y_pred, y):
        """
        自定义MSE损失函数，忽略nan值
        """
        
        y_mask = torch.isnan(y)
        
        # 创建掩码，标记非nan值
        mask = ~y_mask
        
        # 只计算非nan值的MSE
        if mask.sum() > 0:  # 确保有非nan值
            # 将y_pred中的nan值替换为0，以便计算损失
            y_pred_processed = y_pred.clone()
            y_pred_processed[y_mask] = 0.0
            
            # 将y中的nan值替换为0
            y_processed = y.clone()
            y_processed[y_mask] = 0.0
            
            # 计算MSE，只考虑非nan位置
            return nn.MSELoss()(y_pred_processed, y_processed)
        else:
            # 如果所有值都是nan，返回0
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    
    def __normalize__(self, x):
        # 保存原始nan掩码
        x_mask = torch.isnan(x)
        # 将nan值替换为0，以便模型处理
        x_processed = x.clone()
        x_processed[x_mask] = 0.0
        
        return x_processed

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
