import torch
from torch import nn, tensor
import matplotlib.pyplot as plt
import networkx as nx

class LSTMNetwork(nn.Module):
    """
    定义卷积长短期神经网络
    """
    def __init__(self, shape):
        super(LSTMNetwork, self).__init__()
        self.channel = 1  # 数据通道
        self.bias = True  # 是否使用学习偏差

        # 初始化神经元
        self.conv1 = nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, bias=self.bias).to('cuda')  # 卷积层
        self.conv1.weight = nn.Parameter(tensor([[[[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]]]).to('cuda'))
        self.pool1 = nn.AvgPool2d(2, 2).to('cuda')  # 池化层
        self.activation1 = nn.ReLU().to('cuda')

    def forward(self, input):
        output = self.conv1(input)
        output = self.pool1(output)
        output = self.activation1(output)
        return output

# 创建模型实例
model = LSTMNetwork((1, 28, 28))

# 创建输入张量
input = torch.ones(1, 1, 28, 28).to('cuda')

# 前向传播并计算每一层的输出
conv_output = model.conv1(input)
pool_output = model.pool1(conv_output)
activation_output = model.activation1(pool_output)

# 假设我们有每一层的 RMSE 值
rmse_values_conv = torch.rand(10).tolist()  # 示例数据，替换为实际 RMSE 值
rmse_values_pool = torch.rand(10).tolist()  # 示例数据，替换为实际 RMSE 值
rmse_values_activation = torch.rand(10).tolist()  # 示例数据，替换为实际 RMSE 值

# 绘制每一层的误差曲线
def plot_rmse_curve(rmse_values, layer_name):
    x = range(1, len(rmse_values) + 1)
    plt.plot(x, rmse_values, marker='o', linestyle='-', label=f'{layer_name} RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title(f'{layer_name} RMSE Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_rmse_curve(rmse_values_conv, 'Conv Layer')
plot_rmse_curve(rmse_values_pool, 'Pool Layer')
plot_rmse_curve(rmse_values_activation, 'Activation Layer')

# 绘制模型的树结构图
def plot_model_tree():
    G = nx.DiGraph()
    G.add_edges_from([
        ('Input', 'Conv Layer'),
        ('Conv Layer', 'Pool Layer'),
        ('Pool Layer', 'Activation Layer')
    ])
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    plt.title('Model Tree Structure')
    plt.show()

plot_model_tree()