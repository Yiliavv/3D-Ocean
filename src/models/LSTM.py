from torch import nn, tensor

class LSTMNetwork(nn.Module):
    """
    定义卷积长短期神经网络
    """
    def __init__(self, shape):
        super(LSTMNetwork, self).__init__()
        self.channel = 1  # 数据通道
        self.bias = True  # 是否使用学习偏差

        # 初始化神经元
        self.conv1 = nn.Conv2d(self.channel, self.channel, 3, bias=self.bias) # 卷积层
        self.conv2 = nn.Conv2d(self.channel, self.channel, 3, bias=self.bias) # 卷积层
        self.lstm = nn.LSTM(*shape) # 长短期记忆网络
        self.pool1 = nn.MaxPool2d(3)  # 最大池化找分布
        self.pool2 = nn.AvgPool2d(3)  # 平均池化平滑过渡
        self.activation1 = nn.Sigmoid()  # 激活函数
        self.activation2 = nn.ReLU()     # 激活函数
        self.fc1 = nn.Linear(10, 10)     # 全连接层
        self.loss = nn.functional.mse_loss # 损失函数
        

    def forward(self, x, y):
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            input = x[i, :, :, :]
            
            train_output = self.train(input, )
            ur_output= y[i, ]
            
            loss = self.loss(train_output, ur_output)
            loss.backward()
    
    def train(self, x):
        """
        训练网络
        """
        x = self.conv1(x)
        print("卷积： ", x.shape)
        x = self.pool1(x)
        print("池化： ", x.shape)
        x = self.activation1(x)
        print("激活: ", x.shape)
        x = self.fc1(x.shape)
        print("全连接：", x.shape)
        
        return x