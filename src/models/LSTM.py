import numpy as np
import torch
from torch import nn, manual_seed, optim
from lightning import LightningModule

from src.utils.log import Log


class LSTM(LightningModule):
    """
    纯粹的LSTM模型用于海表温度预测
    
    参数:
        width: int, 宽度
        height: int, 高度
        seq_len: int, 序列长度
        hidden_dim: int, LSTM隐藏层维度
        num_layers: int, LSTM层数
        dropout: float, dropout比例
        learning_rate: float, 学习率
        optimizer: optim.Optimizer, 优化器
    """
    
    def __init__(self,
                 width,
                 height,
                 seq_len,
                 hidden_dim=128,
                 num_layers=2,
                 dropout=0.2,
                 learning_rate=1e-3,
                 optimizer=optim.Adam):
        
        manual_seed(1)
        super().__init__()
        
        self.width = width
        self.height = height
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        
        # 输入特征维度 (每个时间步的空间维度)
        self.input_dim = width * height
        
        # 添加调试信息
        print(f"LSTM模型初始化:")
        print(f"  width: {width}, height: {height}")
        print(f"  input_dim: {self.input_dim}")
        print(f"  seq_len: {seq_len}")
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.input_dim)
        )
        
        # 训练和验证损失记录
        self.train_loss = []
        self.val_loss = []
        
        Log.i(f"LSTM模型初始化完成: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}")
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch_size, seq_len - 1, channel, width, height] 输入序列
            
        返回:
            output: [batch_size, 1, channel, width, height] 预测结果
        """
         # 注意：ERA5数据集返回的是[seq_len, channel, height, width]
        batch_size = x.shape[0]
        actual_height = x.shape[3]
        actual_width = x.shape[4]
        
        # 数据预处理
        input_seq = self.__normalize__(x)
        
        # 检查是否有通道维度，如果有则去掉
        if len(input_seq.shape) == 5:  # [batch, seq_len, channel, height, width]
            input_seq = input_seq.squeeze(2)  # 去掉通道维度
        
        input_seq_len = input_seq.shape[1]
        
        # 动态计算输入维度
        actual_input_dim = actual_height * actual_width
        
        # 重塑为LSTM输入格式
        x_reshaped = input_seq.contiguous().view(batch_size, input_seq_len, actual_input_dim)
        
        # 如果实际输入维度与模型初始化时不同，需要调整
        if actual_input_dim != self.input_dim:
            # 重新初始化LSTM层以匹配实际输入维度
            self.input_dim = actual_input_dim
            self.lstm = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=0.2 if self.num_layers > 1 else 0,
                batch_first=True
            ).to(x.device)
            
            # 重新初始化输出层
            self.output_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim // 2, self.input_dim)
            ).to(x.device)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        
        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x_reshaped, (h0, c0))
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # 通过输出层
        prediction = self.output_layer(last_output)  # [batch_size, input_dim]
        
        # 重塑为空间维度
        output = prediction.view(batch_size, 1, actual_height, actual_width)
        
        return output
    
    def custom_mse_loss(self, y_pred, y):
        """
        自定义MSE损失函数，忽略nan值
        
        参数:
            y_pred: 预测值
            y: 真实值
            
        返回:
            loss: 损失值
        """
        # 创建nan掩码
        y_mask = torch.isnan(y)
        
        # 创建非nan值的掩码
        mask = ~y_mask
        
        if mask.sum() > 0:
            # 处理预测值中的nan
            y_pred_processed = y_pred.clone()
            y_pred_processed[y_mask] = 0.0
            
            # 处理真实值中的nan
            y_processed = y.clone()
            y_processed[y_mask] = 0.0
            
            # 计算MSE损失，只考虑非nan位置
            return nn.MSELoss()(y_pred_processed, y_processed)
        else:
            # 如果所有值都是nan，返回0
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    
    def training_step(self, batch):
        """
        训练步骤
        
        参数:
            batch: (x, y) 训练批次
            
        返回:
            loss: 训练损失
        """
        # x: [batch, seq_len - 1, channel, width, height]
        # y: [batch, 1, channel, width, height]
        x, y = batch
        
        # 获取nan掩码
        y_mask = torch.isnan(y)
        
        # 前向传播
        y_pred = self(x)
        
        # 将预测值中对应nan位置也设为nan
        y_pred[y_mask] = np.nan
        
        # 计算损失
        loss = self.custom_mse_loss(y_pred, y)
        
        # 记录损失
        self.log('train_loss', loss, prog_bar=True)
        self.train_loss.append(loss.item())
        
        return loss
    
    def validation_step(self, batch):
        """
        验证步骤
        
        参数:
            batch: (x, y) 验证批次
            
        返回:
            val_loss: 验证损失
        """
        x, y = batch
        
        # 前向传播
        y_pred = self(x)
        
        # 计算验证损失
        val_loss = self.custom_mse_loss(y_pred, y)
        
        # 记录损失
        self.log('val_loss', val_loss)
        self.val_loss.append(val_loss.item())
        
        return val_loss
    
    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        
        返回:
            optimizer配置字典
        """
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        
        # 使用余弦退火学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
    
    def __normalize__(self, x):
        """
        数据归一化处理
        
        参数:
            x: 输入数据
            
        返回:
            x_processed: 处理后的数据
        """
        # 保存原始nan掩码
        x_mask = torch.isnan(x)
        
        # 将nan值替换为0，以便模型处理
        x_processed = x.clone()
        x_processed[x_mask] = 0.0
        
        return x_processed
    
    def predict(self, x):
        """
        预测函数
        
        参数:
            x: 输入序列 [batch_size, seq_len, width, height]
            
        返回:
            prediction: 预测结果 [batch_size, 1, width, height]
        """
        self.eval()
        with torch.no_grad():
            prediction = self(x)
        return prediction
    
    def get_model_info(self):
        """
        获取模型信息
        
        返回:
            info: 模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": "Pure LSTM for SST Prediction",
            "input_shape": f"[batch_size, {self.seq_len}, {self.width}, {self.height}]",
            "output_shape": f"[batch_size, 1, {self.width}, {self.height}]",
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "learning_rate": self.learning_rate
        }
