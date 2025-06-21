# 随机决策森林
import torch
from torch import optim
from lightning import LightningModule
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class RDFNetwork(LightningModule):
    """
    随机决策森林网络，支持 PyTorch 张量和 Lightning 训练器
    
    参数:
        n_estimators: int = 512, 决策树数量
        random_state: int = 42, 随机种子
        n_jobs: int = 16, 并行作业数
        verbose: bool = True, 是否显示训练进度
        learning_rate: float = 1e-4, 学习率（为了兼容 Lightning，实际未使用）
    """
    def __init__(self, 
                 n_estimators=512,
                 random_state=42,
                 n_jobs=16,
                 verbose=True,
                 learning_rate=1e-4):
        super().__init__()
        
        # 创建随机森林回归器
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        self.learning_rate = learning_rate

        # 训练损失
        self.train_loss = []
        # 验证损失
        self.val_loss = []
    
    def forward(self, x, y):
        """
        前向传播
        Args:
            x: numpy.ndarray, 输入数组 [batch_size, features]
            y: numpy.ndarray, 目标数组 [batch_size, features, depth]
        """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        # 打印调试信息
        print(f"Forward - Input shape: {x.shape}, Target shape: {y.shape}")
        
        # 训练模型
        self.model.fit(x, y)
    
    def custom_mse_loss(self, y_pred, y):
        """
        自定义MSE损失函数，忽略nan值
        """
        print(f"Loss - y_pred shape: {y_pred.shape}, y shape: {y.shape}")
        
        y_mask = torch.isnan(y)
        mask = ~y_mask
        
        if mask.sum() > 0:
            y_pred_processed = y_pred.clone()
            y_pred_processed[y_mask] = 0.0
            
            y_processed = y.clone()
            y_processed[y_mask] = 0.0
            
            return torch.nn.MSELoss()(y_pred_processed, y_processed)
        else:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    
    def training_step(self, batch):
        """
        训练步骤
        Args:
            batch: tuple, (输入, 目标)
        Returns:
            torch.Tensor, 损失值
        """
        x, y = batch
        
        # 转换为NumPy数组
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        
        # 获取形状信息
        batch_size = x.shape[0]
        
        # 重塑输入：[batch_size, height, width] -> [batch_size, height*width]
        input_reshaped = x.reshape(batch_size, -1)
        
        # 重塑输出：[batch_size, height, width, depth] -> [batch_size, height*width*depth]
        output_reshaped = y.reshape(batch_size, -1)
        
        print(f"Training - Input shape: {input_reshaped.shape}, Target shape: {output_reshaped.shape}")
        
        # 训练模型
        self.forward(input_reshaped, output_reshaped)
        
        # 预测
        y_pred = self.model.predict(input_reshaped)
        
        # 将预测结果转换回PyTorch张量并重塑为原始形状
        y_pred = torch.from_numpy(y_pred)
        y = torch.from_numpy(y)
        
        y_pred = y_pred.reshape(y.shape)
        
        # 计算损失
        loss = self.custom_mse_loss(y_pred, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.train_loss.append(loss.item())
            
    def configure_optimizers(self):
        """
        配置优化器（为了兼容 Lightning，实际未使用）
        """
        # 由于随机森林不需要梯度优化，这里返回一个虚拟优化器
        return optim.Adam([torch.zeros(1, requires_grad=True)], lr=self.learning_rate)
    
    def __normalize__(self, x):
        """
        标准化输入数据
        Args:
            x: torch.Tensor, 输入张量
        Returns:
            torch.Tensor, 标准化后的张量
        """
        x_mask = torch.isnan(x)
        x_processed = x.clone()
        x_processed[x_mask] = 0.0
        
        return x_processed
