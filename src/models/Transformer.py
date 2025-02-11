from torch import manual_seed, nn, optim
from torch.nn import Transformer, Linear
from lightning import LightningModule

class SSTTransformer(LightningModule):
    def __init__(self):
        manual_seed(1)
        super().__init__()
        
        # 修改模型参数
        self.d_model = 512  # 增加模型维度以处理复杂的空间信息
        self.transformer = Transformer(
            d_model=self.d_model, 
            nhead=8, 
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.2
        )
        
        # 输入投影层：将(20,20)的空间特征映射到d_model维度
        self.input_projection = Linear(400, self.d_model)  # 20*20=400
        
        # 输出投影层：将d_model维度映射回(20,20)空间
        self.output_projection = Linear(self.d_model, 400)  # 400=20*20
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 重塑输入: [batch, 14, 20, 20] -> [batch, 14, 400]
        x = x.view(batch_size, 14, -1)
        
        # 投影到transformer维度: [batch, 14, 400] -> [batch, 14, d_model]
        x = self.input_projection(x)
        
        # 创建一个目标序列（用最后一个时间步作为初始目标）
        tgt = x[:, -1:, :]  # [batch, 1, d_model]
        
        # Transformer期望的输入格式是 [seq_len, batch, d_model]
        x = x.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        
        # 通过transformer
        output = self.transformer(x, tgt)  # [1, batch, d_model]
        
        # 转回原始维度顺序
        output = output.permute(1, 0, 2)  # [batch, 1, d_model]
        
        # 投影回空间维度
        output = self.output_projection(output)  # [batch, 1, 400]
        
        # 重塑回空间维度 [batch, 1, 20, 20]
        output = output.view(batch_size, 1, 20, 20)
        
        return output
    
    def training_step(self, batch):
        x, y = batch  # x: [batch, 14, 20, 20], y: [batch, 1, 20, 20]
        y_pred = self(x)
        loss = nn.MSELoss()(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_pred = self(x)
        val_loss = nn.MSELoss()(y_pred, y)
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
