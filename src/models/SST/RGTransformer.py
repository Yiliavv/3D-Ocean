"""
RGTransformer: 海表温度时空序列预测模型

主要特性：
- ConvStem 局部特征提取
- 球谐波空间位置编码
- EfficientRGAttention 时序注意力
- SwiGLU 激活函数
- SE 通道注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from lightning import LightningModule
from torch import optim

from einops import rearrange

from src.models.SST.PE.SphericalHarmonicEncoding import SpatialSphericalHarmonicEncoding
from src.models.SST.Attention.RGAttention import EfficientRGAttention
from src.models.SST.ConvStem import ConvStem, MultiScaleConvStem
from src.models.SST.MultiScaleDecoder import MultiScaleDecoder


class SwiGLU(nn.Module):
    """SwiGLU 激活函数 (PaLM/LLaMA)"""
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class ChannelFeedForward(nn.Module):
    """FFN: 支持 gelu/swiglu/geglu/mish 激活函数"""
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1, activation: str = 'swiglu'):
        super().__init__()
        self.activation_type = activation
        
        if activation == 'swiglu':
            self.net = SwiGLU(d_model, dim_feedforward, dropout)
        elif activation == 'geglu':
            self.w_gate = nn.Linear(d_model, dim_feedforward)
            self.w_up = nn.Linear(d_model, dim_feedforward)
            self.w_down = nn.Linear(dim_feedforward, d_model)
            self.dropout = nn.Dropout(dropout)
        elif activation == 'mish':
            self.net = nn.Sequential(
                nn.Linear(d_model, dim_feedforward), nn.Mish(), nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == 'geglu':
            return self.dropout(self.w_down(F.gelu(self.w_gate(x)) * self.w_up(x)))
        return self.net(x)


class SqueezeExcitation(nn.Module):
    """SE 通道注意力"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, reduced), nn.SiLU(),
            nn.Linear(reduced, channels), nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class RGTransformer(LightningModule):
    """海表温度时空序列预测模型"""
    
    def __init__(
        self, 
        width: int, 
        height: int, 
        seq_len: int,
        d_model: int = 256, 
        num_heads: int = 8, 
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_attn_layers: int = 1,
        learning_rate: float = 1e-4,
        lat_range: Optional[List[float]] = None,
        lon_range: Optional[List[float]] = None,
        resolution: float = 1.0,
        patch_size: int = 4,
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        use_multiscale: bool = False,
        num_skip_connections: int = 2,
        skip_fusion: str = "add",
        ffn_activation: str = 'swiglu',
        use_se_attention: bool = True,
        use_gradient_checkpointing: bool = False,
        loss_type: str = 'huber',
        huber_delta: float = 1.0,
        use_lr_scheduler: bool = True,
        warmup_epochs: int = 10,
        min_lr: float = 1e-6,
        weight_decay: float = 0.01,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 基础配置
        if not (width and height):
            raise ValueError("Must specify width and height")
        
        self.width, self.height, self.seq_len = width, height, seq_len
        self.d_model, self.patch_size = d_model, patch_size
        self.w_feat, self.h_feat = width // patch_size, height // patch_size
        
        # 训练配置
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.use_lr_scheduler = use_lr_scheduler
        self.min_lr = min_lr
        
        # 多尺度配置
        self.use_multiscale = use_multiscale
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        
        # 记录
        self.train_loss, self.val_loss = [], []
        self.viz = {}
        self._compiled = False
        
        # 空间位置编码
        self.spatial_pos_encoding = SpatialSphericalHarmonicEncoding(
            lat_range=lat_range, lon_range=lon_range,
            max_degree=2, resolution=resolution
        )
        self.spatial_enc_scale = nn.Parameter(torch.tensor(0.5))
        
        # ConvStem
        if use_multiscale:
            self.conv_stem = MultiScaleConvStem(
                in_channels=1, embed_dim=d_model,
                num_skip_outputs=num_skip_connections, use_bn=True
            )
            self._skip_channels = self.conv_stem.get_skip_channels()
        else:
            self.conv_stem = ConvStem(
                in_channels=1, embed_dim=d_model,
                target_reduction=patch_size, use_bn=True
            )
            self._skip_channels = []
        
        # SE 通道注意力
        self.se_attention = SqueezeExcitation(d_model, 16) if use_se_attention else None
        
        # Decoder
        if use_multiscale:
            self.decoder = MultiScaleDecoder(
                in_channels=d_model, out_channels=1,
                skip_channels=list(reversed(self._skip_channels)),
                num_stages=num_skip_connections, fusion=skip_fusion
            )
            self.patch_recovery = None
        else:
            self.patch_recovery = nn.ConvTranspose2d(d_model, 1, patch_size, patch_size)
            self.decoder = None
    
        # Transformer
        self.attention = EfficientRGAttention(
            d_model=d_model, num_heads=num_heads,
            dropout=dropout, num_layers=num_attn_layers, use_gate=True
        )
        self.ffn = ChannelFeedForward(d_model, dim_feedforward, dropout, ffn_activation)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 时序权重
        self.temporal_weights = nn.Parameter(torch.ones(seq_len - 1) / (seq_len - 1))
    
    def _maybe_compile(self):
        """首次前向传播时应用 torch.compile"""
        if self.use_compile and not self._compiled and hasattr(torch, 'compile'):
            try:
                self._forward_impl = torch.compile(self._forward_impl, mode=self.compile_mode)
                self._compiled = True
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")
                self._compiled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: [B, S-1, W, H] -> [B, W, H]"""
        self._maybe_compile()
        return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """实际前向传播实现"""
        batch_size, seq_len_minus_1, width, height = x.shape
        
        # 输入归一化 + 位置编码
        x = self._normalize_sst(x)
        x = x + self.spatial_pos_encoding() * self.spatial_enc_scale
        
        # ConvStem: [B, S, W, H] -> [B*S, 1, W, H] -> [B*S, D, W', H']
        x_flat = rearrange(x, 'b s w h -> (b s) 1 w h')
        
        if self.use_multiscale:
            x_embed, skip_features = self.conv_stem(x_flat, return_skip_features=True)
        else:
            x_embed = self.conv_stem(x_flat)
            skip_features = None
        
        if self.se_attention is not None:
            x_embed = self.se_attention(x_embed)
        
        _, d_model, w_feat, h_feat = x_embed.shape
        
        # Transformer: [B*S, D, W', H'] -> [B*W'*H', S, D]
        x_embed = rearrange(x_embed, '(b s) d w h -> (b w h) s d', b=batch_size, s=seq_len_minus_1)
        
        x_tokens = self.dropout(self.layer_norm(x_embed))
        x_tokens = self.attention(x_tokens)
        x_tokens = x_tokens + self.ffn(self.norm2(x_tokens))
        
        # 时序聚合: [N, S, D] -> [N, D]
        normalized_weights = F.softmax(self.temporal_weights, dim=0)
        output_tokens = (x_tokens * normalized_weights.view(-1, 1)).sum(dim=1)
        
        # 恢复空间: [B*W'*H', D] -> [B, D, W', H']
        output_tokens = rearrange(output_tokens, '(b w h) d -> b d w h', b=batch_size, w=w_feat, h=h_feat)
        
        # Decoder
        if self.use_multiscale and skip_features is not None:
            aggregated_skips = []
            for skip in skip_features:
                skip_reshaped = rearrange(skip, '(b s) c h w -> b s c h w', b=batch_size, s=seq_len_minus_1)
                skip_aggregated = (skip_reshaped * normalized_weights.view(1, -1, 1, 1, 1)).sum(dim=1)
                aggregated_skips.append(skip_aggregated)
            output_map = self.decoder(output_tokens, list(reversed(aggregated_skips)))
        else:
            output_map = self.patch_recovery(output_tokens)
        
        output = output_map.squeeze(1)
        self.viz['temporal_weights'] = normalized_weights
        return output
    
    def _normalize_sst(self, x: torch.Tensor) -> torch.Tensor:
        """将 NaN 替换为 0"""
        x_processed = x.clone()
        x_processed[torch.isnan(x)] = 0.0
        return x_processed

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """计算损失（处理 NaN）"""
        valid_mask = ~(torch.isnan(y_true) | torch.isnan(y_pred))
        if valid_mask.sum() == 0:
            return y_pred.sum() * 0.0
        
        y_pred_valid = y_pred[valid_mask]
        y_true_valid = y_true[valid_mask]
        
        if self.loss_type == 'huber':
            return F.huber_loss(y_pred_valid, y_true_valid, reduction='mean', delta=self.huber_delta)
        elif self.loss_type == 'combined':
            mse = F.mse_loss(y_pred_valid, y_true_valid, reduction='mean')
            mae = F.l1_loss(y_pred_valid, y_true_valid, reduction='mean')
            return 0.7 * mse + 0.3 * mae
        else:
            return F.mse_loss(y_pred_valid, y_true_valid, reduction='mean')
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.compute_loss(self(x), y)
        self.train_loss.append(loss.detach().cpu().item())
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        if self.use_lr_scheduler:
            self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_loss = self.compute_loss(self(x), y)
        self.val_loss.append(val_loss.detach().cpu().item())
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_rmse', torch.sqrt(val_loss), prog_bar=True, on_step=False, on_epoch=True)
        return val_loss
    
    def configure_optimizers(self):
        """配置 AdamW + CosineAnnealing"""
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        if not self.use_lr_scheduler:
            return optimizer
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=self.min_lr)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"}}
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """获取参数量"""
        return sum(p.numel() for p in self.parameters() if (not trainable_only or p.requires_grad))

