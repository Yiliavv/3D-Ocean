"""
RGTransformerV2 单元测试

测试:
- 模块初始化
- 前向传播
- NaN 处理
- API 兼容性
"""

import pytest
import torch
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.SST.RGTransformer import RGTransformer as RGTransformerV2, ChannelFeedForward
from src.models.SST.Attention.RGAttention import EfficientRGAttention
from src.models.SST.ConvStem import ConvStem


class TestConvStem:
    """ConvStem 模块测试"""
    
    def test_init(self):
        """测试初始化"""
        stem = ConvStem(in_channels=1, embed_dim=256, target_reduction=4)
        assert stem is not None
    
    def test_forward_shape(self):
        """测试前向传播输出形状"""
        stem = ConvStem(in_channels=1, embed_dim=256, target_reduction=4)
        x = torch.randn(2, 1, 64, 64)
        output = stem(x)
        
        # 输出应该是 [batch, embed_dim, H/4, W/4]
        assert output.shape == (2, 256, 16, 16)
    
    def test_different_input_sizes(self):
        """测试不同输入尺寸"""
        stem = ConvStem(in_channels=1, embed_dim=128, target_reduction=4)
        
        for size in [(32, 32), (64, 48), (128, 96)]:
            x = torch.randn(1, 1, size[0], size[1])
            output = stem(x)
            expected_shape = (1, 128, size[0]//4, size[1]//4)
            assert output.shape == expected_shape, f"Input {size} -> {output.shape}, expected {expected_shape}"


class TestEfficientRGAttention:
    """EfficientRGAttention 模块测试"""
    
    def test_init(self):
        """测试初始化"""
        attn = EfficientRGAttention(d_model=256, num_heads=8, num_layers=1)
        assert attn is not None
    
    def test_forward_shape(self):
        """测试前向传播输出形状"""
        attn = EfficientRGAttention(d_model=256, num_heads=8, num_layers=1)
        x = torch.randn(4, 7, 256)  # [batch, seq, d_model]
        output = attn(x)
        
        assert output.shape == x.shape
    
    def test_multi_layer(self):
        """测试多层注意力"""
        attn = EfficientRGAttention(d_model=256, num_heads=8, num_layers=3)
        x = torch.randn(4, 7, 256)
        output = attn(x)
        
        assert output.shape == x.shape
    
    def test_without_gate(self):
        """测试无门控模式"""
        attn = EfficientRGAttention(d_model=256, num_heads=8, num_layers=1, use_gate=False)
        x = torch.randn(4, 7, 256)
        output = attn(x)
        
        assert output.shape == x.shape
    
    def test_parameter_count(self):
        """测试参数量"""
        attn = EfficientRGAttention(d_model=256, num_heads=8, num_layers=1, use_gate=True)
        param_count = attn.get_num_parameters()
        
        # 应该远小于原 RGAttention 的 ~394K
        assert param_count < 300000, f"参数量 {param_count} 过大"
        print(f"EfficientRGAttention 参数量: {param_count:,}")


class TestRGTransformerV2:
    """RGTransformerV2 主模型测试"""
    
    @pytest.fixture
    def model(self):
        """创建测试模型
        
        注意: SpatialSphericalHarmonicEncoding 根据 lat/lon range 和 resolution 计算网格尺寸
        - lat_range=[-90, 90] (180°) / resolution=5.625° = 32 点 (height)
        - lon_range=[0, 360] (360°) / resolution=5.625° = 64 点 (width)
        """
        return RGTransformerV2(
            width=64,
            height=32,  # 修正：180° / 5.625° = 32
            seq_len=4,
            d_model=128,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1,
            num_attn_layers=1,
            lat_range=[-90, 90],
            lon_range=[0, 360],
            resolution=5.625,  # 64 points in 360 degrees (lon), 32 points in 180 degrees (lat)
            patch_size=4,
            use_compile=False,
        )
    
    def test_init(self, model):
        """测试初始化"""
        assert model is not None
        assert isinstance(model.conv_stem, ConvStem)
        assert isinstance(model.attention, EfficientRGAttention)
    
    def test_forward_shape(self, model):
        """测试前向传播输出形状"""
        batch_size = 2
        seq_len_minus_1 = 3
        x = torch.randn(batch_size, seq_len_minus_1, 32, 64)  # [B, S, height, width] (lat, lon)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        # 输出应该是 [batch, height, width]
        assert output.shape == (batch_size, 32, 64)
    
    def test_nan_handling(self, model):
        """T028: 测试 NaN 处理"""
        x = torch.randn(2, 3, 32, 64)  # [B, S, height, width] (lat, lon)
        
        # 添加 NaN 区域
        nan_mask = torch.zeros(32, 64, dtype=torch.bool)
        nan_mask[10:20, 20:40] = True
        x[:, :, nan_mask] = float('nan')
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        # 输出应该有有效值（非全 NaN）
        valid_output = ~torch.isnan(output)
        assert valid_output.any(), "输出全为 NaN"
    
    def test_loss_computation(self, model):
        """测试损失计算"""
        x = torch.randn(2, 3, 32, 64)  # [B, S, height, width] (lat, lon)
        y = torch.randn(2, 32, 64)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
            loss = model.custom_mse_loss(output, y)
        
        assert torch.isfinite(loss), "损失无效"
    
    def test_training_step(self, model):
        """测试训练步骤"""
        x = torch.randn(2, 3, 32, 64)  # [B, S, height, width] (lat, lon)
        y = torch.randn(2, 32, 64)
        batch = (x, y)
        
        model.train()
        loss = model.training_step(batch, 0)
        
        assert torch.isfinite(loss), "训练损失无效"
    
    def test_validation_step(self, model):
        """测试验证步骤"""
        x = torch.randn(2, 3, 32, 64)  # [B, S, height, width] (lat, lon)
        y = torch.randn(2, 32, 64)
        batch = (x, y)
        
        model.eval()
        loss = model.validation_step(batch, 0)
        
        assert torch.isfinite(loss), "验证损失无效"
    
    def test_optimizer_configuration(self, model):
        """测试优化器配置"""
        optimizer = model.configure_optimizers()
        assert optimizer is not None
    
    def test_parameter_count(self, model):
        """测试参数量方法"""
        param_count = model.get_num_parameters()
        assert param_count > 0
        print(f"RGTransformerV2 (small) 参数量: {param_count:,}")


class TestAPICompatibility:
    """API 兼容性测试"""
    
    def test_same_input_output_interface(self):
        """测试输入输出接口一致"""
        from src.models.SST.RGTransformer import RGTransformer
        
        # 注意：width=64 (lon), height=32 (lat) 对应 resolution=5.625
        config = {
            'width': 64,
            'height': 32,  # 180° / 5.625° = 32
            'seq_len': 4,
            'd_model': 128,
            'num_heads': 4,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'lat_range': [-90, 90],
            'lon_range': [0, 360],
            'resolution': 5.625,
            'patch_size': 4,
        }
        
        from src.models.SST.RGTransformerLegacy import RGTransformer as RGTransformerLegacy
        v1 = RGTransformerLegacy(**config, recursion_depth=2)
        v2 = RGTransformerV2(**config, num_attn_layers=1, use_compile=False)
        
        x = torch.randn(2, 3, 32, 64)  # [B, S, height, width] (lat, lon)
        
        v1.eval()
        v2.eval()
        
        with torch.no_grad():
            out_v1 = v1(x)
            out_v2 = v2(x)
        
        assert out_v1.shape == out_v2.shape, "输出形状不一致"
    
    def test_lightning_module_interface(self):
        """测试 LightningModule 接口"""
        from lightning import LightningModule
        
        model = RGTransformerV2(
            width=64, height=32, seq_len=4,  # height=32 对应 lat resolution
            lat_range=[-90, 90], lon_range=[0, 360],
            resolution=5.625, patch_size=4
        )
        
        assert isinstance(model, LightningModule)
        assert hasattr(model, 'training_step')
        assert hasattr(model, 'validation_step')
        assert hasattr(model, 'configure_optimizers')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

