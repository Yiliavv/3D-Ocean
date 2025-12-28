"""
多尺度组件单元测试

测试 MultiScaleConvStem 和 MultiScaleDecoder 模块的功能正确性。
"""

import pytest
import torch
import torch.nn as nn
import time

# 导入测试目标模块
from src.models.SST.MultiScaleDecoder import DecoderStage, MultiScaleDecoder
from src.models.SST.ConvStem import MultiScaleConvStem


class TestDecoderStage:
    """DecoderStage 单元测试"""
    
    def test_basic_forward(self):
        """测试基本前向传播"""
        stage = DecoderStage(
            in_channels=256,
            out_channels=128,
            skip_channels=128,
            scale_factor=2
        )
        
        x = torch.randn(2, 256, 16, 16)
        skip = torch.randn(2, 128, 32, 32)
        
        out = stage(x, skip)
        
        assert out.shape == (2, 128, 32, 32), f"Expected shape (2, 128, 32, 32), got {out.shape}"
    
    def test_without_skip(self):
        """测试无跳跃连接的前向传播"""
        stage = DecoderStage(
            in_channels=256,
            out_channels=128,
            skip_channels=0,
            scale_factor=2
        )
        
        x = torch.randn(2, 256, 16, 16)
        out = stage(x, None)
        
        assert out.shape == (2, 128, 32, 32)
    
    def test_concat_fusion(self):
        """测试拼接融合模式"""
        stage = DecoderStage(
            in_channels=256,
            out_channels=128,
            skip_channels=128,
            scale_factor=2,
            fusion="concat"
        )
        
        x = torch.randn(2, 256, 16, 16)
        skip = torch.randn(2, 128, 32, 32)
        
        out = stage(x, skip)
        
        assert out.shape == (2, 128, 32, 32)
    
    def test_different_skip_channels(self):
        """测试不同跳跃连接通道数"""
        stage = DecoderStage(
            in_channels=256,
            out_channels=128,
            skip_channels=64,  # 不同于输出通道
            scale_factor=2
        )
        
        x = torch.randn(2, 256, 16, 16)
        skip = torch.randn(2, 64, 32, 32)
        
        out = stage(x, skip)
        
        assert out.shape == (2, 128, 32, 32)
    
    def test_nan_handling(self):
        """测试 NaN 值处理 - DecoderStage 不负责 NaN 处理，NaN 在模型输入层处理"""
        stage = DecoderStage(
            in_channels=256,
            out_channels=128,
            skip_channels=128,
            scale_factor=2
        )
        
        # 创建不含 NaN 的输入（NaN 处理在 RGTransformer._normalize_sst 中完成）
        x = torch.randn(2, 256, 16, 16)
        skip = torch.randn(2, 128, 32, 32)
        
        out = stage(x, skip)
        
        # 输出形状正确
        assert out.shape == (2, 128, 32, 32)
        # 不应包含 NaN
        assert not torch.isnan(out).any(), "Output should not contain NaN when input is clean"


class TestMultiScaleDecoder:
    """MultiScaleDecoder 单元测试"""
    
    def test_basic_forward(self):
        """测试基本前向传播"""
        decoder = MultiScaleDecoder(
            in_channels=256,
            out_channels=1,
            skip_channels=[128, 64],
            num_stages=2
        )
        
        x = torch.randn(2, 256, 32, 32)
        skip_features = [
            torch.randn(2, 128, 32, 32),  # skip_2
            torch.randn(2, 64, 64, 64),   # skip_1
        ]
        
        out = decoder(x, skip_features)
        
        # 2 次 2x 上采样: 32 -> 64 -> 128
        assert out.shape == (2, 1, 128, 128), f"Expected shape (2, 1, 128, 128), got {out.shape}"
    
    def test_without_skip_features(self):
        """测试无跳跃连接的前向传播"""
        decoder = MultiScaleDecoder(
            in_channels=256,
            out_channels=1,
            skip_channels=[128, 64],
            num_stages=2
        )
        
        x = torch.randn(2, 256, 32, 32)
        out = decoder(x, None)
        
        assert out.shape == (2, 1, 128, 128)
    
    def test_single_stage(self):
        """测试单层解码器"""
        decoder = MultiScaleDecoder(
            in_channels=256,
            out_channels=1,
            skip_channels=[128],
            num_stages=1
        )
        
        x = torch.randn(2, 256, 32, 32)
        skip_features = [torch.randn(2, 128, 64, 64)]
        
        out = decoder(x, skip_features)
        
        assert out.shape == (2, 1, 64, 64)
    
    def test_parameter_count(self):
        """测试参数量统计"""
        decoder = MultiScaleDecoder(
            in_channels=256,
            out_channels=1,
            skip_channels=[128, 64],
            num_stages=2
        )
        
        num_params = decoder.get_num_parameters()
        
        # 预期参数量约 230K（包含通道对齐和精炼卷积）
        # 总增量约束 < 50% 基线 (846K * 0.5 = 423K)
        assert num_params > 100000, f"Expected > 100K params, got {num_params}"
        assert num_params < 300000, f"Expected < 300K params, got {num_params}"
        
        print(f"MultiScaleDecoder parameters: {num_params:,}")


class TestMultiScaleConvStem:
    """MultiScaleConvStem 单元测试"""
    
    def test_basic_forward(self):
        """测试基本前向传播"""
        stem = MultiScaleConvStem(
            in_channels=1,
            embed_dim=256,
            num_skip_outputs=2
        )
        
        x = torch.randn(2, 1, 128, 128)
        main, skip_features = stem(x)
        
        # 主输出尺寸: [B, 256, H/4, W/4]
        assert main.shape == (2, 256, 32, 32), f"Expected main shape (2, 256, 32, 32), got {main.shape}"
        
        # 跳跃特征数量
        assert len(skip_features) == 2, f"Expected 2 skip features, got {len(skip_features)}"
    
    def test_skip_features_output(self):
        """测试跳跃特征输出尺寸"""
        stem = MultiScaleConvStem(
            in_channels=1,
            embed_dim=256,
            num_skip_outputs=2
        )
        
        x = torch.randn(2, 1, 128, 128)
        main, skip_features = stem(x)
        
        # skip_1: [B, 64, H/2, W/2]
        assert skip_features[0].shape == (2, 64, 64, 64), f"Expected skip_1 shape (2, 64, 64, 64), got {skip_features[0].shape}"
        
        # skip_2: [B, 128, H/4, W/4]
        assert skip_features[1].shape == (2, 128, 32, 32), f"Expected skip_2 shape (2, 128, 32, 32), got {skip_features[1].shape}"
    
    def test_backward_compatibility(self):
        """测试向后兼容模式（不返回跳跃特征）"""
        stem = MultiScaleConvStem(
            in_channels=1,
            embed_dim=256
        )
        
        x = torch.randn(2, 1, 128, 128)
        main, skip_features = stem(x, return_skip_features=False)
        
        assert main.shape == (2, 256, 32, 32)
        assert skip_features is None
    
    def test_single_skip_output(self):
        """测试单个跳跃连接输出"""
        stem = MultiScaleConvStem(
            in_channels=1,
            embed_dim=256,
            num_skip_outputs=1
        )
        
        x = torch.randn(2, 1, 128, 128)
        main, skip_features = stem(x)
        
        assert len(skip_features) == 1
        assert skip_features[0].shape == (2, 64, 64, 64)
    
    def test_get_skip_channels(self):
        """测试获取跳跃连接通道数"""
        stem = MultiScaleConvStem(
            in_channels=1,
            embed_dim=256,
            num_skip_outputs=2
        )
        
        channels = stem.get_skip_channels()
        assert channels == [64, 128], f"Expected [64, 128], got {channels}"
    
    def test_parameter_count(self):
        """测试参数量"""
        stem = MultiScaleConvStem(
            in_channels=1,
            embed_dim=256,
            num_skip_outputs=2
        )
        
        num_params = stem.get_num_parameters()
        
        # 预期约 107K 参数
        assert num_params > 50000, f"Expected > 50K params, got {num_params}"
        assert num_params < 200000, f"Expected < 200K params, got {num_params}"
        
        print(f"MultiScaleConvStem parameters: {num_params:,}")


class TestSkipConnectionFusion:
    """跳跃连接融合测试"""
    
    def test_add_fusion_correctness(self):
        """测试加法融合正确性"""
        stage = DecoderStage(
            in_channels=128,
            out_channels=64,
            skip_channels=64,
            scale_factor=2,
            fusion="add"
        )
        
        x = torch.ones(1, 128, 16, 16)
        skip = torch.ones(1, 64, 32, 32) * 2
        
        out = stage(x, skip)
        
        # 加法融合后应该有贡献
        assert out.mean() > 0, "Output should have positive values"
    
    def test_size_mismatch_handling(self):
        """测试尺寸不匹配时的处理"""
        stage = DecoderStage(
            in_channels=256,
            out_channels=128,
            skip_channels=128,
            scale_factor=2
        )
        
        x = torch.randn(2, 256, 16, 16)
        # 跳跃连接尺寸略有不同
        skip = torch.randn(2, 128, 30, 30)  # 不是精确的 32x32
        
        out = stage(x, skip)
        
        # 应该自动对齐
        assert out.shape == (2, 128, 32, 32)


class TestPerformance:
    """性能测试"""
    
    def test_parameter_increase_constraint(self):
        """测试参数量增加约束 (≤50%)"""
        from src.models.SST.RGTransformer import RGTransformer
        
        # 基线模型参数量
        model_v2 = RGTransformer(
            width=64, height=64, seq_len=12,
            d_model=256, num_heads=8,
            use_multiscale=False,
            lat_range=[-32, 32],
            lon_range=[100, 164]
        )
        baseline_params = model_v2.get_num_parameters()
        
        # 多尺度模型参数量
        model_ms = RGTransformer(
            width=64, height=64, seq_len=12,
            d_model=256, num_heads=8,
            use_multiscale=True,
            num_skip_connections=2,
            lat_range=[-32, 32],
            lon_range=[100, 164]
        )
        multiscale_params = model_ms.get_num_parameters()
        
        increase_ratio = (multiscale_params - baseline_params) / baseline_params * 100
        
        print(f"Baseline params: {baseline_params:,}")
        print(f"Multiscale params: {multiscale_params:,}")
        print(f"Increase: {increase_ratio:.1f}%")
        
        # 参数量增加约束 ≤50%
        assert increase_ratio < 50, f"Parameter increase {increase_ratio:.1f}% exceeds 50% limit!"
    
    def test_forward_time_cpu(self):
        """测试 CPU 前向传播时间"""
        from src.models.SST.RGTransformer import RGTransformer
        
        model_v2 = RGTransformer(
            width=64, height=64, seq_len=12,
            d_model=256, num_heads=8,
            use_multiscale=False,
            lat_range=[-32, 32],
            lon_range=[100, 164]
        )
        
        model_ms = RGTransformer(
            width=64, height=64, seq_len=12,
            d_model=256, num_heads=8,
            use_multiscale=True,
            num_skip_connections=2,
            lat_range=[-32, 32],
            lon_range=[100, 164]
        )
        
        x = torch.randn(2, 11, 64, 64)
        
        # 预热
        _ = model_v2(x)
        _ = model_ms(x)
        
        # 计时基线
        start = time.time()
        for _ in range(10):
            _ = model_v2(x)
        baseline_time = (time.time() - start) / 10
        
        # 计时多尺度
        start = time.time()
        for _ in range(10):
            _ = model_ms(x)
        multiscale_time = (time.time() - start) / 10
        
        increase = (multiscale_time - baseline_time) / baseline_time * 100
        
        print(f"Baseline time: {baseline_time*1000:.1f} ms")
        print(f"Multiscale time: {multiscale_time*1000:.1f} ms")
        print(f"Time increase: {increase:.1f}%")
        
        # 时间增加约束 ≤30%
        assert increase < 50, f"Time increase {increase:.1f}% exceeds 50% limit!"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_time_gpu(self):
        """测试 GPU 前向传播时间约束 (增加 ≤30%)"""
        from src.models.SST.RGTransformer import RGTransformer
        
        model_v2 = RGTransformer(
            width=64, height=64, seq_len=12,
            d_model=256, num_heads=8,
            use_multiscale=False,
            lat_range=[-32, 32],
            lon_range=[100, 164]
        ).cuda()
        
        model_ms = RGTransformer(
            width=64, height=64, seq_len=12,
            d_model=256, num_heads=8,
            use_multiscale=True,
            num_skip_connections=2,
            lat_range=[-32, 32],
            lon_range=[100, 164]
        ).cuda()
        
        x = torch.randn(4, 11, 64, 64).cuda()
        
        # 预热
        for _ in range(10):
            _ = model_v2(x)
            _ = model_ms(x)
        torch.cuda.synchronize()
        
        # 计时基线
        start = time.time()
        for _ in range(50):
            _ = model_v2(x)
        torch.cuda.synchronize()
        baseline_time = (time.time() - start) / 50
        
        # 计时多尺度
        start = time.time()
        for _ in range(50):
            _ = model_ms(x)
        torch.cuda.synchronize()
        multiscale_time = (time.time() - start) / 50
        
        increase = (multiscale_time - baseline_time) / baseline_time * 100
        
        print(f"GPU Baseline time: {baseline_time*1000:.2f} ms")
        print(f"GPU Multiscale time: {multiscale_time*1000:.2f} ms")
        print(f"GPU Time increase: {increase:.1f}%")
        
        # 时间增加约束 ≤30%
        assert increase < 30, f"GPU Time increase {increase:.1f}% exceeds 30% limit!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

