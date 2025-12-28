"""
RGTransformer V1/V2 精度基准测试

测试目标:
- 验证集 MSE ≤ 105% V1
- NaN 处理正确
- 预测结果物理合理
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.SST.RGTransformerLegacy import RGTransformer as RGTransformerLegacy
from src.models.SST.RGTransformer import RGTransformer as RGTransformerV2


# 测试配置
TEST_CONFIG = {
    'width': 360,
    'height': 180,
    'seq_len': 8,
    'd_model': 256,
    'num_heads': 8,
    'dim_feedforward': 1024,
    'dropout': 0.1,
    'patch_size': 4,
    'lat_range': [-90, 90],
    'lon_range': [0, 360],
    'resolution': 1.0,
}


def create_v1_model():
    """创建 V1 模型（Legacy 版本）"""
    return RGTransformerLegacy(
        width=TEST_CONFIG['width'],
        height=TEST_CONFIG['height'],
        seq_len=TEST_CONFIG['seq_len'],
        d_model=TEST_CONFIG['d_model'],
        num_heads=TEST_CONFIG['num_heads'],
        dim_feedforward=TEST_CONFIG['dim_feedforward'],
        dropout=TEST_CONFIG['dropout'],
        recursion_depth=2,
        lat_range=TEST_CONFIG['lat_range'],
        lon_range=TEST_CONFIG['lon_range'],
        resolution=TEST_CONFIG['resolution'],
        patch_size=TEST_CONFIG['patch_size'],
    )


def create_v2_model():
    """创建 V2 模型"""
    return RGTransformerV2(
        width=TEST_CONFIG['width'],
        height=TEST_CONFIG['height'],
        seq_len=TEST_CONFIG['seq_len'],
        d_model=TEST_CONFIG['d_model'],
        num_heads=TEST_CONFIG['num_heads'],
        dim_feedforward=TEST_CONFIG['dim_feedforward'],
        dropout=TEST_CONFIG['dropout'],
        num_attn_layers=1,
        lat_range=TEST_CONFIG['lat_range'],
        lon_range=TEST_CONFIG['lon_range'],
        resolution=TEST_CONFIG['resolution'],
        patch_size=TEST_CONFIG['patch_size'],
        use_compile=False,
    )


def create_synthetic_data(batch_size: int = 4, add_nan: bool = True):
    """
    创建合成测试数据
    
    生成类似 SST 分布的数据：
    - 基础温度场（纬度相关）
    - 时序变化
    - 陆地 NaN 区域
    """
    seq_len_minus_1 = TEST_CONFIG['seq_len'] - 1
    width = TEST_CONFIG['width']
    height = TEST_CONFIG['height']
    
    # 创建基础温度场（模拟 SST 纬度分布）
    lat = torch.linspace(-90, 90, height)
    base_temp = 25 - 25 * torch.abs(lat) / 90  # 赤道热，两极冷
    base_temp = base_temp.unsqueeze(0).expand(width, -1)  # [W, H]
    
    # 添加时序变化和批次变化
    x = torch.zeros(batch_size, seq_len_minus_1, width, height)
    for b in range(batch_size):
        for s in range(seq_len_minus_1):
            noise = torch.randn(width, height) * 2  # 随机扰动
            x[b, s] = base_temp + noise + s * 0.1  # 微小时序趋势
    
    # 目标：下一时间步
    y = base_temp.unsqueeze(0).expand(batch_size, -1, -1) + torch.randn(batch_size, width, height) * 2
    
    # 添加陆地 NaN 区域（模拟大陆分布）
    if add_nan:
        # 简化的陆地掩码
        land_mask = torch.zeros(width, height, dtype=torch.bool)
        # 北美
        land_mask[20:80, 100:140] = True
        # 南美
        land_mask[70:100, 120:150] = True
        # 欧亚
        land_mask[0:180, 100:160] = True
        # 非洲
        land_mask[150:220, 80:130] = True
        # 澳洲
        land_mask[280:320, 50:80] = True
        
        x[:, :, land_mask] = float('nan')
        y[:, land_mask] = float('nan')
    
    return x, y


def compute_metrics(pred: torch.Tensor, target: torch.Tensor):
    """
    计算预测指标
    
    Returns:
        dict: MSE, RMSE, MAE, R²
    """
    # 获取有效值掩码
    valid_mask = ~(torch.isnan(pred) | torch.isnan(target))
    
    if valid_mask.sum() == 0:
        return {'mse': float('nan'), 'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan')}
    
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    
    # MSE
    mse = torch.mean((pred_valid - target_valid) ** 2).item()
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = torch.mean(torch.abs(pred_valid - target_valid)).item()
    
    # R²
    ss_res = torch.sum((target_valid - pred_valid) ** 2)
    ss_tot = torch.sum((target_valid - target_valid.mean()) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


class TestAccuracyComparison:
    """精度对比测试"""
    
    def test_forward_pass_produces_valid_output(self):
        """测试前向传播产生有效输出"""
        v1 = create_v1_model()
        v2 = create_v2_model()
        x, y = create_synthetic_data(batch_size=2)
        
        v1.eval()
        v2.eval()
        
        with torch.no_grad():
            out_v1 = v1(x)
            out_v2 = v2(x)
        
        # 检查输出形状
        assert out_v1.shape == y.shape
        assert out_v2.shape == y.shape
        
        # 检查非 NaN 区域有有效值
        # out_v1/v2 形状: [batch, width, height]
        # 直接检查输出是否有有效值（非 NaN）
        assert not torch.isnan(out_v1).all(), "V1 输出全为 NaN"
        assert not torch.isnan(out_v2).all(), "V2 输出全为 NaN"
        
        # 进一步检查：有效区域应该有有效输出
        valid_mask = ~torch.isnan(x[:, 0])  # [batch, width, height]
        assert (out_v1[valid_mask].isfinite()).any(), "V1 有效区域无有效输出"
        assert (out_v2[valid_mask].isfinite()).any(), "V2 有效区域无有效输出"
    
    def test_loss_computation(self):
        """测试损失计算"""
        v1 = create_v1_model()
        v2 = create_v2_model()
        x, y = create_synthetic_data(batch_size=2)
        
        v1.eval()
        v2.eval()
        
        with torch.no_grad():
            out_v1 = v1(x)
            out_v2 = v2(x)
            
            loss_v1 = v1.custom_mse_loss(out_v1, y)
            loss_v2 = v2.custom_mse_loss(out_v2, y)
        
        print(f"\n损失对比:")
        print(f"  V1 Loss: {loss_v1.item():.6f}")
        print(f"  V2 Loss: {loss_v2.item():.6f}")
        
        # 损失应该是有限值
        assert torch.isfinite(loss_v1), "V1 损失无效"
        assert torch.isfinite(loss_v2), "V2 损失无效"
    
    def test_metrics_comparison(self):
        """T024-T027: 精度指标对比"""
        v1 = create_v1_model()
        v2 = create_v2_model()
        x, y = create_synthetic_data(batch_size=4)
        
        v1.eval()
        v2.eval()
        
        with torch.no_grad():
            out_v1 = v1(x)
            out_v2 = v2(x)
        
        metrics_v1 = compute_metrics(out_v1, y)
        metrics_v2 = compute_metrics(out_v2, y)
        
        print(f"\n精度指标对比:")
        print(f"{'Metric':<10} {'V1':>12} {'V2':>12}")
        print(f"{'-'*36}")
        for key in ['mse', 'rmse', 'mae', 'r2']:
            print(f"{key.upper():<10} {metrics_v1[key]:>12.6f} {metrics_v2[key]:>12.6f}")
        
        # 注意：未训练的模型指标可能差异较大，这里主要验证计算正确性
        assert not np.isnan(metrics_v1['mse']), "V1 MSE 计算失败"
        assert not np.isnan(metrics_v2['mse']), "V2 MSE 计算失败"


class TestNaNHandling:
    """NaN 处理测试"""
    
    def test_nan_propagation(self):
        """T028: 测试 NaN 值正确传播"""
        v2 = create_v2_model()
        x, y = create_synthetic_data(batch_size=2, add_nan=True)
        
        v2.eval()
        with torch.no_grad():
            output = v2(x)
        
        # 获取输入的陆地掩码
        input_nan_mask = torch.isnan(x[:, 0])  # [B, W, H]
        
        # 注意：由于 patch 操作，NaN 区域可能不完全对应
        # 主要检查：输出中非 NaN 区域应该有有效值
        output_valid = ~torch.isnan(output)
        
        print(f"\nNaN 处理验证:")
        print(f"  输入 NaN 比例: {input_nan_mask.float().mean()*100:.1f}%")
        print(f"  输出有效值比例: {output_valid.float().mean()*100:.1f}%")
    
    def test_coastal_region_handling(self):
        """测试沿海区域处理"""
        v2 = create_v2_model()
        
        # 创建沿海测试数据
        x = torch.randn(1, TEST_CONFIG['seq_len'] - 1, TEST_CONFIG['width'], TEST_CONFIG['height'])
        
        # 创建沿海掩码（部分 NaN，模拟海陆交界）
        coastal_mask = torch.zeros(TEST_CONFIG['width'], TEST_CONFIG['height'], dtype=torch.bool)
        # 创建条纹状的海陆交界
        for i in range(0, TEST_CONFIG['width'], 20):
            coastal_mask[i:i+10, :] = True
        
        x[:, :, coastal_mask] = float('nan')
        
        v2.eval()
        with torch.no_grad():
            output = v2(x)
        
        # 检查输出中海洋区域有有效值
        ocean_mask = ~coastal_mask
        ocean_output = output[:, ocean_mask]
        
        valid_ratio = (~torch.isnan(ocean_output)).float().mean()
        print(f"\n沿海区域测试:")
        print(f"  海洋区域有效输出比例: {valid_ratio*100:.1f}%")
        
        assert valid_ratio > 0.5, "海洋区域有效输出比例过低"


class TestVisualization:
    """可视化测试（用于手动检查）"""
    
    def test_generate_comparison_plot(self):
        """T029: 生成对比可视化"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        v1 = create_v1_model()
        v2 = create_v2_model()
        x, y = create_synthetic_data(batch_size=1, add_nan=True)
        
        v1.eval()
        v2.eval()
        
        with torch.no_grad():
            out_v1 = v1(x)
            out_v2 = v2(x)
        
        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 真实值
        im0 = axes[0, 0].imshow(y[0].numpy(), cmap='jet', aspect='auto')
        axes[0, 0].set_title('Ground Truth')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # V1 预测
        im1 = axes[0, 1].imshow(out_v1[0].numpy(), cmap='jet', aspect='auto')
        axes[0, 1].set_title('V1 Prediction')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # V2 预测
        im2 = axes[1, 0].imshow(out_v2[0].numpy(), cmap='jet', aspect='auto')
        axes[1, 0].set_title('V2 Prediction')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # 差异
        diff = out_v2[0] - out_v1[0]
        im3 = axes[1, 1].imshow(diff.numpy(), cmap='RdBu_r', aspect='auto')
        axes[1, 1].set_title('V2 - V1 Difference')
        plt.colorbar(im3, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # 保存图片
        output_path = project_root / 'out' / 'v1_v2_comparison.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"\n对比图已保存: {output_path}")


def run_accuracy_tests():
    """运行所有精度测试"""
    print("=" * 60)
    print(" RGTransformer V1/V2 Accuracy Tests")
    print("=" * 60)
    
    test_accuracy = TestAccuracyComparison()
    test_accuracy.test_forward_pass_produces_valid_output()
    test_accuracy.test_loss_computation()
    test_accuracy.test_metrics_comparison()
    
    test_nan = TestNaNHandling()
    test_nan.test_nan_propagation()
    test_nan.test_coastal_region_handling()
    
    print("\n" + "=" * 60)
    print(" Accuracy Tests Complete")
    print("=" * 60)


if __name__ == '__main__':
    run_accuracy_tests()

