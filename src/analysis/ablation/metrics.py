"""
消融实验评估指标模块

包含:
- compute_metrics: 计算所有评估指标（排除 NaN 区域）
- significance_test: 统计显著性检验
- compute_component_contribution: 计算组件贡献度
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import time
import torch


def compute_metrics(
    y_pred: np.ndarray, 
    y_true: np.ndarray,
    exclude_nan: bool = True
) -> Dict[str, float]:
    """
    计算所有评估指标（排除 NaN 区域）
    
    遵循 Constitution II. Data Integrity & NaN Handling 原则，
    所有指标计算排除 NaN 区域。
    
    Args:
        y_pred: 预测值 [N, H, W] 或 [H, W]
        y_true: 真实值 [N, H, W] 或 [H, W]
        exclude_nan: 是否排除 NaN 值
    
    Returns:
        dict: 包含 MSE, RMSE, MAE, R², 空间相关系数
    """
    # 确保是 numpy 数组
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    # 展平为一维（如果需要）
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    
    if exclude_nan:
        # 创建有效值掩码（排除 NaN 和异常值）
        valid_mask = ~(
            np.isnan(y_pred_flat) | 
            np.isnan(y_true_flat) |
            (y_true_flat > 99) |   # 排除 > 99°C
            (y_true_flat < -10)    # 排除 < -10°C
        )
        
        if valid_mask.sum() == 0:
            return {
                'MSE': float('nan'),
                'RMSE': float('nan'),
                'MAE': float('nan'),
                'R2': float('nan'),
                'SpatialCorr': float('nan')
            }
        
        y_pred_valid = y_pred_flat[valid_mask]
        y_true_valid = y_true_flat[valid_mask]
    else:
        y_pred_valid = y_pred_flat
        y_true_valid = y_true_flat
    
    # 计算基础指标
    mse = float(np.mean((y_pred_valid - y_true_valid) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred_valid - y_true_valid)))
    
    # 计算 R²
    ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
    ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # 计算空间相关系数
    spatial_corr = _compute_spatial_correlation(y_pred, y_true, exclude_nan)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'SpatialCorr': spatial_corr
    }


def _compute_spatial_correlation(
    y_pred: np.ndarray,
    y_true: np.ndarray, 
    exclude_nan: bool = True
) -> float:
    """
    计算空间相关系数（每个样本的平均）
    
    Args:
        y_pred: 预测值
        y_true: 真实值
        exclude_nan: 是否排除 NaN
        
    Returns:
        平均空间相关系数
    """
    # 确保是 3D
    if y_pred.ndim == 2:
        y_pred = y_pred[np.newaxis, ...]
        y_true = y_true[np.newaxis, ...]
    
    correlations = []
    for i in range(len(y_pred)):
        pred_i = y_pred[i].flatten()
        true_i = y_true[i].flatten()
        
        if exclude_nan:
            mask_i = ~(np.isnan(pred_i) | np.isnan(true_i))
            pred_i = pred_i[mask_i]
            true_i = true_i[mask_i]
        
        if len(pred_i) > 10:
            try:
                corr = np.corrcoef(pred_i, true_i)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            except Exception:
                pass
    
    return float(np.mean(correlations)) if correlations else 0.0


def significance_test(
    baseline_scores: List[float],
    variant_scores: List[float],
    test_type: str = 'paired_t',
    alpha: float = 0.05
) -> Tuple[float, str, float]:
    """
    计算统计显著性
    
    使用配对 t 检验作为主要方法，支持 Wilcoxon 检验作为非参数备选。
    
    Args:
        baseline_scores: 基准模型的分数列表
        variant_scores: 变体模型的分数列表
        test_type: 检验类型 'paired_t' 或 'wilcoxon'
        alpha: 显著性水平
        
    Returns:
        (p_value, significance_marker, effect_size)
        - p_value: p 值
        - significance_marker: '', '*', '**', '***'
        - effect_size: Cohen's d 效应量
    """
    baseline = np.array(baseline_scores)
    variant = np.array(variant_scores)
    
    if len(baseline) != len(variant):
        raise ValueError("Baseline and variant must have same length")
    
    if len(baseline) < 2:
        return 1.0, '', 0.0
    
    # 检查数据是否完全相同（或非常接近）
    diff = baseline - variant
    if np.allclose(diff, 0, atol=1e-10):
        # 数据完全相同，无显著差异
        return 1.0, '', 0.0
    
    # 选择检验方法
    if test_type == 'paired_t':
        t_stat, p_value = stats.ttest_rel(baseline, variant)
        # 处理 NaN 情况
        if np.isnan(p_value):
            return 1.0, '', 0.0
    elif test_type == 'wilcoxon':
        # Wilcoxon 符号秩检验
        try:
            stat, p_value = stats.wilcoxon(baseline, variant)
        except ValueError:
            # 如果数据全相同，返回不显著
            return 1.0, '', 0.0
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # 计算效应量 (Cohen's d)
    diff = baseline - variant
    effect_size = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.0
    
    # 显著性标记
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < alpha:
        significance = '*'
    else:
        significance = ''
    
    return float(p_value), significance, effect_size


def compute_component_contribution(
    baseline_metrics: Dict[str, float],
    variant_metrics: Dict[str, float],
    metric_key: str = 'RMSE'
) -> Dict[str, float]:
    """
    计算组件贡献度
    
    通过比较 baseline 和移除组件后的变体，量化组件的贡献。
    
    Args:
        baseline_metrics: 基准模型指标
        variant_metrics: 变体模型指标
        metric_key: 用于计算贡献的指标
        
    Returns:
        dict: 包含绝对贡献、相对贡献、改善/恶化方向
    """
    baseline_val = baseline_metrics.get(metric_key, 0)
    variant_val = variant_metrics.get(metric_key, 0)
    
    # 绝对差异
    absolute_diff = variant_val - baseline_val
    
    # 相对差异（百分比）
    relative_diff = (absolute_diff / baseline_val * 100) if baseline_val != 0 else 0
    
    # 对于 RMSE/MSE/MAE，值增加表示性能下降
    # 对于 R²/SpatialCorr，值减少表示性能下降
    is_error_metric = metric_key in ['MSE', 'RMSE', 'MAE']
    performance_change = 'degraded' if (is_error_metric and absolute_diff > 0) or \
                                        (not is_error_metric and absolute_diff < 0) \
                         else 'improved'
    
    return {
        'baseline': baseline_val,
        'variant': variant_val,
        'absolute_diff': absolute_diff,
        'relative_diff_percent': relative_diff,
        'performance_change': performance_change
    }


def benchmark_inference(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    推理速度基准测试
    
    Args:
        model: PyTorch 模型
        sample_input: 样本输入张量
        num_runs: 测试运行次数
        warmup_runs: 预热运行次数
        device: 设备
        
    Returns:
        dict: 包含平均/最小/最大推理时间
    """
    model.eval()
    model.to(device)
    sample_input = sample_input.to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)
    
    # 同步 CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 计时
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(sample_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转换为毫秒
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times))
    }


def measure_peak_memory(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    测量峰值显存占用
    
    Args:
        model: PyTorch 模型
        sample_input: 样本输入张量
        device: 设备
        
    Returns:
        dict: 包含峰值显存、分配显存等
    """
    if device != 'cuda' or not torch.cuda.is_available():
        return {
            'peak_memory_mb': 0.0,
            'allocated_memory_mb': 0.0,
            'reserved_memory_mb': 0.0
        }
    
    model.to(device)
    sample_input = sample_input.to(device)
    
    # 重置峰值统计
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        _ = model(sample_input)
    
    # 获取显存统计
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)
    
    return {
        'peak_memory_mb': float(peak_memory),
        'allocated_memory_mb': float(allocated_memory),
        'reserved_memory_mb': float(reserved_memory)
    }


def compute_all_significance_tests(
    results_by_variant: Dict[str, List[Dict[str, float]]],
    baseline_key: str = 'baseline',
    metrics: List[str] = None
) -> List[Dict]:
    """
    计算所有变体相对于 baseline 的统计显著性
    
    Args:
        results_by_variant: {variant_name: [run1_metrics, run2_metrics, ...]}
        baseline_key: 基准变体名称
        metrics: 要检验的指标列表
        
    Returns:
        显著性检验结果列表
    """
    if metrics is None:
        metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'SpatialCorr']
    
    if baseline_key not in results_by_variant:
        return []
    
    baseline_results = results_by_variant[baseline_key]
    test_results = []
    
    for variant_name, variant_results in results_by_variant.items():
        if variant_name == baseline_key:
            continue
            
        for metric in metrics:
            baseline_scores = [r[metric] for r in baseline_results]
            variant_scores = [r[metric] for r in variant_results]
            
            p_value, significance, effect_size = significance_test(
                baseline_scores, variant_scores
            )
            
            test_results.append({
                'variant': variant_name,
                'metric': metric,
                'baseline_mean': float(np.mean(baseline_scores)),
                'baseline_std': float(np.std(baseline_scores)),
                'variant_mean': float(np.mean(variant_scores)),
                'variant_std': float(np.std(variant_scores)),
                'p_value': p_value,
                'significance': significance,
                'effect_size': effect_size
            })
    
    return test_results

