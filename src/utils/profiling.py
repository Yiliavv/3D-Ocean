"""
性能分析工具模块
用于 RGTransformer V1/V2 的 A/B 对比测试

功能：
- 训练/推理时间计时
- GPU 显存监控
- 参数量统计
- 性能报告生成
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    统计模型参数量
    
    Args:
        model: PyTorch 模型
        trainable_only: 是否只统计可训练参数
    
    Returns:
        参数总数
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """
    获取模型大小（MB）
    
    Args:
        model: PyTorch 模型
    
    Returns:
        模型大小（MB）
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    return total_size / (1024 ** 2)


@contextmanager
def cuda_timer(device: Optional[torch.device] = None, synchronize: bool = True):
    """
    CUDA 计时器上下文管理器
    
    Args:
        device: CUDA 设备（用于同步）
        synchronize: 是否同步 CUDA 流
    
    Yields:
        计时结果字典
    
    Example:
        with cuda_timer() as timer:
            model(x)
        print(f"耗时: {timer['elapsed_ms']:.2f} ms")
    """
    result = {'elapsed_ms': 0.0, 'elapsed_s': 0.0}
    
    if torch.cuda.is_available():
        try:
            if synchronize:
                torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            # record() 不需要参数，使用当前流
            start_event.record()
            
            yield result
            
            end_event.record()
            if synchronize:
                torch.cuda.synchronize(device)
            result['elapsed_ms'] = start_event.elapsed_time(end_event)
            result['elapsed_s'] = result['elapsed_ms'] / 1000
        except RuntimeError:
            # CUDA 不可用时回退到 CPU 计时
            start_time = time.perf_counter()
            yield result
            end_time = time.perf_counter()
            result['elapsed_s'] = end_time - start_time
            result['elapsed_ms'] = result['elapsed_s'] * 1000
    else:
        start_time = time.perf_counter()
        yield result
        end_time = time.perf_counter()
        result['elapsed_s'] = end_time - start_time
        result['elapsed_ms'] = result['elapsed_s'] * 1000


class MemoryTracker:
    """
    GPU 显存追踪器
    
    用于监控模型训练/推理时的峰值显存占用
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: CUDA 设备，默认为当前设备
        """
        self.device = device
        self.reset()
    
    def reset(self):
        """重置显存统计"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
    
    def get_current_memory_mb(self) -> float:
        """获取当前显存占用（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        return 0.0
    
    def get_peak_memory_mb(self) -> float:
        """获取峰值显存占用（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        return 0.0
    
    def get_reserved_memory_mb(self) -> float:
        """获取预留显存（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        return 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """获取完整显存统计"""
        return {
            'current_mb': self.get_current_memory_mb(),
            'peak_mb': self.get_peak_memory_mb(),
            'reserved_mb': self.get_reserved_memory_mb()
        }


class BenchmarkRunner:
    """
    性能基准测试运行器
    
    用于对比 V1/V2 模型的性能指标
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: 运行设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_tracker = MemoryTracker(self.device)
    
    def benchmark_inference(
        self, 
        model: nn.Module, 
        input_tensor: torch.Tensor,
        num_warmup: int = 10,
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        推理性能基准测试
        
        Args:
            model: 待测试模型
            input_tensor: 输入张量
            num_warmup: 预热迭代次数
            num_iterations: 正式迭代次数
        
        Returns:
            性能指标字典
        """
        model = model.to(self.device)
        model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_tensor)
        
        # 重置显存统计
        self.memory_tracker.reset()
        
        # 正式测试
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                with cuda_timer(self.device) as timer:
                    _ = model(input_tensor)
                times.append(timer['elapsed_ms'])
        
        memory_stats = self.memory_tracker.get_stats()
        
        return {
            'avg_time_ms': sum(times) / len(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
            'throughput_per_sec': 1000 / (sum(times) / len(times)),
            'peak_memory_mb': memory_stats['peak_mb'],
            'num_iterations': num_iterations
        }
    
    def benchmark_training_step(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        loss_fn: Callable,
        num_warmup: int = 5,
        num_iterations: int = 20
    ) -> Dict[str, Any]:
        """
        训练步骤性能基准测试
        
        Args:
            model: 待测试模型
            input_tensor: 输入张量
            target_tensor: 目标张量
            loss_fn: 损失函数
            num_warmup: 预热迭代次数
            num_iterations: 正式迭代次数
        
        Returns:
            性能指标字典
        """
        model = model.to(self.device)
        model.train()
        input_tensor = input_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # 预热
        for _ in range(num_warmup):
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = loss_fn(output, target_tensor)
            loss.backward()
            optimizer.step()
        
        # 重置显存统计
        self.memory_tracker.reset()
        
        # 正式测试
        times = []
        for _ in range(num_iterations):
            with cuda_timer(self.device) as timer:
                optimizer.zero_grad()
                output = model(input_tensor)
                loss = loss_fn(output, target_tensor)
                loss.backward()
                optimizer.step()
            times.append(timer['elapsed_ms'])
        
        memory_stats = self.memory_tracker.get_stats()
        
        return {
            'avg_time_ms': sum(times) / len(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
            'peak_memory_mb': memory_stats['peak_mb'],
            'num_iterations': num_iterations
        }


def compare_models(
    results_v1: Dict[str, Any],
    results_v2: Dict[str, Any],
    metrics: Optional[list] = None
) -> Dict[str, Dict[str, Any]]:
    """
    对比两个模型的性能结果
    
    Args:
        results_v1: V1 模型结果
        results_v2: V2 模型结果
        metrics: 要对比的指标列表
    
    Returns:
        对比结果字典
    """
    if metrics is None:
        metrics = ['avg_time_ms', 'peak_memory_mb', 'throughput_per_sec']
    
    comparison = {}
    for metric in metrics:
        if metric in results_v1 and metric in results_v2:
            v1_value = results_v1[metric]
            v2_value = results_v2[metric]
            
            if v1_value > 0:
                ratio = v2_value / v1_value
                improvement = (1 - ratio) * 100
            else:
                ratio = float('inf')
                improvement = 0
            
            comparison[metric] = {
                'v1': v1_value,
                'v2': v2_value,
                'ratio': ratio,
                'improvement_percent': improvement
            }
    
    return comparison


def print_comparison_report(comparison: Dict[str, Dict[str, Any]], title: str = "Performance Comparison"):
    """
    打印性能对比报告
    
    Args:
        comparison: compare_models 返回的对比结果
        title: 报告标题
    """
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")
    print(f"{'Metric':<25} {'V1':>12} {'V2':>12} {'Ratio':>10} {'Improve':>10}")
    print(f"{'-' * 60}")
    
    for metric, values in comparison.items():
        v1 = values['v1']
        v2 = values['v2']
        ratio = values['ratio']
        improve = values['improvement_percent']
        
        # 格式化数值
        if 'time' in metric.lower():
            v1_str = f"{v1:.2f} ms"
            v2_str = f"{v2:.2f} ms"
        elif 'memory' in metric.lower():
            v1_str = f"{v1:.1f} MB"
            v2_str = f"{v2:.1f} MB"
        else:
            v1_str = f"{v1:.2f}"
            v2_str = f"{v2:.2f}"
        
        improve_str = f"{improve:+.1f}%" if improve != 0 else "0.0%"
        
        print(f"{metric:<25} {v1_str:>12} {v2_str:>12} {ratio:>10.2f}x {improve_str:>10}")
    
    print(f"{'=' * 60}\n")

