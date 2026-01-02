"""
消融实验集成测试

测试完整的消融实验流程：
- 模型变体创建
- 单变体实验运行
- 结果导出
- 可视化生成
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.analysis.ablation.config import (
    AblationConfig,
    ExperimentResult,
    ExperimentState,
    ABLATION_VARIANTS,
    STYLES,
)
from src.analysis.ablation.metrics import compute_metrics
from src.analysis.ablation.variants import (
    create_variant_model,
    PatchEmbedding,
    SimpleDecoder,
    StandardAttention,
)
from src.analysis.ablation.runner import AblationRunner


class TestIntegrationAblationRunner:
    """集成测试：消融实验运行器"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """创建临时输出目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_results_df(self):
        """创建示例结果 DataFrame"""
        data = []
        
        for variant in ['baseline', 'wo_convstem', 'wo_attention', 'wo_shpe']:
            for run_id in range(1, 4):
                # 生成模拟指标
                base_rmse = 0.3 if variant == 'baseline' else 0.35 + np.random.rand() * 0.05
                
                data.append({
                    'config_name': variant,
                    'display_name': ABLATION_VARIANTS[variant].display_name,
                    'run_id': run_id,
                    'experiment_id': f'{variant}_run{run_id}',
                    'mse': base_rmse ** 2,
                    'rmse': base_rmse,
                    'mae': base_rmse * 0.8,
                    'r2': 0.95 - (base_rmse - 0.3) * 0.5,
                    'spatial_corr': 0.98 - (base_rmse - 0.3) * 0.2,
                    'train_time_seconds': 3600 + np.random.rand() * 600,
                    'inference_time_ms': 10 + np.random.rand() * 5,
                    'peak_memory_mb': 4000 + np.random.rand() * 1000,
                    'num_parameters': 5000000 + np.random.randint(-100000, 100000),
                    'seed': 42 + run_id,
                })
        
        return pd.DataFrame(data)
    
    def test_runner_initialization(self, temp_output_dir):
        """测试运行器初始化"""
        runner = AblationRunner(
            output_dir=str(temp_output_dir),
            runs_per_variant=1,
            seed=42
        )
        
        assert runner.output_dir == temp_output_dir
        assert runner.runs_per_variant == 1
        assert runner.base_seed == 42
        
        # 检查目录创建
        assert (temp_output_dir / "results").exists()
        assert (temp_output_dir / "figures").exists()
        assert (temp_output_dir / "tables").exists()
        assert (temp_output_dir / "checkpoints").exists()
    
    def test_state_persistence(self, temp_output_dir):
        """测试状态持久化"""
        state_file = temp_output_dir / "results" / "experiment_state.json"
        
        # 创建状态
        state = ExperimentState(state_file=state_file)
        
        # 标记实验完成
        result = ExperimentResult(
            config_name="test",
            display_name="Test",
            run_id=1,
            mse=0.1
        )
        state.mark_completed("test_run1", result)
        
        # 重新加载
        state2 = ExperimentState(state_file=state_file)
        assert state2.is_completed("test_run1")
        assert len(state2.results) == 1
    
    def test_export_results_csv(self, temp_output_dir, sample_results_df):
        """测试 CSV 导出"""
        runner = AblationRunner(output_dir=str(temp_output_dir))
        
        # 添加模拟结果
        for _, row in sample_results_df.iterrows():
            result = ExperimentResult(**row.to_dict())
            runner.results.append(result)
        
        # 导出
        runner.export_results_csv()
        
        # 验证文件存在
        csv_path = temp_output_dir / "results" / "ablation_results.csv"
        assert csv_path.exists()
        
        # 验证内容
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == len(sample_results_df)
        assert 'config_name' in loaded_df.columns
        assert 'rmse' in loaded_df.columns
    
    def test_save_experiment_config(self, temp_output_dir):
        """测试配置保存"""
        runner = AblationRunner(output_dir=str(temp_output_dir))
        runner.save_experiment_config()
        
        # 验证文件存在
        config_path = temp_output_dir / "results" / "experiment_configs.yaml"
        assert config_path.exists()
        
        # 验证内容
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        assert 'experiment' in config
        assert 'variants' in config
    
    def test_analyze_component(self, temp_output_dir, sample_results_df):
        """测试组件分析"""
        runner = AblationRunner(output_dir=str(temp_output_dir))
        
        # 添加模拟结果
        for _, row in sample_results_df.iterrows():
            result = ExperimentResult(**row.to_dict())
            runner.results.append(result)
        
        # 分析组件
        analysis = runner.analyze_component('convstem')
        
        assert 'component' in analysis
        assert 'baseline' in analysis
        assert 'variant' in analysis
        assert 'absolute_diff' in analysis
    
    def test_generate_efficiency_report(self, temp_output_dir, sample_results_df):
        """测试效率报告生成"""
        runner = AblationRunner(output_dir=str(temp_output_dir))
        
        # 添加模拟结果
        for _, row in sample_results_df.iterrows():
            result = ExperimentResult(**row.to_dict())
            runner.results.append(result)
        
        # 生成报告
        df = runner.generate_efficiency_report()
        
        assert len(df) > 0
        assert 'variant' in df.columns
        assert 'avg_rmse' in df.columns
        
        # 验证文件保存
        report_path = temp_output_dir / "results" / "efficiency_report.csv"
        assert report_path.exists()


class TestIntegrationVariantModels:
    """集成测试：模型变体"""
    
    def test_patch_embedding_forward(self):
        """测试 PatchEmbedding 前向传播"""
        embed = PatchEmbedding(
            in_channels=1,
            embed_dim=256,
            patch_size=4
        )
        
        x = torch.randn(2, 1, 180, 360)
        out = embed(x)
        
        assert out.shape == (2, 256, 45, 90)
    
    def test_simple_decoder_forward(self):
        """测试 SimpleDecoder 前向传播"""
        decoder = SimpleDecoder(
            in_channels=256,
            out_channels=1,
            scale_factor=4
        )
        
        x = torch.randn(2, 256, 45, 90)
        out = decoder(x)
        
        assert out.shape == (2, 1, 180, 360)
    
    def test_standard_attention_forward(self):
        """测试 StandardAttention 前向传播"""
        attn = StandardAttention(
            d_model=256,
            num_heads=8,
            num_layers=2
        )
        
        x = torch.randn(2, 100, 256)  # [batch, seq, dim]
        out = attn(x)
        
        assert out.shape == x.shape


class TestIntegrationMetrics:
    """集成测试：评估指标"""
    
    def test_compute_metrics_large_array(self):
        """测试大数组指标计算"""
        # 模拟真实数据尺寸
        y_true = np.random.rand(10, 180, 360) * 30  # SST 范围 0-30°C
        y_pred = y_true + np.random.randn(10, 180, 360) * 0.5  # 添加噪声
        
        # 添加一些 NaN（模拟陆地）
        mask = np.random.rand(180, 360) > 0.7  # 30% 陆地
        y_true[:, mask] = np.nan
        y_pred[:, mask] = np.nan
        
        metrics = compute_metrics(y_pred, y_true, exclude_nan=True)
        
        assert 'MSE' in metrics
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'R2' in metrics
        assert 'SpatialCorr' in metrics
        
        # 验证值合理
        assert metrics['RMSE'] > 0
        assert metrics['RMSE'] < 5  # 应该小于 5°C
        assert 0 < metrics['R2'] < 1
    
    def test_compute_metrics_with_torch_tensor(self):
        """测试 PyTorch 张量输入"""
        y_true = torch.randn(5, 64, 64)
        y_pred = y_true + torch.randn_like(y_true) * 0.1
        
        metrics = compute_metrics(y_pred, y_true)
        
        assert isinstance(metrics['MSE'], float)
        assert metrics['RMSE'] > 0


class TestIntegrationVisualization:
    """集成测试：可视化"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """创建临时输出目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_results_df(self):
        """创建示例结果 DataFrame"""
        data = []
        
        for variant in ['baseline', 'wo_convstem', 'wo_attention']:
            for run_id in range(1, 4):
                base_rmse = 0.3 if variant == 'baseline' else 0.35
                
                data.append({
                    'config_name': variant,
                    'display_name': ABLATION_VARIANTS[variant].display_name,
                    'run_id': run_id,
                    'rmse': base_rmse + np.random.rand() * 0.02,
                    'mae': base_rmse * 0.8,
                    'r2': 0.95,
                    'train_time_seconds': 3600,
                    'inference_time_ms': 10,
                    'peak_memory_mb': 4000,
                })
        
        return pd.DataFrame(data)
    
    def test_visualization_imports(self):
        """测试可视化模块导入"""
        from src.plot.ablation import (
            AblationVisualizer,
            TableGenerator,
        )
        
        assert AblationVisualizer is not None
        assert TableGenerator is not None
    
    def test_visualizer_initialization(self, temp_output_dir):
        """测试可视化器初始化"""
        from src.plot.ablation import AblationVisualizer
        
        visualizer = AblationVisualizer(
            style='agu',
            output_dir=str(temp_output_dir)
        )
        
        assert visualizer.style.name == 'agu'
        assert visualizer.output_dir == temp_output_dir
    
    def test_plot_performance_comparison(self, temp_output_dir, sample_results_df):
        """测试性能对比图生成"""
        from src.plot.ablation import AblationVisualizer
        import matplotlib
        matplotlib.use('Agg')  # 无头模式
        
        visualizer = AblationVisualizer(
            style='agu',
            output_dir=str(temp_output_dir),
            save_formats=['png']
        )
        
        fig = visualizer.plot_performance_comparison(
            sample_results_df,
            metric='rmse',
            figname='test_performance'
        )
        
        assert fig is not None
        
        # 验证文件生成
        output_path = temp_output_dir / 'test_performance.png'
        assert output_path.exists()
    
    def test_table_generator(self, temp_output_dir, sample_results_df):
        """测试表格生成器"""
        from src.plot.ablation import TableGenerator
        
        generator = TableGenerator(output_dir=str(temp_output_dir))
        
        latex = generator.generate_main_results_table(
            sample_results_df,
            metrics=['rmse', 'mae'],
            output_name='test_table.tex'
        )
        
        assert r'\begin{table}' in latex
        assert r'\end{table}' in latex
        
        # 验证文件生成
        output_path = temp_output_dir / 'test_table.tex'
        assert output_path.exists()


class TestEndToEndWorkflow:
    """端到端工作流测试"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """创建临时输出目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_complete_workflow(self, temp_output_dir):
        """测试完整工作流（不含实际训练）"""
        # 1. 初始化运行器
        runner = AblationRunner(
            output_dir=str(temp_output_dir),
            runs_per_variant=2,
            seed=42
        )
        
        # 2. 添加模拟结果（跳过实际训练）
        for variant_name in ['baseline', 'wo_convstem']:
            for run_id in range(1, 3):
                result = ExperimentResult(
                    config_name=variant_name,
                    display_name=ABLATION_VARIANTS[variant_name].display_name,
                    run_id=run_id,
                    mse=0.09 + np.random.rand() * 0.02,
                    rmse=0.3 + np.random.rand() * 0.03,
                    mae=0.25 + np.random.rand() * 0.02,
                    r2=0.95 - np.random.rand() * 0.05,
                    train_time_seconds=3600,
                    inference_time_ms=10,
                    peak_memory_mb=4000,
                    num_parameters=5000000,
                    seed=42 + run_id
                )
                runner.results.append(result)
                runner.state.mark_completed(result.experiment_id, result)
        
        # 3. 导出结果
        runner.export_results_csv()
        runner.save_experiment_config()
        
        # 4. 生成可视化
        from src.plot.ablation import AblationVisualizer, TableGenerator
        import matplotlib
        matplotlib.use('Agg')
        
        results_df = pd.read_csv(temp_output_dir / "results" / "ablation_results.csv")
        
        visualizer = AblationVisualizer(
            output_dir=str(temp_output_dir / "figures"),
            save_formats=['png']
        )
        visualizer.plot_performance_comparison(results_df)
        
        # 5. 生成表格
        table_gen = TableGenerator(output_dir=str(temp_output_dir / "tables"))
        table_gen.generate_main_results_table(results_df)
        
        # 验证所有输出
        assert (temp_output_dir / "results" / "ablation_results.csv").exists()
        assert (temp_output_dir / "results" / "experiment_configs.yaml").exists()
        assert (temp_output_dir / "figures" / "performance_comparison.png").exists()
        assert (temp_output_dir / "tables" / "main_results.tex").exists()

