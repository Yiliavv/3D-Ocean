"""
消融实验模块单元测试

测试:
- AblationConfig 配置验证
- ExperimentResult 数据类
- ExperimentState 状态管理
- compute_metrics 指标计算
- significance_test 显著性检验
- create_variant_model 模型变体创建
- VisualizationStyle 样式配置
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path

from src.analysis.ablation.config import (
    AblationConfig,
    ExperimentResult,
    ExperimentState,
    VisualizationStyle,
    ABLATION_VARIANTS,
    STYLES,
)
from src.analysis.ablation.metrics import (
    compute_metrics,
    significance_test,
    compute_component_contribution,
)
from src.analysis.ablation.variants import (
    PatchEmbedding,
    NoSphericalEncoding,
    SimpleDecoder,
    StandardAttention,
    get_variant_description,
)


class TestAblationConfig:
    """测试 AblationConfig 配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = AblationConfig(name="test")
        
        assert config.name == "test"
        assert config.display_name == "Test"  # 自动生成
        assert config.use_conv_stem == True
        assert config.use_efficient_attention == True
        assert config.use_spherical_encoding == True
        assert config.use_multiscale_decoder == True
        assert config.use_gated_residual == True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = AblationConfig(
            name="wo_convstem",
            display_name="w/o ConvStem",
            use_conv_stem=False
        )
        
        assert config.name == "wo_convstem"
        assert config.display_name == "w/o ConvStem"
        assert config.use_conv_stem == False
    
    def test_to_dict(self):
        """测试字典转换"""
        config = AblationConfig(name="test", use_conv_stem=False)
        d = config.to_dict()
        
        assert isinstance(d, dict)
        assert d['name'] == "test"
        assert d['use_conv_stem'] == False
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'name': 'test',
            'display_name': 'Test Config',
            'use_conv_stem': False,
        }
        config = AblationConfig.from_dict(data)
        
        assert config.name == 'test'
        assert config.display_name == 'Test Config'
        assert config.use_conv_stem == False
    
    def test_get_model_kwargs(self):
        """测试模型参数合并"""
        config = AblationConfig(name="test", d_model=128)
        base_params = {'d_model': 256, 'num_heads': 8}
        
        kwargs = config.get_model_kwargs(base_params)
        
        assert kwargs['d_model'] == 128  # 覆盖值
        assert kwargs['num_heads'] == 8  # 基础值
    
    def test_predefined_variants(self):
        """测试预定义变体"""
        assert 'baseline' in ABLATION_VARIANTS
        assert 'wo_convstem' in ABLATION_VARIANTS
        assert 'wo_attention' in ABLATION_VARIANTS
        assert 'wo_shpe' in ABLATION_VARIANTS
        assert 'wo_multiscale' in ABLATION_VARIANTS
        assert 'wo_gate' in ABLATION_VARIANTS
        
        # 验证 baseline 是完整模型
        baseline = ABLATION_VARIANTS['baseline']
        assert baseline.use_conv_stem == True
        assert baseline.use_efficient_attention == True
        
        # 验证变体配置
        wo_convstem = ABLATION_VARIANTS['wo_convstem']
        assert wo_convstem.use_conv_stem == False


class TestExperimentResult:
    """测试 ExperimentResult 数据类"""
    
    def test_default_result(self):
        """测试默认结果"""
        result = ExperimentResult(
            config_name="baseline",
            display_name="Baseline",
            run_id=1
        )
        
        assert result.experiment_id == "baseline_run1"
        assert result.mse == 0.0
        assert result.timestamp != ""
    
    def test_custom_result(self):
        """测试自定义结果"""
        result = ExperimentResult(
            config_name="baseline",
            display_name="Baseline",
            run_id=1,
            mse=0.1,
            rmse=0.316,
            mae=0.2,
            r2=0.95,
            train_time_seconds=3600.0
        )
        
        assert result.mse == 0.1
        assert result.rmse == 0.316
        assert result.train_time_seconds == 3600.0
    
    def test_to_dict(self):
        """测试字典转换"""
        result = ExperimentResult(
            config_name="test",
            display_name="Test",
            run_id=1,
            mse=0.1
        )
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert d['mse'] == 0.1


class TestExperimentState:
    """测试 ExperimentState 状态管理"""
    
    def test_state_persistence(self):
        """测试状态持久化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            
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
            
            # 验证保存
            assert state_file.exists()
            assert state.is_completed("test_run1")
            
            # 重新加载
            state2 = ExperimentState(state_file=state_file)
            assert state2.is_completed("test_run1")
    
    def test_get_pending_experiments(self):
        """测试获取待运行实验"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state = ExperimentState(state_file=state_file)
            
            # 标记一个完成
            result = ExperimentResult(
                config_name="test",
                display_name="Test",
                run_id=1
            )
            state.mark_completed("exp1", result)
            
            # 获取待运行
            all_exp = ["exp1", "exp2", "exp3"]
            pending = state.get_pending_experiments(all_exp)
            
            assert "exp1" not in pending
            assert "exp2" in pending
            assert "exp3" in pending


class TestComputeMetrics:
    """测试 compute_metrics 指标计算"""
    
    def test_perfect_prediction(self):
        """测试完美预测"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        metrics = compute_metrics(y_pred, y_true)
        
        assert metrics['MSE'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['RMSE'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['MAE'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['R2'] == pytest.approx(1.0, abs=1e-6)
    
    def test_with_error(self):
        """测试有误差的预测"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        metrics = compute_metrics(y_pred, y_true)
        
        assert metrics['MSE'] == pytest.approx(0.01, abs=1e-6)
        assert metrics['RMSE'] == pytest.approx(0.1, abs=1e-6)
        assert metrics['MAE'] == pytest.approx(0.1, abs=1e-6)
        assert metrics['R2'] > 0.99
    
    def test_with_nan(self):
        """测试含 NaN 的数据"""
        y_true = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        metrics = compute_metrics(y_pred, y_true, exclude_nan=True)
        
        # 应该只计算非 NaN 位置
        assert not np.isnan(metrics['MSE'])
        assert not np.isnan(metrics['RMSE'])
    
    def test_2d_input(self):
        """测试 2D 输入"""
        y_true = np.random.rand(10, 20)
        y_pred = y_true + np.random.rand(10, 20) * 0.1
        
        metrics = compute_metrics(y_pred, y_true)
        
        assert 'MSE' in metrics
        assert 'RMSE' in metrics
        assert metrics['RMSE'] > 0
    
    def test_3d_input(self):
        """测试 3D 输入 (batch)"""
        y_true = np.random.rand(5, 10, 20)
        y_pred = y_true + np.random.rand(5, 10, 20) * 0.1
        
        metrics = compute_metrics(y_pred, y_true)
        
        assert 'MSE' in metrics
        assert 'SpatialCorr' in metrics


class TestSignificanceTest:
    """测试 significance_test 显著性检验"""
    
    def test_identical_samples(self):
        """测试相同样本"""
        baseline = [0.5, 0.5, 0.5]
        variant = [0.5, 0.5, 0.5]
        
        p_value, sig, effect = significance_test(baseline, variant)
        
        assert p_value > 0.05  # 不显著
        assert sig == ''
    
    def test_significant_difference(self):
        """测试显著差异"""
        baseline = [0.1, 0.11, 0.09]
        variant = [0.5, 0.51, 0.49]
        
        p_value, sig, effect = significance_test(baseline, variant)
        
        assert p_value < 0.05
        assert sig in ['*', '**', '***']
        assert effect != 0
    
    def test_significance_markers(self):
        """测试显著性标记"""
        # 高度显著
        baseline = [1.0, 1.01, 0.99]
        variant = [2.0, 2.01, 1.99]
        
        _, sig, _ = significance_test(baseline, variant)
        assert sig in ['*', '**', '***']


class TestComponentContribution:
    """测试组件贡献度计算"""
    
    def test_contribution_calculation(self):
        """测试贡献度计算"""
        baseline = {'RMSE': 0.3, 'MAE': 0.2}
        variant = {'RMSE': 0.35, 'MAE': 0.25}
        
        result = compute_component_contribution(baseline, variant, 'RMSE')
        
        assert result['baseline'] == 0.3
        assert result['variant'] == 0.35
        assert result['absolute_diff'] == pytest.approx(0.05, abs=1e-6)
        assert result['performance_change'] == 'degraded'  # RMSE 增加 = 性能下降


class TestVariantModules:
    """测试变体模块"""
    
    def test_patch_embedding(self):
        """测试 PatchEmbedding"""
        embed = PatchEmbedding(
            in_channels=1,
            embed_dim=256,
            patch_size=4
        )
        
        x = torch.randn(2, 1, 64, 64)
        out = embed(x)
        
        assert out.shape == (2, 256, 16, 16)
    
    def test_no_spherical_encoding(self):
        """测试空球谐波编码"""
        enc = NoSphericalEncoding(height=64, width=128)
        out = enc()
        
        assert out.shape == (64, 128)
        assert torch.all(out == 0)
    
    def test_simple_decoder(self):
        """测试简单解码器"""
        decoder = SimpleDecoder(
            in_channels=256,
            out_channels=1,
            scale_factor=4
        )
        
        x = torch.randn(2, 256, 16, 16)
        out = decoder(x)
        
        assert out.shape == (2, 1, 64, 64)
    
    def test_standard_attention(self):
        """测试标准注意力"""
        attn = StandardAttention(
            d_model=256,
            num_heads=8,
            num_layers=2
        )
        
        x = torch.randn(2, 10, 256)
        out = attn(x)
        
        assert out.shape == x.shape


class TestVisualizationStyle:
    """测试可视化样式配置"""
    
    def test_default_style(self):
        """测试默认样式"""
        style = VisualizationStyle(name="test")
        
        assert style.dpi == 300
        assert len(style.primary_colors) == 6
    
    def test_predefined_styles(self):
        """测试预定义样式"""
        assert 'agu' in STYLES
        assert 'ieee' in STYLES
        assert 'nature' in STYLES
        
        agu = STYLES['agu']
        assert agu.dpi == 300
    
    def test_get_rcparams(self):
        """测试获取 matplotlib 参数"""
        style = VisualizationStyle(name="test", font_size=12)
        params = style.get_matplotlib_rcparams()
        
        assert params['font.size'] == 12
        assert params['savefig.dpi'] == 300
    
    def test_get_figure_size(self):
        """测试获取图表尺寸"""
        style = VisualizationStyle(
            name="test",
            single_column_width=3.5,
            double_column_width=7.0
        )
        
        single = style.get_figure_size(width='single', aspect=0.75)
        double = style.get_figure_size(width='double', aspect=0.75)
        
        assert single[0] == 3.5
        assert double[0] == 7.0


class TestGetVariantDescription:
    """测试变体描述生成"""
    
    def test_baseline_description(self):
        """测试 baseline 描述"""
        config = AblationConfig(name="baseline")
        desc = get_variant_description(config)
        
        assert "Baseline" in desc or "no modifications" in desc
    
    def test_wo_convstem_description(self):
        """测试 w/o ConvStem 描述"""
        config = AblationConfig(name="wo_convstem", use_conv_stem=False)
        desc = get_variant_description(config)
        
        assert "ConvStem" in desc
        assert "PatchEmbedding" in desc

