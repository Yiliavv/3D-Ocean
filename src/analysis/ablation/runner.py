"""
消融实验运行器模块

提供完整的消融实验执行框架，支持:
- 单变体训练和评估
- 多变体批量运行
- 断点续传
- 结果导出
"""

import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import pandas as pd
import yaml
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.analysis.ablation.config import (
    AblationConfig,
    ExperimentResult,
    ExperimentState,
    ABLATION_VARIANTS,
    load_config_from_yaml,
)
from src.analysis.ablation.metrics import (
    compute_metrics,
    compute_all_significance_tests,
    benchmark_inference,
    measure_peak_memory,
)
from src.analysis.ablation.variants import create_variant_model, get_variant_description


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AblationRunner:
    """
    消融实验运行器
    
    管理完整的消融实验流程，包括模型训练、评估、结果导出。
    支持断点续传和并行任务。
    
    错误处理策略:
    - GPU 显存不足: 自动降低 batch size 重试
    - 训练失败: 记录错误并跳过，继续下一个实验
    """
    
    # 默认 batch size 降级序列
    BATCH_SIZE_FALLBACK = [24, 16, 8, 4, 2]
    
    def __init__(
        self,
        output_dir: str = "out/ablation",
        config_path: Optional[str] = None,
        runs_per_variant: int = 3,
        seed: int = 42,
        device: str = "auto",
        auto_reduce_batch: bool = True
    ):
        """
        Args:
            output_dir: 输出目录
            config_path: 配置文件路径
            runs_per_variant: 每变体运行次数
            seed: 随机种子
            device: 设备 ("auto", "cuda", "cpu")
            auto_reduce_batch: 是否自动降低 batch size
        """
        self.output_dir = Path(output_dir)
        self.runs_per_variant = runs_per_variant
        self.base_seed = seed
        self.device = self._resolve_device(device)
        self.auto_reduce_batch = auto_reduce_batch
        self.current_batch_size = None  # 动态调整
        
        # 缓存的数据集（避免重复加载）
        self._cached_dataset = None
        self._cached_datamodule = None
        
        # 创建输出目录
        self._setup_directories()
        
        # 加载配置
        if config_path:
            self.config = load_config_from_yaml(Path(config_path))
        else:
            self.config = self._get_default_config()
        
        # 初始化状态管理
        self.state = ExperimentState(
            state_file=self.output_dir / "results" / "experiment_state.json"
        )
        
        # 结果存储
        self.results: List[ExperimentResult] = list(self.state.results)
        
        logger.info(f"AblationRunner initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Runs per variant: {self.runs_per_variant}")
        logger.info(f"Device: {self.device}")
    
    def _resolve_device(self, device: str) -> str:
        """解析设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _setup_directories(self):
        """创建输出目录结构"""
        dirs = [
            self.output_dir / "results",
            self.output_dir / "figures",
            self.output_dir / "tables",
            self.output_dir / "checkpoints",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'experiment': {
                'name': 'RGTransformer_Ablation_Study',
                'runs_per_variant': self.runs_per_variant,
                'seed': self.base_seed,
            },
            'training': {
                'epochs': 100,
                'batch_size': 24,
                'learning_rate': 0.001,
                'num_workers': 4,
            },
            'model': {
                'd_model': 512,
                'num_heads': 8,
                'dim_feedforward': 256,
                'num_attn_layers': 2,
                'patch_size': 4,
            }
        }
    
    def _get_model_params(self) -> dict:
        """从配置获取模型参数"""
        # 从 trainer/config.py 导入基础参数
        try:
            from src.trainer.config import model_params, dataset_params
            return model_params.copy()
        except ImportError:
            # 回退到配置文件
            return self.config.get('model', {})
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def run_single_experiment(
        self,
        config: AblationConfig,
        run_id: int,
        datamodule = None
    ) -> ExperimentResult:
        """
        运行单个消融实验
        
        Args:
            config: 消融配置
            run_id: 运行编号
            datamodule: 数据模块（可选，用于测试）
            
        Returns:
            ExperimentResult 实验结果
        """
        experiment_id = f"{config.name}_run{run_id}"
        logger.info(f"Starting experiment: {experiment_id}")
        logger.info(f"Variant: {config.display_name}")
        logger.info(f"Description: {get_variant_description(config)}")
        
        # 设置种子
        seed = self.base_seed + run_id
        self._set_seed(seed)
        
        # 标记开始
        self.state.mark_started(experiment_id)
        
        start_time = time.time()
        
        try:
            # 获取模型参数
            model_params = self._get_model_params()
            
            # 创建模型变体
            model = create_variant_model(config, model_params)
            num_parameters = model.get_num_parameters()
            
            logger.info(f"Model parameters: {num_parameters:,}")
            
            # 创建数据模块（如果未提供）
            if datamodule is None:
                datamodule = self._create_datamodule()
            
            # 配置 Trainer
            checkpoint_dir = self.output_dir / "checkpoints" / config.name / f"run_{run_id}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            callbacks = [
                ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename="best-{epoch:02d}-{val_loss:.4f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.get('training', {}).get('early_stopping_patience', 10),
                    mode="min",
                ),
            ]
            
            # 启用 Tensor Cores 优化（适用于 RTX 系列 GPU）
            if torch.cuda.is_available():
                torch.set_float32_matmul_precision('high')
            
            trainer = Trainer(
                max_epochs=self.config.get('training', {}).get('epochs', 100),
                accelerator=self.device,
                devices=1,
                callbacks=callbacks,
                enable_progress_bar=True,
                logger=False,  # 禁用默认 logger
                precision='16-mixed',  # 混合精度训练
                num_sanity_val_steps=0,  # 跳过验证检查
            )
            
            # 训练
            trainer.fit(model, datamodule)
            
            # 加载最佳模型
            best_model_path = checkpoint_dir / "best.ckpt"
            if list(checkpoint_dir.glob("best-*.ckpt")):
                best_model_path = list(checkpoint_dir.glob("best-*.ckpt"))[0]
            
            # 评估
            trainer.validate(model, datamodule)
            
            # 获取预测结果并计算指标
            model.eval()
            metrics = self._evaluate_model(model, datamodule)
            
            train_time = time.time() - start_time
            
            # 基准测试（如果有 GPU）
            inference_stats = {'mean_ms': 0.0}
            memory_stats = {'peak_memory_mb': 0.0}
            
            if self.device == 'cuda' and torch.cuda.is_available():
                try:
                    sample_input = self._get_sample_input(datamodule)
                    inference_stats = benchmark_inference(model, sample_input, device=self.device)
                    memory_stats = measure_peak_memory(model, sample_input, device=self.device)
                except Exception as e:
                    logger.warning(f"Benchmark failed: {e}")
            
            # 创建结果
            result = ExperimentResult(
                config_name=config.name,
                display_name=config.display_name,
                run_id=run_id,
                mse=metrics.get('MSE', 0),
                rmse=metrics.get('RMSE', 0),
                mae=metrics.get('MAE', 0),
                r2=metrics.get('R2', 0),
                spatial_corr=metrics.get('SpatialCorr', 0),
                train_time_seconds=train_time,
                inference_time_ms=inference_stats.get('mean_ms', 0),
                peak_memory_mb=memory_stats.get('peak_memory_mb', 0),
                num_parameters=num_parameters,
                checkpoint_path=str(best_model_path),
                seed=seed,
            )
            
            # 标记完成
            self.state.mark_completed(experiment_id, result)
            self.results.append(result)
            
            logger.info(f"Experiment {experiment_id} completed")
            logger.info(f"RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}, R²: {result.r2:.4f}")
            logger.info(f"Train time: {train_time:.1f}s")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Experiment {experiment_id} failed: {error_msg}")
            self.state.mark_failed(experiment_id, error_msg)
            raise
    
    def _create_datamodule(self, batch_size: Optional[int] = None, force_new: bool = False):
        """
        创建或获取缓存的数据模块
        
        Args:
            batch_size: 可选的 batch size 覆盖（用于 OOM 重试）
            force_new: 是否强制创建新的 DataModule
        """
        try:
            from src.trainer.config import dataset_params, trainer_params, area
            from src.dataset.OISST import OISSTMonthlyDataset
            from torch.utils.data import DataLoader, Subset
            from lightning import LightningDataModule
            
            # 使用指定的 batch size 或默认值
            bs = batch_size or self.current_batch_size or trainer_params.get('batch_size', 24)
            num_workers = trainer_params.get('num_workers', 8)  # 增加默认 workers
            
            # 如果数据集已缓存且只需要更新 batch_size，直接更新
            if self._cached_datamodule is not None and not force_new:
                self._cached_datamodule.batch_size = bs
                return self._cached_datamodule
            
            # 保存对 runner 实例的引用（用于访问缓存的数据集）
            runner = self
            
            # 创建内部 DataModule
            class InternalDataModule(LightningDataModule):
                def __init__(self, batch_size, num_workers, **kwargs):
                    super().__init__()
                    self.batch_size = batch_size
                    self.num_workers = num_workers
                    self.dataset_kwargs = kwargs
                    self.train_dataset = None
                    self.val_dataset = None
                
                def setup(self, stage=None):
                    # 使用 runner 的缓存数据集
                    if runner._cached_dataset is None:
                        # 创建完整数据集（只创建一次）
                        logger.info("Loading dataset (first time)...")
                        runner._cached_dataset = OISSTMonthlyDataset(
                            lon=area.lon,
                            lat=area.lat,
                            **self.dataset_kwargs
                        )
                        logger.info(f"Dataset loaded: {len(runner._cached_dataset)} samples")
                    
                    full_dataset = runner._cached_dataset
                    
                    # 划分训练集和验证集 (90/10，与 BaseTrainer 一致)
                    total_len = len(full_dataset)
                    train_len = int(total_len * 0.9)
                    
                    # 使用 Subset 按时间顺序分割（与 BaseTrainer 一致）
                    train_indices = list(range(train_len))
                    val_indices = list(range(train_len, total_len))
                    
                    self.train_dataset = Subset(full_dataset, train_indices)
                    self.val_dataset = Subset(full_dataset, val_indices)
                
                def train_dataloader(self):
                    kwargs = {
                        'batch_size': self.batch_size,
                        'shuffle': False,  # 时序数据不 shuffle
                        'num_workers': self.num_workers,
                        'pin_memory': True if torch.cuda.is_available() else False,
                    }
                    # Windows 上 persistent_workers 可能有问题，只在 num_workers > 0 时启用
                    if self.num_workers > 0:
                        kwargs['persistent_workers'] = True
                        kwargs['prefetch_factor'] = 2
                    return DataLoader(self.train_dataset, **kwargs)
                
                def val_dataloader(self):
                    val_workers = max(0, self.num_workers // 2)
                    kwargs = {
                        'batch_size': self.batch_size,
                        'shuffle': False,
                        'num_workers': val_workers,
                        'pin_memory': True if torch.cuda.is_available() else False,
                    }
                    if val_workers > 0:
                        kwargs['persistent_workers'] = True
                        kwargs['prefetch_factor'] = 2
                    return DataLoader(self.val_dataset, **kwargs)
            
            datamodule = InternalDataModule(
                batch_size=bs,
                num_workers=num_workers,
                **dataset_params
            )
            
            # 缓存 DataModule
            self._cached_datamodule = datamodule
            
            return datamodule
            
        except Exception as e:
            logger.warning(f"Could not create data module: {e}")
            return None
    
    def _try_with_reduced_batch(
        self,
        config: AblationConfig,
        run_id: int,
        initial_batch_size: int = 24
    ) -> ExperimentResult:
        """
        尝试运行实验，OOM 时自动降低 batch size（复用缓存的数据集）
        
        Args:
            config: 消融配置
            run_id: 运行编号
            initial_batch_size: 初始 batch size
            
        Returns:
            ExperimentResult
            
        Raises:
            RuntimeError: 所有 batch size 都失败
        """
        batch_sizes = [bs for bs in self.BATCH_SIZE_FALLBACK if bs <= initial_batch_size]
        
        last_error = None
        for batch_size in batch_sizes:
            try:
                logger.info(f"Trying with batch_size={batch_size}")
                self.current_batch_size = batch_size
                
                # 清理 GPU 缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 获取或更新 DataModule（复用缓存的数据集）
                datamodule = self._create_datamodule(batch_size)
                return self.run_single_experiment(config, run_id, datamodule)
                
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                error_str = str(e).lower()
                if 'out of memory' in error_str or 'cuda' in error_str:
                    logger.warning(f"OOM with batch_size={batch_size}, trying smaller...")
                    last_error = e
                    
                    # 清理
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        raise RuntimeError(f"All batch sizes failed. Last error: {last_error}")
    
    def _get_sample_input(self, datamodule) -> torch.Tensor:
        """获取样本输入用于基准测试"""
        try:
            datamodule.setup('fit')
            batch = next(iter(datamodule.train_dataloader()))
            x, _ = batch
            return x[:1]  # 只取一个样本
        except Exception:
            # 返回默认形状的随机张量
            return torch.randn(1, 1, 180, 360)
    
    def _evaluate_model(self, model, datamodule) -> Dict[str, float]:
        """评估模型"""
        import numpy as np
        
        model.eval()
        all_preds = []
        all_targets = []
        
        try:
            datamodule.setup('validate')
            val_loader = datamodule.val_dataloader()
            
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x = x.to(model.device)
                    y_pred = model(x)
                    
                    all_preds.append(y_pred.cpu().numpy())
                    all_targets.append(y.cpu().numpy())
            
            preds = np.concatenate(all_preds, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            
            return compute_metrics(preds, targets)
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return {'MSE': 0, 'RMSE': 0, 'MAE': 0, 'R2': 0, 'SpatialCorr': 0}
    
    def run_all_variants(
        self,
        variants: Optional[List[str]] = None,
        resume: bool = True
    ) -> List[ExperimentResult]:
        """
        运行所有消融变体
        
        Args:
            variants: 要运行的变体列表（None 表示所有）
            resume: 是否从断点恢复
            
        Returns:
            所有实验结果列表
        """
        # 获取要运行的变体
        if variants is None:
            variants = list(ABLATION_VARIANTS.keys())
        
        # 生成所有实验 ID
        all_experiments = []
        for variant_name in variants:
            for run_id in range(1, self.runs_per_variant + 1):
                all_experiments.append((variant_name, run_id))
        
        # 获取待运行的实验
        if resume:
            pending = []
            for variant_name, run_id in all_experiments:
                exp_id = f"{variant_name}_run{run_id}"
                if not self.state.is_completed(exp_id):
                    pending.append((variant_name, run_id))
            logger.info(f"Resuming: {len(pending)} experiments pending, "
                       f"{len(all_experiments) - len(pending)} already completed")
        else:
            pending = all_experiments
            logger.info(f"Starting fresh: {len(pending)} experiments to run")
        
        # 创建数据模块（共享）
        datamodule = self._create_datamodule()
        
        # 运行实验
        results = []
        failed_experiments = []
        total = len(pending)
        
        with tqdm(total=total, desc="Ablation Study") as pbar:
            for variant_name, run_id in pending:
                config = ABLATION_VARIANTS[variant_name]
                experiment_id = f"{variant_name}_run{run_id}"
                
                try:
                    # 如果启用自动降低 batch size，使用包装方法
                    if self.auto_reduce_batch:
                        initial_bs = self.config.get('training', {}).get('batch_size', 24)
                        result = self._try_with_reduced_batch(config, run_id, initial_bs)
                    else:
                        result = self.run_single_experiment(
                            config, run_id, datamodule
                        )
                    results.append(result)
                    
                except torch.cuda.OutOfMemoryError as e:
                    # OOM 错误 - 记录并继续
                    error_msg = f"GPU Out of Memory: {e}"
                    logger.error(f"Failed {experiment_id}: {error_msg}")
                    self.state.mark_failed(experiment_id, error_msg)
                    failed_experiments.append((experiment_id, error_msg))
                    
                    # 清理 GPU 缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    # 其他错误 - 记录并继续
                    error_msg = str(e)
                    logger.error(f"Failed {experiment_id}: {error_msg}")
                    self.state.mark_failed(experiment_id, error_msg)
                    failed_experiments.append((experiment_id, error_msg))
                
                pbar.update(1)
                pbar.set_postfix({
                    'variant': variant_name,
                    'run': run_id,
                    'completed': len(self.state.completed_experiments),
                    'failed': len(failed_experiments)
                })
        
        # 报告失败的实验
        if failed_experiments:
            logger.warning(f"\n{'='*50}")
            logger.warning(f"Failed experiments ({len(failed_experiments)}):")
            for exp_id, error in failed_experiments:
                logger.warning(f"  - {exp_id}: {error[:100]}...")
            logger.warning(f"{'='*50}\n")
        
        # 导出结果
        self.export_results_csv()
        self.save_experiment_config()
        
        logger.info(f"Ablation study completed: {len(results)} new results")
        return results
    
    def export_results_csv(self, output_path: Optional[Path] = None):
        """
        导出结果到 CSV
        
        Args:
            output_path: 输出路径（默认为 results/ablation_results.csv）
        """
        if output_path is None:
            output_path = self.output_dir / "results" / "ablation_results.csv"
        
        if not self.results:
            logger.warning("No results to export")
            return
        
        # 转换为 DataFrame
        data = [r.to_dict() for r in self.results]
        df = pd.DataFrame(data)
        
        # 排序
        df = df.sort_values(['config_name', 'run_id'])
        
        # 保存
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Results exported to {output_path}")
        
        # 同时导出统计检验结果
        self._export_statistical_tests()
    
    def _export_statistical_tests(self):
        """导出统计检验结果"""
        # 按变体分组
        results_by_variant = {}
        for r in self.results:
            if r.config_name not in results_by_variant:
                results_by_variant[r.config_name] = []
            results_by_variant[r.config_name].append({
                'MSE': r.mse,
                'RMSE': r.rmse,
                'MAE': r.mae,
                'R2': r.r2,
                'SpatialCorr': r.spatial_corr
            })
        
        # 计算显著性
        test_results = compute_all_significance_tests(results_by_variant)
        
        if test_results:
            df = pd.DataFrame(test_results)
            output_path = self.output_dir / "results" / "statistical_tests.csv"
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Statistical tests exported to {output_path}")
    
    def save_experiment_config(self, output_path: Optional[Path] = None):
        """
        保存实验配置到 YAML
        
        Args:
            output_path: 输出路径
        """
        if output_path is None:
            output_path = self.output_dir / "results" / "experiment_configs.yaml"
        
        config_data = {
            'experiment': {
                'name': self.config.get('experiment', {}).get('name', 'Ablation Study'),
                'timestamp': datetime.now().isoformat(),
                'runs_per_variant': self.runs_per_variant,
                'base_seed': self.base_seed,
                'device': self.device,
            },
            'training': self.config.get('training', {}),
            'model': self.config.get('model', {}),
            'variants': {
                name: {
                    'display_name': cfg.display_name,
                    'description': cfg.description,
                    **{k: v for k, v in cfg.to_dict().items() 
                       if k not in ['name', 'display_name', 'description']}
                }
                for name, cfg in ABLATION_VARIANTS.items()
            },
            'completed_experiments': list(self.state.completed_experiments),
            'failed_experiments': self.state.failed_experiments,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Config saved to {output_path}")
    
    def analyze_component(
        self,
        component_name: str,
        metric: str = 'RMSE'
    ) -> Dict[str, Any]:
        """
        分析单个组件的贡献
        
        Args:
            component_name: 组件名称 (convstem, attention, shpe, multiscale, gate)
            metric: 评估指标
            
        Returns:
            组件分析结果
        """
        from src.analysis.ablation.metrics import compute_component_contribution
        
        # 组件到变体的映射
        component_to_variant = {
            'convstem': 'wo_convstem',
            'attention': 'wo_attention',
            'shpe': 'wo_shpe',
            'multiscale': 'wo_multiscale',
            'gate': 'wo_gate',
        }
        
        variant_name = component_to_variant.get(component_name.lower())
        if not variant_name:
            raise ValueError(f"Unknown component: {component_name}")
        
        # 获取 baseline 和变体的结果
        baseline_results = [r for r in self.results if r.config_name == 'baseline']
        variant_results = [r for r in self.results if r.config_name == variant_name]
        
        if not baseline_results or not variant_results:
            return {'error': 'Missing results for comparison'}
        
        # 计算平均指标
        baseline_metrics = {
            'MSE': sum(r.mse for r in baseline_results) / len(baseline_results),
            'RMSE': sum(r.rmse for r in baseline_results) / len(baseline_results),
            'MAE': sum(r.mae for r in baseline_results) / len(baseline_results),
            'R2': sum(r.r2 for r in baseline_results) / len(baseline_results),
        }
        
        variant_metrics = {
            'MSE': sum(r.mse for r in variant_results) / len(variant_results),
            'RMSE': sum(r.rmse for r in variant_results) / len(variant_results),
            'MAE': sum(r.mae for r in variant_results) / len(variant_results),
            'R2': sum(r.r2 for r in variant_results) / len(variant_results),
        }
        
        contribution = compute_component_contribution(
            baseline_metrics, variant_metrics, metric
        )
        
        return {
            'component': component_name,
            'variant': variant_name,
            'metric': metric,
            **contribution
        }
    
    def generate_efficiency_report(self) -> pd.DataFrame:
        """
        生成效率分析报告
        
        Returns:
            效率分析 DataFrame
        """
        if not self.results:
            return pd.DataFrame()
        
        # 按变体聚合
        efficiency_data = []
        
        variants = set(r.config_name for r in self.results)
        for variant in variants:
            variant_results = [r for r in self.results if r.config_name == variant]
            
            efficiency_data.append({
                'variant': variant,
                'display_name': variant_results[0].display_name,
                'num_parameters': variant_results[0].num_parameters,
                'avg_train_time_s': sum(r.train_time_seconds for r in variant_results) / len(variant_results),
                'avg_inference_time_ms': sum(r.inference_time_ms for r in variant_results) / len(variant_results),
                'avg_peak_memory_mb': sum(r.peak_memory_mb for r in variant_results) / len(variant_results),
                'avg_rmse': sum(r.rmse for r in variant_results) / len(variant_results),
            })
        
        df = pd.DataFrame(efficiency_data)
        
        # 保存
        output_path = self.output_dir / "results" / "efficiency_report.csv"
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Efficiency report saved to {output_path}")
        
        return df


def run_sensitivity_analysis(
    runner: AblationRunner,
    param_name: str,
    values: List[Any],
    runs_per_value: int = 3
) -> pd.DataFrame:
    """
    运行超参数敏感性分析
    
    Args:
        runner: AblationRunner 实例
        param_name: 参数名称
        values: 参数值列表
        runs_per_value: 每个值的运行次数
        
    Returns:
        敏感性分析结果 DataFrame
    """
    results = []
    
    for value in tqdm(values, desc=f"Sensitivity: {param_name}"):
        # 创建配置
        config = AblationConfig(
            name=f"sensitivity_{param_name}_{value}",
            display_name=f"{param_name}={value}",
        )
        
        # 设置参数
        setattr(config, param_name, value)
        
        # 运行实验
        for run_id in range(1, runs_per_value + 1):
            try:
                result = runner.run_single_experiment(config, run_id)
                results.append({
                    'param': param_name,
                    'value': value,
                    'run_id': run_id,
                    'rmse': result.rmse,
                    'mae': result.mae,
                    'r2': result.r2,
                    'train_time': result.train_time_seconds,
                })
            except Exception as e:
                logger.error(f"Sensitivity run failed: {e}")
    
    df = pd.DataFrame(results)
    
    # 保存
    output_path = runner.output_dir / "results" / f"sensitivity_{param_name}.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    return df

