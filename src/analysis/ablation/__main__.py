"""
消融实验命令行入口

用法:
    python -m src.analysis.ablation [OPTIONS]

示例:
    # 运行所有消融变体
    python -m src.analysis.ablation
    
    # 只运行特定变体
    python -m src.analysis.ablation --variants baseline wo_convstem wo_shpe
    
    # 从断点恢复
    python -m src.analysis.ablation --resume
    
    # 强制重新运行
    python -m src.analysis.ablation --force
    
    # 指定配置文件
    python -m src.analysis.ablation --config configs/ablation_config.yaml
    
    # 分析单个组件
    python -m src.analysis.ablation --component convstem
    
    # 运行超参数敏感性分析
    python -m src.analysis.ablation --sensitivity d_model
    
    # 运行效率分析
    python -m src.analysis.ablation --efficiency
"""

import argparse
import sys
import logging
import pandas as pd
from pathlib import Path

from src.analysis.ablation.runner import AblationRunner, run_sensitivity_analysis
from src.analysis.ablation.config import ABLATION_VARIANTS
from src.plot.ablation import AblationVisualizer, TableGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='RGTransformer 消融实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 基础参数
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='配置文件路径 (YAML)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='out/ablation',
        help='输出目录 (默认: out/ablation)'
    )
    
    parser.add_argument(
        '--variants', '-v',
        nargs='+',
        default=None,
        choices=list(ABLATION_VARIANTS.keys()),
        help='要运行的变体列表'
    )
    
    parser.add_argument(
        '--runs', '-r',
        type=int,
        default=1,
        help='每变体运行次数 (默认: 1)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='设备 (默认: auto)'
    )
    
    # 控制参数
    parser.add_argument(
        '--resume',
        action='store_true',
        help='从断点恢复实验'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新运行所有实验'
    )
    
    # 分析模式
    parser.add_argument(
        '--component',
        type=str,
        default=None,
        choices=['convstem', 'attention', 'shpe', 'multiscale', 'gate'],
        help='分析单个组件的贡献'
    )
    
    parser.add_argument(
        '--sensitivity',
        type=str,
        default=None,
        choices=['d_model', 'num_heads', 'num_attn_layers', 'patch_size'],
        help='运行超参数敏感性分析'
    )
    
    parser.add_argument(
        '--efficiency',
        action='store_true',
        help='生成效率分析报告'
    )
    
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='仅生成图表和表格（不运行实验）'
    )
    
    # 输出控制
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='减少输出'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("RGTransformer 消融实验")
    logger.info("=" * 60)
    
    # 创建运行器
    runner = AblationRunner(
        output_dir=args.output,
        config_path=args.config,
        runs_per_variant=args.runs,
        seed=args.seed,
        device=args.device
    )
    
    # 组件分析模式
    if args.component:
        logger.info(f"分析组件: {args.component}")
        result = runner.analyze_component(args.component)
        print("\n" + "=" * 40)
        print(f"组件分析: {args.component}")
        print("=" * 40)
        for key, value in result.items():
            print(f"  {key}: {value}")
        return 0
    
    # 敏感性分析模式
    if args.sensitivity:
        logger.info(f"敏感性分析: {args.sensitivity}")
        
        # 默认参数范围
        param_ranges = {
            'd_model': [128, 256, 512],
            'num_heads': [4, 8, 16],
            'num_attn_layers': [1, 2, 4],
            'patch_size': [2, 4, 8],
        }
        
        values = param_ranges.get(args.sensitivity, [])
        df = run_sensitivity_analysis(runner, args.sensitivity, values, args.runs)
        
        print("\n" + "=" * 40)
        print(f"敏感性分析: {args.sensitivity}")
        print("=" * 40)
        print(df.groupby('value').agg({
            'rmse': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'train_time': 'mean'
        }))
        return 0
    
    # 效率分析模式
    if args.efficiency:
        logger.info("生成效率分析报告")
        df = runner.generate_efficiency_report()
        
        print("\n" + "=" * 40)
        print("效率分析报告")
        print("=" * 40)
        print(df.to_string(index=False))
        return 0
    
    # 仅生成图表和表格模式
    if args.generate_only:
        logger.info("仅生成图表和表格")
        generate_figures_and_tables(args.output)
        return 0
    
    # 默认: 运行消融实验
    logger.info("运行消融实验")
    
    resume = args.resume and not args.force
    results = runner.run_all_variants(
        variants=args.variants,
        resume=resume
    )
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("消融实验完成")
    print("=" * 60)
    print(f"完成实验数: {len(results)}")
    print(f"输出目录: {args.output}")
    
    # 打印每个变体的平均结果
    if results:
        print("\n变体性能摘要:")
        print("-" * 50)
        
        variants = set(r.config_name for r in results)
        for variant in sorted(variants):
            variant_results = [r for r in results if r.config_name == variant]
            avg_rmse = sum(r.rmse for r in variant_results) / len(variant_results)
            avg_mae = sum(r.mae for r in variant_results) / len(variant_results)
            avg_r2 = sum(r.r2 for r in variant_results) / len(variant_results)
            
            display_name = variant_results[0].display_name
            print(f"  {display_name:20s} RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, R2: {avg_r2:.4f}")
        
        # 生成图表和表格
        generate_figures_and_tables(args.output)
    
    return 0


def generate_figures_and_tables(output_dir: str):
    """
    生成消融实验的图表和表格
    
    Args:
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    results_csv = output_path / 'results' / 'ablation_results.csv'
    
    if not results_csv.exists():
        logger.warning(f"结果文件不存在: {results_csv}")
        return
    
    # 加载结果
    results_df = pd.read_csv(results_csv)
    
    if results_df.empty:
        logger.warning("结果文件为空，跳过图表生成")
        return
    
    print("\n" + "=" * 60)
    print("生成图表和表格")
    print("=" * 60)
    
    # 生成图表
    figures_dir = output_path / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        visualizer = AblationVisualizer(output_dir=str(figures_dir))
        figures = visualizer.generate_all_figures(results_df)
        print(f"[OK] 生成 {len(figures)} 个图表到: {figures_dir}")
    except Exception as e:
        logger.error(f"生成图表时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成表格
    tables_dir = output_path / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        table_gen = TableGenerator(output_dir=str(tables_dir))
        
        # 主结果表格
        latex_main = table_gen.generate_main_results_table(
            results_df,
            caption='RGTransformer Ablation Study Results',
            label='tab:ablation_main'
        )
        print(f"[OK] 生成主结果表格: {tables_dir / 'main_results.tex'}")
        
        # 效率表格
        if 'num_parameters' in results_df.columns or 'train_time_seconds' in results_df.columns:
            latex_eff = table_gen.generate_efficiency_table(
                results_df,
                caption='Model Efficiency Comparison',
                label='tab:ablation_efficiency',
                output_name='efficiency.tex'
            )
            print(f"[OK] 生成效率表格: {tables_dir / 'efficiency.tex'}")
        
    except Exception as e:
        logger.error(f"生成表格时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n输出位置:")
    print(f"  - 图表: {figures_dir}")
    print(f"  - 表格: {tables_dir}")


if __name__ == '__main__':
    sys.exit(main())

