# RGTransformer 训练脚本（V3.2 - 小样本优化版）
# 用法: python src/RGT.py
#
# 针对月平均 SST 数据（~530样本）的优化策略：
# - 适中的模型容量（避免过拟合）
# - 强正则化（dropout=0.2, weight_decay=0.05）
# - 早停机制

if __name__ == '__main__':
    import sys
    import platform
    
    sys.path.append('X:/Workspace/3D-Ocean')
    
    from src.trainer.base import BaseTrainer
    from src.trainer.config import (
        area, resolution, width, height,
        dataset_params, model_params, trainer_params,
    )
    from src.models.SST.RGTransformer import RGTransformer
    from src.config.params import PROJECT_PATH
    from src.dataset.OISST import OISSTMonthlyDataset

    # ============================================================
    # 打印配置信息
    # ============================================================
    
    print("=" * 70)
    print("[RGTransformer V3.2 - 小样本优化版]")
    print("=" * 70)
    print(f"  OS: {platform.system()}")
    print(f"  Project: {PROJECT_PATH}")
    print("=" * 70)

    print("\n[Model Config]")
    print(f"  d_model: {model_params.get('d_model', 'N/A')}")
    print(f"  num_heads: {model_params.get('num_heads', 'N/A')}")
    print(f"  num_attn_layers: {model_params.get('num_attn_layers', 'N/A')}")
    print(f"  dim_feedforward: {model_params.get('dim_feedforward', 'N/A')}")
    print(f"  ffn_activation: {model_params.get('ffn_activation', 'gelu')}")
    print(f"  dropout: {model_params.get('dropout', 0.1)}")
    print(f"  weight_decay: {model_params.get('weight_decay', 0.01)}")
    print("=" * 70)

    print("\n[Dataset]")
    print(f"  Dataset: OISSTMonthlyDataset (~530 samples)")
    print(f"  Area: {area.title}")
    print(f"  Resolution: {resolution} deg")
    print(f"  Spatial Size: {width} x {height}")
    print("=" * 70)

    print("\n[Training Strategy]")
    print(f"  Epochs: {trainer_params.get('epochs', 'N/A')}")
    print(f"  Batch Size: {trainer_params.get('batch_size', 'N/A')}")
    print(f"  Early Stopping: {trainer_params.get('early_stopping_patience', 'N/A')} epochs")
    print("=" * 70)

    # ============================================================
    # 训练
    # ============================================================

    trainer = BaseTrainer(
        area=area,
        model_class=RGTransformer,
        dataset_class=OISSTMonthlyDataset,
        use_checkpoint=trainer_params.get('use_checkpoint', True),
        dataset_params=dataset_params,
        trainer_params=trainer_params,
        model_params=model_params,
    )

    model = trainer.train()
    
    print("\n[OK] Training completed!")

