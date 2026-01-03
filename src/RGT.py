# RGTransformer 训练脚本
# 用法: python src/RGT.py

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
    # 训练
    # ============================================================
    
    print("=" * 70)
    print("[System Info]")
    print("=" * 70)
    print(f"  OS: {platform.system()}")
    print(f"  Workers: {trainer_params['num_workers']}")
    print(f"  Project: {PROJECT_PATH}")
    print("=" * 70)

    print("\n[Config]")
    print(f"  Area: {area.title}")
    print(f"  Resolution: {resolution} deg")
    print(f"  Spatial Size: {width} x {height}")
    print("=" * 70)

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

