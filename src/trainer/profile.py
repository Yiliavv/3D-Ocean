"""
三维温度场反演模型训练器

支持三种模型的统一训练和评估：
- Thermocline: 张春玲统计模型
- UNet3D: 深度学习模型
- RandomForest: 机器学习模型
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

from src.config.constants import deep
from src.config.params import MODEL_SAVE_PATH
from src.dataset.Argo import Argo3DTemperatureDataset, ArgoDepthMap


class ProfileTrainer:
    """
    三维温度场反演模型训练器
    
    参数：
        lon_range: 经度范围，如 [-180, 180]
        lat_range: 纬度范围，如 [-60, 60]
        depth_range: 深度层范围，如 [0, 30] 表示前30层
        resolution: 空间分辨率（度），默认 2
        train_ratio: 训练集比例，默认 0.8
    
    使用示例：
        trainer = ProfileTrainer(
            lon_range=[-180, 180],
            lat_range=[-60, 60],
            depth_range=[0, 30],
            resolution=2
        )
        
        # 训练所有模型
        results = trainer.train_all()
        
        # 绘制对比图
        trainer.plot_comparison()
    """
    
    def __init__(self,
                 lon_range=[-180, 180],
                 lat_range=[-60, 60],
                 depth_range=[0, 30],
                 resolution=2,
                 train_ratio=0.8):
        
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.depth_range = depth_range
        self.resolution = resolution
        self.train_ratio = train_ratio
        
        # 目标深度
        self.target_depths = deep[:depth_range[1]]
        self.n_depths = len(self.target_depths)
        
        # 数据
        self.sst_train = None
        self.profile_train = None
        self.sst_test = None
        self.profile_test = None
        self.data_loaded = False
        
        # 模型
        self.models = {}
        
        # 结果
        self.results = {}
    
    def load_data(self):
        """加载 Argo 数据"""
        if self.data_loaded:
            return
        
        print('=' * 70)
        print('📦 加载 Argo 数据集')
        print('=' * 70)
        print(f'   经度范围: {self.lon_range}')
        print(f'   纬度范围: {self.lat_range}')
        print(f'   深度范围: {ArgoDepthMap.get(self.depth_range[0])}m - {ArgoDepthMap.get(self.depth_range[1]-1)}m')
        print(f'   空间分辨率: {self.resolution}°')
        
        dataset = Argo3DTemperatureDataset(
            lon=self.lon_range,
            lat=self.lat_range,
            depth=self.depth_range,
            resolution=self.resolution
        )
        
        n_samples = len(dataset)
        n_train = int(n_samples * self.train_ratio)
        
        print(f'\n   总样本数: {n_samples} 个月')
        print(f'   训练样本: {n_train} 个月')
        print(f'   测试样本: {n_samples - n_train} 个月')
        
        # 加载训练数据
        sst_train_list, profile_train_list = [], []
        for i in tqdm(range(n_train), desc='加载训练数据'):
            sst, profile = dataset[i]
            sst_train_list.append(sst)
            profile_train_list.append(profile)
        
        self.sst_train = np.array(sst_train_list)
        self.profile_train = np.array(profile_train_list)
        
        # 加载测试数据
        sst_test_list, profile_test_list = [], []
        for i in tqdm(range(n_train, n_samples), desc='加载测试数据'):
            sst, profile = dataset[i]
            sst_test_list.append(sst)
            profile_test_list.append(profile)
        
        self.sst_test = np.array(sst_test_list)
        self.profile_test = np.array(profile_test_list)
        
        self.data_loaded = True
        print(f'\n   训练 SST 形状: {self.sst_train.shape}')
        print(f'   训练剖面形状: {self.profile_train.shape}')
    
    def _compute_metrics(self, pred, true):
        """计算评估指标"""
        valid_mask = ~(np.isnan(pred) | np.isnan(true))
        if valid_mask.sum() == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
        
        pred_valid = pred[valid_mask]
        true_valid = true[valid_mask]
        
        rmse = np.sqrt(np.mean((pred_valid - true_valid) ** 2))
        mae = np.mean(np.abs(pred_valid - true_valid))
        
        ss_res = np.sum((true_valid - pred_valid) ** 2)
        ss_tot = np.sum((true_valid - np.mean(true_valid)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def _compute_rmse_by_depth(self, predictions):
        """计算各深度层的 RMSE"""
        n_depths = predictions.shape[-1]
        rmse_list = []
        for d in range(n_depths):
            metrics = self._compute_metrics(
                predictions[:, :, :, d],
                self.profile_test[:, :, :, d]
            )
            rmse_list.append(metrics['rmse'])
        return rmse_list
    
    def train_thermocline(self):
        """训练 Thermocline 统计模型"""
        from src.models.Profile.Thermocline import Thermocline
        
        self.load_data()
        
        print('\n' + '=' * 70)
        print('🌡️ 训练 Thermocline 模型')
        print('=' * 70)
        
        model = Thermocline(
            target_depths=self.target_depths,
            mld_ref_depth=10,
            mld_threshold=0.5,
            use_gradient_dependent=True
        )
        
        fit_results = model.fit(self.sst_train, self.profile_train, compute_climatology=False)
        
        # 预测
        predictions = []
        for i in tqdm(range(len(self.sst_test)), desc='Thermocline 预测'):
            pred = model.predict(self.sst_test[i])
            predictions.append(pred[0])
        predictions = np.array(predictions)
        
        # 计算 RMSE
        rmse_by_depth = self._compute_rmse_by_depth(predictions)
        overall_rmse = np.nanmean(rmse_by_depth)
        
        self.models['thermocline'] = model
        self.results['thermocline'] = {
            'predictions': predictions,
            'rmse_by_depth': rmse_by_depth,
            'depths': self.target_depths,
            'overall_rmse': overall_rmse,
            'fit_results': fit_results
        }
        
        print(f'   ✅ Thermocline 总体 RMSE: {overall_rmse:.3f}°C')
        return model
    
    def train_unet3d(self, checkpoint_path=None, n_depth=None, 
                     epochs=100, batch_size=8, learning_rate=1e-4,
                     base_channels=64, force_retrain=False):
        """
        训练或加载 UNet3D 模型
        
        参数：
            checkpoint_path: checkpoint 路径，如果存在则加载
            n_depth: 输出深度层数，默认与 depth_range 一致
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            base_channels: 基础通道数
            force_retrain: 是否强制重新训练（忽略 checkpoint）
        """
        from src.models.Profile.UNet3D import UNet3D
        from lightning import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
        from torch.utils.data import DataLoader, TensorDataset
        import datetime
        
        self.load_data()
        
        # 默认深度层数与 depth_range 一致
        if n_depth is None:
            n_depth = self.n_depths
        
        print('\n' + '=' * 70)
        print('🧠 训练/加载 UNet3D 模型')
        print('=' * 70)
        print(f'   输出深度层数: {n_depth}')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'   使用设备: {device}')
        
        # 判断是否需要训练
        need_train = force_retrain or not (checkpoint_path and os.path.exists(checkpoint_path))
        
        if not need_train:
            print(f'   从 checkpoint 加载: {checkpoint_path}')
            model = UNet3D.load_from_checkpoint(
                checkpoint_path,
                n_channels=1,
                n_depth=n_depth,
                base_channels=base_channels
            )
        else:
            print(f'   开始训练 UNet3D...')
            print(f'   参数: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}')
            
            # 创建模型
            model = UNet3D(
                n_channels=1,
                n_depth=n_depth,
                base_channels=base_channels,
                learning_rate=learning_rate,
                max_epochs=epochs
            )
            
            # 准备数据
            # 训练数据: sst [N, H, W] -> [N, 1, H, W], profile [N, H, W, D]
            sst_train_tensor = torch.from_numpy(self.sst_train).float().unsqueeze(1)
            profile_train_tensor = torch.from_numpy(self.profile_train[:, :, :, :n_depth]).float()
            
            sst_val_tensor = torch.from_numpy(self.sst_test).float().unsqueeze(1)
            profile_val_tensor = torch.from_numpy(self.profile_test[:, :, :, :n_depth]).float()
            
            train_dataset = TensorDataset(sst_train_tensor, profile_train_tensor)
            val_dataset = TensorDataset(sst_val_tensor, profile_val_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            
            # 设置 checkpoint 保存路径
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
            ckpt_dir = f'out/checkpoints/{timestamp}'
            
            callbacks = [
                ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename='UNet3D',
                    save_top_k=1,
                    monitor='val_loss',
                    mode='min'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=50,  # 增大 patience，避免过早停止
                    mode='min',
                    verbose=True
                )
            ]
            
            # 训练
            trainer = Trainer(
                max_epochs=epochs,
                accelerator='auto',
                devices=1,
                callbacks=callbacks,
                enable_progress_bar=True,
                log_every_n_steps=10
            )
            
            trainer.fit(model, train_loader, val_loader)
            
            # 加载最佳模型
            best_ckpt = f'{ckpt_dir}/UNet3D.ckpt'
            if os.path.exists(best_ckpt):
                model = UNet3D.load_from_checkpoint(
                    best_ckpt,
                    n_channels=1,
                    n_depth=n_depth,
                    base_channels=base_channels
                )
                print(f'   ✅ 最佳模型保存至: {best_ckpt}')
        
        model = model.to(device)
        model.eval()
        
        # 预测
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(len(self.sst_test)), desc='UNet3D 预测'):
                sst_input = torch.from_numpy(self.sst_test[i]).float()
                sst_input = sst_input.unsqueeze(0).unsqueeze(0).to(device)
                sst_input = torch.nan_to_num(sst_input, nan=0.0)
                
                pred = model(sst_input)
                pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 恢复 NaN 掩码 (将陆地区域设为 NaN)
        # sst_test 的形状是 (N, H, W)，NaN 表示陆地
        mask = np.isnan(self.sst_test)
        
        # 将 mask 扩展到深度维度 (N, H, W, 1)
        mask_expanded = np.expand_dims(mask, axis=-1)
        
        # 将 mask 广播到 (N, H, W, D) 并应用
        predictions[np.broadcast_to(mask_expanded, predictions.shape)] = np.nan
        
        unet_n_depth = predictions.shape[-1]
        
        # 计算 RMSE
        rmse_by_depth = []
        for d in range(unet_n_depth):
            metrics = self._compute_metrics(
                predictions[:, :, :, d],
                self.profile_test[:, :, :, d]
            )
            rmse_by_depth.append(metrics['rmse'])
        
        overall_rmse = np.nanmean(rmse_by_depth)
        
        self.models['unet3d'] = model
        self.results['unet3d'] = {
            'predictions': predictions,
            'rmse_by_depth': rmse_by_depth,
            'depths': self.target_depths[:unet_n_depth],
            'overall_rmse': overall_rmse,
            'n_depth': unet_n_depth
        }
        
        print(f'   ✅ UNet3D 总体 RMSE ({unet_n_depth}层, 0-{self.target_depths[unet_n_depth-1]}m): {overall_rmse:.3f}°C')
        return model
    
    def train_random_forest(self, n_estimators=50, max_depth=20):
        """训练 Random Forest 模型"""
        self.load_data()
        
        print('\n' + '=' * 70)
        print('🌲 训练 Random Forest 模型')
        print('=' * 70)
        
        # 准备训练数据
        sst_flat = self.sst_train.reshape(-1)
        profile_flat = self.profile_train.reshape(-1, self.n_depths)
        
        valid_mask = ~(np.isnan(sst_flat) | np.any(np.isnan(profile_flat), axis=1))
        sst_valid = sst_flat[valid_mask].reshape(-1, 1)
        profile_valid = profile_flat[valid_mask]
        
        print(f'   有效训练样本: {len(sst_valid)}')
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        print(f'   训练中 (n_estimators={n_estimators}, max_depth={max_depth})...')
        model.fit(sst_valid, profile_valid)
        
        # 预测
        sst_test_flat = self.sst_test.reshape(-1)
        valid_test_mask = ~np.isnan(sst_test_flat)
        
        pred_flat = np.full((len(sst_test_flat), self.n_depths), np.nan)
        pred_flat[valid_test_mask] = model.predict(
            sst_test_flat[valid_test_mask].reshape(-1, 1)
        )
        
        predictions = pred_flat.reshape(
            self.sst_test.shape[0],
            self.sst_test.shape[1],
            self.sst_test.shape[2],
            self.n_depths
        )
        
        # 计算 RMSE
        rmse_by_depth = self._compute_rmse_by_depth(predictions)
        overall_rmse = np.nanmean(rmse_by_depth)
        
        self.models['rf'] = model
        self.results['rf'] = {
            'predictions': predictions,
            'rmse_by_depth': rmse_by_depth,
            'depths': self.target_depths,
            'overall_rmse': overall_rmse
        }
        
        print(f'   ✅ Random Forest 总体 RMSE: {overall_rmse:.3f}°C')
        return model
    
    def train_all(self, unet_checkpoint=None):
        """训练所有模型"""
        self.train_thermocline()
        self.train_random_forest()
        if unet_checkpoint:
            self.train_unet3d(checkpoint_path=unet_checkpoint)
        return self.results
    
    def plot_comparison(self, save_path=None):
        """
        绘制三模型 RMSE 对比图（单图多元素）
        """
        if not self.results:
            print('⚠️ 请先训练模型')
            return
        
        # 设置字体
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 12
        
        # 定义模型样式
        styles = {
            'thermocline': {
                'color': '#1E88E5',
                'marker': 'o',
                'label': 'Thermocline',
                'fill_alpha': 0.15
            },
            'unet3d': {
                'color': '#E53935',
                'marker': 's',
                'label': 'UNet3D',
                'fill_alpha': 0.15
            },
            'rf': {
                'color': '#43A047',
                'marker': '^',
                'label': 'Random Forest',
                'fill_alpha': 0.15
            }
        }
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        
        # 绘制各模型曲线 + 填充区域
        for name, result in self.results.items():
            if name in styles:
                style = styles[name]
                depths = np.array(result['depths'])
                rmse = np.array(result['rmse_by_depth'])
                
                # 填充区域（从0到曲线）
                ax.fill_betweenx(depths, 0, rmse, 
                                alpha=style['fill_alpha'], 
                                color=style['color'])
                
                # 主曲线
                ax.plot(rmse, depths,
                       color=style['color'],
                       marker=style['marker'],
                       markersize=8,
                       linewidth=2.5,
                       markerfacecolor='white',
                       markeredgewidth=2,
                       label=style['label'])
        
        # 添加深度分层背景
        max_rmse = max([max(r['rmse_by_depth']) for r in self.results.values()])
        ax.axhspan(0, 100, alpha=0.05, color='#2196F3', zorder=0)
        ax.axhspan(100, 200, alpha=0.05, color='#FF9800', zorder=0)
        ax.axhspan(200, 400, alpha=0.05, color='#9C27B0', zorder=0)
        
        # 深度层标注（右侧）
        ax.text(max_rmse * 1.02, 50, 'Mixed Layer\n(0-100m)', 
               fontsize=9, va='center', color='#1565C0', style='italic')
        ax.text(max_rmse * 1.02, 150, 'Thermocline\n(100-200m)', 
               fontsize=9, va='center', color='#E65100', style='italic')
        ax.text(max_rmse * 1.02, 300, 'Deep Layer\n(>200m)', 
               fontsize=9, va='center', color='#6A1B9A', style='italic')
        
        # 坐标轴设置
        ax.set_xlabel('RMSE (°C)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(0, max_rmse * 1.25)
        
        # 网格
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # 图例
        legend = ax.legend(loc='lower right', fontsize=11, 
                          framealpha=0.95, fancybox=True)
        legend.get_frame().set_edgecolor('#CCCCCC')
        
        
        # 在每条曲线旁标注平均 RMSE 值（放在下方 3/4 位置，避免遮挡）
        for name, result in self.results.items():
            if name in styles:
                style = styles[name]
                depths = np.array(result['depths'])
                rmse = np.array(result['rmse_by_depth'])
                # 在曲线 3/4 位置标注（更深的地方）
                label_idx = int(len(depths) * 0.75)
                ax.annotate(f'{result["overall_rmse"]:.2f}°C',
                           xy=(rmse[label_idx], depths[label_idx]),
                           xytext=(15, 0), textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           color=style['color'],
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', alpha=0.8,
                                   edgecolor=style['color']))
        
        # 移除上右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'{MODEL_SAVE_PATH}/three_models_rmse_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f'✅ 保存: {save_path}')
        return fig
    
    def print_summary(self):
        """打印结果摘要"""
        if not self.results:
            print('⚠️ 请先训练模型')
            return
        
        print('\n' + '=' * 70)
        print('📊 三模型 RMSE 对比')
        print('=' * 70)
        
        # 表头
        headers = ['深度(m)']
        for name in ['thermocline', 'unet3d', 'rf']:
            if name in self.results:
                headers.append(name.upper())
        
        print(f"\n{headers[0]:<10}", end='')
        for h in headers[1:]:
            print(f"{h:<15}", end='')
        print()
        print('-' * (10 + 15 * (len(headers) - 1)))
        
        # 数据行
        for i, depth in enumerate(self.target_depths):
            print(f"{depth:<10.0f}", end='')
            for name in ['thermocline', 'unet3d', 'rf']:
                if name in self.results:
                    rmse_list = self.results[name]['rmse_by_depth']
                    if i < len(rmse_list):
                        print(f"{rmse_list[i]:<15.3f}", end='')
                    else:
                        print(f"{'N/A':<15}", end='')
            print()
        
        # 总体
        print('-' * (10 + 15 * (len(headers) - 1)))
        print(f"{'总体':<10}", end='')
        for name in ['thermocline', 'unet3d', 'rf']:
            if name in self.results:
                print(f"{self.results[name]['overall_rmse']:<15.3f}", end='')
        print()
    
    def plot_3d_predictions(self, sample_idx=0, save_path=None):
        """
        绘制三个模型的三维温度预测图（在一个画布上）
        
        参数：
            sample_idx: 测试样本索引
            save_path: 保存路径（目录）
        """
        if not self.results:
            print('⚠️ 请先训练模型')
            return
        
        from src.plot.profile import plot_3d_temperature
        from src.plot.base import create_3d_axes
        
        # 准备参数
        depths = list(self.target_depths)
        lon = self.lon_range
        lat = self.lat_range
        step = self.resolution
        
        if save_path is None:
            save_path = MODEL_SAVE_PATH
        
        # 创建 2x2 子图
        axes = create_3d_axes(row=2, col=2)
        # axes可能是(2,2)的数组，展平以便索引
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            # 如果只有一个子图（虽然这里是2x2，但为了健壮性）
            axes = [axes]
        
        # 1. 绘制真值 (Argo)
        print('🎨 绘制 Argo')
        temp_true = self.profile_test[sample_idx]
        # 数据形状: (lat, lon, depth) -> 需要转置为 (lon, lat, depth)
        temp_true_transposed = np.transpose(temp_true, (1, 0, 2))
        plot_3d_temperature(temp_true_transposed, lon, lat, depths, step=step,
                           label='Temperature (°C)', ax=axes[0], colorbar=False)
        axes[0].set_title('Argo (Ground Truth)', fontsize=6, fontweight='bold', pad=10)
        
        # 2. 绘制各模型预测
        model_info = {
            'thermocline': 'Thermocline',
            'unet3d': 'UNet3D',
            'rf': 'Random Forest'
        }
        
        # 定义绘制顺序
        model_names = ['thermocline', 'unet3d', 'rf']
        
        # 依次绘制模型预测
        plot_idx = 1
        for name in model_names:
            if name in self.results and plot_idx < 4:
                print(f'🎨 绘制 {model_info[name]} 预测...')
                pred = self.results[name]['predictions'][sample_idx]
                pred_transposed = np.transpose(pred, (1, 0, 2))
                
                rmse = self.results[name]['overall_rmse']
                
                plot_3d_temperature(pred_transposed, lon, lat, 
                                   list(self.results[name]['depths']), 
                                   step=step, label='Temperature (°C)', ax=axes[plot_idx], colorbar=False)
                axes[plot_idx].set_title(f'{model_info[name]} (RMSE: {rmse:.3f}°C)', 
                         fontsize=6, fontweight='bold', pad=10)
                plot_idx += 1
        
        # 隐藏多余的子图
        while plot_idx < 4:
            axes[plot_idx].axis('off')
            plot_idx += 1
            
        # 添加公共色标
        import matplotlib
        
        # 调整布局，为底部色标留出空间
        # 增加子图间距(wspace, hspace)和底部边距(bottom)防止重叠
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15, wspace=0.4, hspace=0.6)
        
        # 创建一个对应的 ScalarMappable
        # 注意：这里需要与 plot_3d_temperature 中的设置保持一致 (vmin=0, vmax=30, cmap='jet')
        cmap = plt.get_cmap('jet')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=30)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        
        # 在底部添加色标 [left, bottom, width, height]
        # 使用 figure坐标系，位置稍微下移，避开子图坐标轴
        # 居中且变短：width=0.4, left=(1-0.4)/2=0.3
        cbar_ax = plt.gcf().add_axes([0.3, 0.03, 0.4, 0.02]) 
        cb = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Temperature (°C)')
        cb.ax.tick_params(labelsize=6)
        cb.set_label('Temperature (°C)', fontsize=7)
        
        output_file = f'{save_path}/three_models_3d_prediction.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f'\n✅ 所有三维温度图已保存至 {output_file}')

