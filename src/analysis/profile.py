"""
UNet3D Model Analysis Tool - Nature Quality
Generate publication-ready figures for academic journals
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from matplotlib import rcParams

# 设置Nature级别的图片参数
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 16
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 16
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1
rcParams['axes.unicode_minus'] = False  # 正确显示负号
rcParams['axes.labelpad'] = 12
rcParams['xtick.major.pad'] = 8
rcParams['ytick.major.pad'] = 8


def load_model_safely(model_path: Union[str, Path], 
                     model_class: Optional[type] = None,
                     **model_kwargs) -> nn.Module:
    """
    Safely load PyTorch model (compatible with PyTorch 2.6+)
    
    Args:
        model_path: Path to model file (.pkl or .ckpt)
        model_class: Model class (required for checkpoint loading)
        **model_kwargs: Model initialization parameters
        
    Returns:
        Loaded model in eval mode
    """
    import sys
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if model_path.suffix == '.ckpt':
        if model_class is None:
            raise ValueError("model_class required for checkpoint loading")
        print(f"📦 Loading from Lightning checkpoint: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'hyper_parameters' in checkpoint and not model_kwargs:
            print("   Using saved hyperparameters")
            model = model_class.load_from_checkpoint(str(model_path))
        else:
            if not model_kwargs:
                print("   Inferring parameters from state_dict...")
                state_dict = checkpoint['state_dict']
                
                first_conv_weight = state_dict.get('inc.double_conv.0.weight')
                if first_conv_weight is not None:
                    base_channels = first_conv_weight.shape[0]
                    print(f"   Inferred: base_channels={base_channels}")
                    model_kwargs = {'base_channels': base_channels}
                else:
                    print("   ⚠️  Cannot infer parameters, using defaults")
                    model_kwargs = {}
            
            print(f"   Creating model manually: {model_kwargs}")
            model = model_class(**model_kwargs)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("   ✅ Weights loaded")
    else:
        print(f"📦 Loading model file: {model_path}")
        
        try:
            model = torch.load(model_path, weights_only=False)
        except TypeError:
            model = torch.load(model_path)
    
    model.eval()
    print(f"✅ Model loaded successfully!")
    return model


class UNet3DNatureAnalyzer:
    """UNet3D Model Analyzer for Nature-quality figures"""
    
    def __init__(self, model: nn.Module, output_dir: str = "out/models/nature"):
        """
        Initialize analyzer
        
        Args:
            model: UNet3D model instance
            output_dir: Output directory for figures
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def analyze_all(self, 
                   sample_data: Optional[np.ndarray] = None,
                   verbose: bool = True,
                   groups: Optional[List[str]] = None) -> Dict:
        """
        Perform complete model analysis with Nature-quality figures
        
        Args:
            sample_data: Sample SST data [H, W]
            verbose: Print detailed information
            groups: List of analysis groups to generate
            
        Returns:
            Dictionary of analysis results
        """
        if groups is None:
            groups = ['fig1_architecture', 'fig2_feature_maps', 'fig3_skip_connections',
                     'fig4_nan_handling', 'fig5_vertical_reconstruction', 'fig6_temperature_profiles']
        
        print("=" * 100)
        print("🔬 UNet3D Model Analysis - Nature Quality Figures")
        print("=" * 100)
        
        if sample_data is None:
            sample_data = np.random.randn(80, 180).astype(np.float32)
            sample_data[sample_data < -2] = np.nan
        
        # Figure 1: Model Architecture
        if 'fig1_architecture' in groups:
            if verbose:
                print("\n📊 Figure 1: Model Architecture and Parameters...")
            self.results['fig1'] = self.generate_fig1_architecture(verbose=verbose)
        
        # Figure 2: Feature Maps
        if 'fig2_feature_maps' in groups:
            if verbose:
                print("\n🎨 Figure 2: Feature Maps Visualization...")
            self.results['fig2'] = self.generate_fig2_feature_maps(sample_data, verbose=verbose)
        
        # Figure 3: Skip Connections
        if 'fig3_skip_connections' in groups:
            if verbose:
                print("\n🔗 Figure 3: Skip Connection Analysis...")
            self.results['fig3'] = self.generate_fig3_skip_connections(sample_data, verbose=verbose)
        
        # Figure 4: NaN Handling
        if 'fig4_nan_handling' in groups:
            if verbose:
                print("\n🌊 Figure 4: NaN Handling Mechanism...")
            self.results['fig4'] = self.generate_fig4_nan_handling(sample_data, verbose=verbose)
        
        # Figure 5: Vertical Reconstruction
        if 'fig5_vertical_reconstruction' in groups:
            if verbose:
                print("\n📐 Figure 5: Vertical Temperature Reconstruction...")
            self.results['fig5'] = self.generate_fig5_vertical_reconstruction(sample_data, verbose=verbose)
        
        # Figure 6: Temperature Profiles
        if 'fig6_temperature_profiles' in groups:
            if verbose:
                print("\n📈 Figure 6: Temperature Profiles at Multiple Locations...")
            # 使用 Figure 5 的输出，或重新计算
            if 'fig5' in self.results and 'output_3d' in self.results['fig5']:
                output_3d = self.results['fig5']['output_3d']
                depths = self.results['fig5']['depths']
            else:
                # 重新计算
                sample_input = torch.from_numpy(sample_data).unsqueeze(0).unsqueeze(0).float()
                self.model.eval()
                with torch.no_grad():
                    output = self.model(sample_input)
                output_3d = output[0].cpu().numpy()
                nan_mask = np.isnan(sample_data)
                for d in range(output_3d.shape[0]):
                    output_3d[d][nan_mask] = np.nan
                depths = np.array([0, 10, 50, 100, 200, 400, 600, 800, 1000, 1500])[:output_3d.shape[0]]
            
            self.results['fig6'] = self.generate_fig6_temperature_profiles(output_3d, depths, verbose=verbose)
        
        print("\n" + "=" * 100)
        print("✅ Analysis Complete! Figures saved to:", self.output_dir)
        print("=" * 100)
        
        return self.results
    
    def generate_fig1_architecture(self, verbose: bool = True) -> Dict:
        """
        Figure 1: Model Architecture Diagram
        Shows U-Net structure, parameters, and receptive field
        """
        fig = plt.figure(figsize=(10, 8))
        
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        
        # (a) Architecture diagram
        ax1 = fig.add_subplot(gs[0, :])
        ax1.text(0.5, 0.9, '(a) U-Net Architecture', transform=ax1.transAxes,
                fontsize=16, fontweight='bold', ha='center')
        
        # 简化的架构图
        architecture_text = """
        Input (SST)           Encoder              Bottleneck            Decoder              Output (Profile)
        [1, H, W]    →    [64→128→256→512]  →   [1024]   →   [512→256→128→64]  →   [10, H, W]
                          ↓                                              ↑
                          Skip Connections (Concat)
        """
        ax1.text(0.5, 0.5, architecture_text, transform=ax1.transAxes,
                fontfamily='monospace', fontsize=16, ha='center', va='center')
        ax1.axis('off')
        
        # (b) Parameters statistics
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.text(0.05, 0.95, '(b) Parameter Distribution', transform=ax2.transAxes,
                fontsize=16, fontweight='bold')
        
        # 计算参数
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        encoder_params = sum(count_params(getattr(self.model, f'down{i}')) 
                           for i in range(1, 5)) + count_params(self.model.inc)
        decoder_params = sum(count_params(getattr(self.model, f'up{i}')) 
                           for i in range(1, 5)) + count_params(self.model.outc)
        
        labels = ['Encoder', 'Decoder']
        sizes = [encoder_params, decoder_params]
        colors = ['#FF6B6B', '#4ECDC4']
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
                                            autopct='%1.1f%%', startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # (c) Receptive field
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.text(0.05, 0.95, '(c) Receptive Field', transform=ax3.transAxes,
                fontsize=16, fontweight='bold')
        
        layers = ['Input', 'Inc', 'Down1', 'Down2', 'Down3', 'Down4\n(Bottleneck)']
        rf_sizes = [1, 5, 15, 35, 75, 155]
        
        ax3.plot(range(len(layers)), rf_sizes, 'o-', linewidth=2, 
                markersize=8, color='#2E86AB')
        ax3.set_xticks(range(len(layers)))
        ax3.set_xticklabels(layers, rotation=45, ha='right', fontsize=14)
        ax3.set_ylabel('Receptive Field (pixels)', fontsize=14, labelpad=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-0.5, len(layers)-0.5)
        
        plt.tight_layout()
        save_path = self.output_dir / 'Figure1_Architecture.png'
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        if verbose:
            print(f"✅ Figure 1 saved: {save_path}")
        plt.close()
        
        return {'save_path': str(save_path), 'encoder_params': encoder_params, 
                'decoder_params': decoder_params}
    
    def generate_fig2_feature_maps(self, sample_data: np.ndarray, verbose: bool = True) -> Dict:
        """
        Figure 2: Feature Maps Through Encoder-Decoder
        """
        # 提取特征
        device = next(self.model.parameters()).device
        sample_input = torch.from_numpy(sample_data).unsqueeze(0).unsqueeze(0).float().to(device)
        
        feature_maps = {}
        def get_activation(name):
            def hook(model, input, output):
                feature_maps[name] = output.detach()
            return hook
        
        hooks = []
        for name in ['inc', 'down1', 'down2', 'down3', 'down4', 'up1', 'up2', 'up3', 'up4']:
            module = getattr(self.model, name)
            hook = module.register_forward_hook(get_activation(name))
            hooks.append(hook)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_input)
        
        for hook in hooks:
            hook.remove()
        
        def _channel_label(tensor):
            return f'[{tensor.shape[1]}ch]'

        layer_names = [
            '(a) Input SST',
            f'(b) Inc {_channel_label(feature_maps["inc"])}',
            f'(c) Down1 {_channel_label(feature_maps["down1"])}',
            f'(d) Down2 {_channel_label(feature_maps["down2"])}',
            f'(e) Down3 {_channel_label(feature_maps["down3"])}',
            f'(f) Down4 {_channel_label(feature_maps["down4"])}',
            f'(g) Up1 {_channel_label(feature_maps["up1"])}',
            f'(h) Up2 {_channel_label(feature_maps["up2"])}',
            f'(i) Up3 {_channel_label(feature_maps["up3"])}',
            f'(j) Up4 {_channel_label(feature_maps["up4"])}',
            f'(k) Output [{output.shape[1]}ch]',
        ]
        layer_keys = [None, 'inc', 'down1', 'down2', 'down3', 'down4',
                      'up1', 'up2', 'up3', 'up4', None]

        # 绘制：两列排版，便于论文版面中保持单个面板可读性。
        n_cols = 2
        n_rows = int(np.ceil(len(layer_keys) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(12, 18),
            subplot_kw={'projection': ccrs.PlateCarree()},
        )
        
        axes_flat = axes.ravel()
        for idx, (layer_name, layer_key) in enumerate(zip(layer_names, layer_keys)):
            ax = axes_flat[idx]
            
            if layer_key is None:
                if idx == 0:
                    data = sample_input[0, 0].cpu().numpy()
                else:
                    data = output[0, 0].cpu().numpy()
            else:
                data = feature_maps[layer_key][0, 0].cpu().numpy()
            
            cmap = 'RdYlBu_r' if idx <= 5 else 'viridis'
            im = ax.imshow(data, cmap=cmap, origin='lower',
                          extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree())
            
            ax.coastlines(linewidth=0.3)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
            ax.set_title(layer_name, fontsize=16, fontweight='bold', pad=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.02, 
                        fraction=0.05, aspect=30).ax.tick_params(labelsize=12, pad=4)

        for ax in axes_flat[len(layer_names):]:
            ax.axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / 'Figure2_FeatureMaps.png'
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        if verbose:
            print(f"✅ Figure 2 saved: {save_path}")
        plt.close()
        
        return {'save_path': str(save_path), 'feature_maps': feature_maps}
    
    def generate_fig3_skip_connections(self, sample_data: np.ndarray, verbose: bool = True) -> Dict:
        """
        Figure 3: Impact of Skip Connections
        通过用零张量替代跳跃连接来展示其重要性（使用同一个训练好的模型）
        """
        
        def forward_no_skip(model, x):
            """无跳跃连接的前向传播 - 用零张量替代跳跃连接"""
            x = torch.nan_to_num(x, nan=0.0)
            
            # 编码器 - 正常执行
            x1 = model.inc(x)
            x2 = model.down1(x1)
            x3 = model.down2(x2)
            x4 = model.down3(x3)
            x5 = model.down4(x4)
            
            # 解码器 - 用零张量替代跳跃连接特征
            # 这样保持模型权重不变，只是不传递高分辨率信息
            x = model.up1(x5, torch.zeros_like(x4))
            x = model.up2(x, torch.zeros_like(x3))
            x = model.up3(x, torch.zeros_like(x2))
            x = model.up4(x, torch.zeros_like(x1))
            
            return model.outc(x)
        
        device = next(self.model.parameters()).device
        sample_input = torch.from_numpy(sample_data).unsqueeze(0).unsqueeze(0).float().to(device)
        
        self.model.eval()
        
        with torch.no_grad():
            output_with_skip = self.model(sample_input)
            output_no_skip = forward_no_skip(self.model, sample_input)
        
        # 绘制对比：每行一个深度，左列保留跳跃连接，右列移除跳跃连接。
        fig, axes = plt.subplots(
            5,
            2,
            figsize=(10, 14),
            subplot_kw={'projection': ccrs.PlateCarree()},
        )
        
        depth_indices = [0, 2, 4, 6, 8]
        depth_labels = ['Surface', '50m', '150m', '400m', '900m']
        
        for i, (depth_idx, depth_name) in enumerate(zip(depth_indices, depth_labels)):
            # With skip connections
            ax1 = axes[i, 0]
            data_skip = output_with_skip[0, depth_idx].cpu().numpy()
            im1 = ax1.imshow(data_skip, cmap='RdYlBu_r', origin='lower',
                            extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                            vmin=-2, vmax=30)
            ax1.coastlines(linewidth=0.3)
            ax1.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)

            ax1.set_ylabel(depth_name, fontsize=14, fontweight='bold', labelpad=12)
            ax1.set_title('With Skip Connections' if i == 0 else '', fontsize=16, fontweight='bold', pad=10)
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Without skip connections
            ax2 = axes[i, 1]
            data_no_skip = output_no_skip[0, depth_idx].cpu().numpy()
            im2 = ax2.imshow(data_no_skip, cmap='RdYlBu_r', origin='lower',
                            extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                            vmin=-2, vmax=30)
            ax2.coastlines(linewidth=0.3)
            ax2.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)

            ax2.set_title('Without Skip Connections' if i == 0 else '', fontsize=16, fontweight='bold', pad=10)
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        # 添加底部共享 colorbar
        fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.09, hspace=0.24, wspace=0.12)
        cbar_ax = fig.add_axes([0.18, 0.04, 0.64, 0.018])
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Temperature (°C)', fontsize=14, labelpad=8)
        cbar.ax.tick_params(labelsize=12, pad=4)
        
        save_path = self.output_dir / 'Figure3_SkipConnections.png'
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        if verbose:
            print(f"✅ Figure 3 saved: {save_path}")
        plt.close()
        
        return {'save_path': str(save_path)}
    
    def generate_fig4_nan_handling(self, sample_data: np.ndarray, verbose: bool = True) -> Dict:
        """
        Figure 4: NaN Handling Mechanism
        """
        fig, axes = plt.subplots(
            3,
            2,
            figsize=(12, 12),
            subplot_kw={'projection': ccrs.PlateCarree()},
        )
        
        # 准备数据
        nan_mask = np.isnan(sample_data)
        data_replaced = np.nan_to_num(sample_data, nan=0.0)
        
        device = next(self.model.parameters()).device
        sample_input = torch.from_numpy(sample_data).unsqueeze(0).unsqueeze(0).float().to(device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_input)
        
        data_raw = output[0, 0].cpu().numpy()
        data_masked = data_raw.copy()
        data_masked[nan_mask] = np.nan
        
        # 绘制
        titles = ['(a) Input SST (Land as NaN)',
                 '(b) NaN Mask (Red=Land)',
                 '(c) Preprocessed (NaN→0)',
                 '(d) Model Output (Raw)',
                 '(e) Final Output (Land Restored)']
        
        datas = [sample_data, nan_mask.astype(float), data_replaced, data_raw, data_masked]
        cmaps = ['RdYlBu_r', 'RdYlGn_r', 'RdYlBu_r', 'RdYlBu_r', 'RdYlBu_r']
        
        axes_flat = axes.ravel()
        for i, (title, data, cmap) in enumerate(zip(titles, datas, cmaps)):
            ax = axes_flat[i]
            
            im = ax.imshow(data, cmap=cmap, origin='lower',
                          extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.3)
            if i == 4:
                ax.add_feature(cfeature.LAND, facecolor='gray', alpha=0.5)
            
            ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05,
                               fraction=0.08, aspect=18)
            if cmap == 'RdYlBu_r':
                cbar.set_label('Temperature (\u00b0C)', fontsize=14, labelpad=8)
            cbar.ax.tick_params(labelsize=14, pad=5)

        for ax in axes_flat[len(titles):]:
            ax.axis('off')
        
        fig.subplots_adjust(hspace=0.72, wspace=0.18)
        save_path = self.output_dir / 'Figure4_NaNHandling.png'
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        if verbose:
            print(f"✅ Figure 4 saved: {save_path}")
        plt.close()
        
        return {'save_path': str(save_path)}
    
    def generate_fig5_vertical_reconstruction(self, sample_data: np.ndarray, verbose: bool = True) -> Dict:
        """
        Figure 5: Vertical Temperature Reconstruction from SST
        Shows the 2D to 3D reconstruction process
        """
        # 准备输入
        sample_input = torch.from_numpy(sample_data).unsqueeze(0).unsqueeze(0).float()
        nan_mask = np.isnan(sample_data)
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            output_3d = self.model(sample_input)  # [1, depth, H, W]
        
        output_np = output_3d[0].cpu().numpy()  # [depth, H, W]
        
        # 应用NaN掩码
        for d in range(output_np.shape[0]):
            output_np[d][nan_mask] = np.nan
        
        # 创建图形
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.25, 
                             height_ratios=[1, 1])
        
        # Row 1: 输入SST
        ax0 = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
        im0 = ax0.imshow(sample_data, cmap='RdYlBu_r', origin='upper',
                        extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                        vmin=-2, vmax=30)
        ax0.coastlines(linewidth=0.3)
        ax0.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
        ax0.set_title('')
        ax0.set_xticks([])
        ax0.set_yticks([])
        plt.colorbar(im0, ax=ax0, orientation='horizontal', pad=0.02, fraction=0.05)
        
        # Row 1: 不同深度层的水平切面 (选择4个代表性深度)
        depth_indices = [0, 3, 6, 9]
        depth_labels = ['Surface (0m)', '100m', '600m', '1500m']
        
        for i, (d_idx, d_label) in enumerate(zip(depth_indices, depth_labels)):
            ax = plt.subplot(gs[0, i+1], projection=ccrs.PlateCarree())
            
            data = output_np[d_idx]
            im = ax.imshow(data, cmap='RdYlBu_r', origin='upper',
                          extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                          vmin=-2, vmax=30)
            ax.coastlines(linewidth=0.3)
            ax.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
            ax.set_title('')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.02, fraction=0.05)
        
        # 创建深度坐标
        depths = np.array([0, 10, 50, 100, 200, 400, 600, 800, 1000, 1500])[:output_np.shape[0]]
        lons = np.linspace(-180, 180, output_np.shape[2])
        lats = np.linspace(-80, 80, output_np.shape[1])
        
        # Row 2: 垂直剖面 - 沿赤道 (lat=0)
        ax_eq = plt.subplot(gs[1, 0:3])
        
        lat_idx = output_np.shape[1] // 2
        vertical_section_eq = output_np[:, lat_idx, :]
        
        LON, DEP = np.meshgrid(lons, depths)
        
        im_eq = ax_eq.contourf(LON, DEP, vertical_section_eq, levels=20, 
                               cmap='RdYlBu_r', vmin=-2, vmax=30)
        ax_eq.contour(LON, DEP, vertical_section_eq, levels=10, 
                     colors='black', linewidths=0.3, alpha=0.3)
        ax_eq.set_xlabel('Longitude (°E)', fontsize=14, labelpad=12)
        ax_eq.set_ylabel('Depth (m)', fontsize=14, labelpad=12)
        ax_eq.set_title('')
        ax_eq.invert_yaxis()
        ax_eq.set_ylim(depths[-1], 0)
        plt.colorbar(im_eq, ax=ax_eq, orientation='vertical', 
                    label='Temperature (°C)', pad=0.02)
        
        # Row 2: 垂直剖面 - 沿特定经度
        ax_pac = plt.subplot(gs[1, 3:5])
        
        lon_idx = output_np.shape[2] // 2
        vertical_section_pac = output_np[:, :, lon_idx]
        
        LAT, DEP2 = np.meshgrid(lats, depths)
        
        im_pac = ax_pac.contourf(LAT, DEP2, vertical_section_pac, levels=20,
                                cmap='RdYlBu_r', vmin=-2, vmax=30)
        ax_pac.contour(LAT, DEP2, vertical_section_pac, levels=10,
                      colors='black', linewidths=0.3, alpha=0.3)
        ax_pac.set_xlabel('Latitude (°N)', fontsize=14, labelpad=12)
        ax_pac.set_ylabel('Depth (m)', fontsize=14, labelpad=12)
        ax_pac.set_title('')
        ax_pac.invert_yaxis()
        ax_pac.set_ylim(depths[-1], 0)
        plt.colorbar(im_pac, ax=ax_pac, orientation='vertical',
                    label='Temperature (°C)', pad=0.02)
        
        plt.tight_layout()
        save_path = self.output_dir / 'Figure5_VerticalReconstruction.png'
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        if verbose:
            print(f"✅ Figure 5 saved: {save_path}")
        plt.close()
        
        return {'save_path': str(save_path), 'output_3d': output_np, 'depths': depths}
    
    def generate_fig6_temperature_profiles(self, output_3d: np.ndarray, 
                                           depths: np.ndarray,
                                           verbose: bool = True) -> Dict:
        """
        Figure 6: Temperature Profiles at Multiple Locations
        多个代表性位置的温度廓线对比图
        """
        H, W = output_3d.shape[1], output_3d.shape[2]
        lats = np.linspace(-80, 80, H)
        lons = np.linspace(-180, 180, W)
        
        # 定义多个代表性观测点（覆盖不同纬度带和海域）
        # (lat_idx, lon_idx, name, color, marker)
        locations = [
            # 热带海域
            (H//2, W//4, 'Tropical Pacific (0°, 90°W)', '#E63946', 'o'),
            (H//2, W*3//4, 'Tropical Indian (0°, 90°E)', '#F4A261', 's'),
            (H//2, W//2, 'Tropical Atlantic (0°, 0°)', '#E76F51', '^'),
            # 副热带
            (int(H*0.625), W//4, 'Subtropical N. Pacific (20°N, 90°W)', '#2A9D8F', 'D'),
            (int(H*0.375), W*3//4, 'Subtropical S. Indian (20°S, 90°E)', '#264653', 'v'),
            # 中纬度
            (int(H*0.75), W//2, 'Mid-lat N. Atlantic (40°N, 0°)', '#457B9D', 'p'),
            (int(H*0.25), W//2, 'Mid-lat S. Atlantic (40°S, 0°)', '#1D3557', 'h'),
            # 高纬度
            (int(H*0.875), W//4, 'Subpolar N. Pacific (60°N, 90°W)', '#9B2335', '*'),
            (int(H*0.125), W*3//4, 'Southern Ocean (60°S, 90°E)', '#023047', 'X'),
        ]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制所有廓线
        for lat_i, lon_i, loc_name, color, marker in locations:
            # 边界检查
            lat_i = min(max(lat_i, 0), H-1)
            lon_i = min(max(lon_i, 0), W-1)
            
            temperature_profile = output_3d[:, lat_i, lon_i]
            
            # 跳过全NaN的廓线
            if np.all(np.isnan(temperature_profile)):
                continue
            
            ax.plot(temperature_profile, depths, marker=marker, linestyle='-', 
                   linewidth=2, markersize=6, color=color, label=loc_name, alpha=0.85)
        
        ax.set_xlabel('Temperature (°C)', fontsize=14, labelpad=12)
        ax.set_ylabel('Depth (m)', fontsize=14, labelpad=12)
        ax.invert_yaxis()
        ax.set_ylim(depths[-1], 0)
        ax.set_xlim(-2, 32)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加图例
        ax.legend(loc='lower right', fontsize=16, framealpha=0.95,
                 ncol=1, borderaxespad=1)
        
        # 添加典型温度结构标注
        ax.axhspan(0, 100, alpha=0.05, color='red', label='_Mixed Layer')
        ax.axhspan(100, 500, alpha=0.05, color='blue', label='_Thermocline')
        
        # 添加文字标注
        ax.text(28, 50, 'Mixed\nLayer', fontsize=16, ha='center', va='center',
               color='gray', style='italic')
        ax.text(28, 300, 'Thermocline', fontsize=16, ha='center', va='center',
               color='gray', style='italic')
        ax.text(28, 1000, 'Deep\nWater', fontsize=16, ha='center', va='center',
               color='gray', style='italic')
        
        plt.tight_layout()
        save_path = self.output_dir / 'Figure6_TemperatureProfiles.png'
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        if verbose:
            print(f"✅ Figure 6 saved: {save_path}")
        plt.close()
        
        return {'save_path': str(save_path)}
    


def analyze_unet3d_nature(model: nn.Module,
                          sample_data: Optional[np.ndarray] = None,
                          output_dir: str = "out/models/nature",
                          groups: Optional[List[str]] = None,
                          verbose: bool = True) -> Dict:
    """
    Analyze UNet3D model with Nature-quality figures
    
    Args:
        model: UNet3D model instance
        sample_data: Sample SST data [H, W]
        output_dir: Output directory
        groups: Analysis groups to generate
        verbose: Print detailed information
        
    Returns:
        Dictionary of analysis results
    
    Example:
        >>> from src.models.Profile.UNet3D import UNet3D
        >>> model = UNet3D(base_channels=128)
        >>> results = analyze_unet3d_nature(model)
    """
    analyzer = UNet3DNatureAnalyzer(model, output_dir)
    return analyzer.analyze_all(sample_data, verbose, groups)

