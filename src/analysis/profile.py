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
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1
rcParams['axes.unicode_minus'] = False  # 正确显示负号


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
                     'fig4_nan_handling']
        
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
                fontsize=12, fontweight='bold', ha='center')
        
        # 简化的架构图
        architecture_text = """
        Input (SST)           Encoder              Bottleneck            Decoder              Output (Profile)
        [1, H, W]    →    [64→128→256→512]  →   [1024]   →   [512→256→128→64]  →   [10, H, W]
                          ↓                                              ↑
                          Skip Connections (Concat)
        """
        ax1.text(0.5, 0.5, architecture_text, transform=ax1.transAxes,
                fontfamily='monospace', fontsize=9, ha='center', va='center')
        ax1.axis('off')
        
        # (b) Parameters statistics
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.text(0.05, 0.95, '(b) Parameter Distribution', transform=ax2.transAxes,
                fontsize=11, fontweight='bold')
        
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
                fontsize=11, fontweight='bold')
        
        layers = ['Input', 'Inc', 'Down1', 'Down2', 'Down3', 'Down4\n(Bottleneck)']
        rf_sizes = [1, 5, 15, 35, 75, 155]
        
        ax3.plot(range(len(layers)), rf_sizes, 'o-', linewidth=2, 
                markersize=8, color='#2E86AB')
        ax3.set_xticks(range(len(layers)))
        ax3.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Receptive Field (pixels)', fontsize=10)
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
        sample_input = torch.from_numpy(sample_data).unsqueeze(0).unsqueeze(0).float()
        
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
        
        # 绘制
        fig = plt.figure(figsize=(12, 8))
        
        layer_names = ['(a) Input\nSST', '(b) Inc\n[64ch]', '(c) Down1\n[128ch]', 
                      '(d) Down2\n[256ch]', '(e) Down3\n[512ch]', '(f) Down4\n[1024ch]',
                      '(g) Up1\n[512ch]', '(h) Up2\n[256ch]', '(i) Up3\n[128ch]', 
                      '(j) Up4\n[64ch]', '(k) Output\n[10ch]']
        layer_keys = [None, 'inc', 'down1', 'down2', 'down3', 'down4',
                      'up1', 'up2', 'up3', 'up4', None]
        
        for idx, (layer_name, layer_key) in enumerate(zip(layer_names, layer_keys)):
            ax = plt.subplot(3, 4, idx + 1, projection=ccrs.PlateCarree())
            
            if layer_key is None:
                if idx == 0:
                    data = sample_input[0, 0].cpu().numpy()
                else:
                    data = output[0, 0].cpu().numpy()
            else:
                data = feature_maps[layer_key][0, 0].cpu().numpy()
            
            cmap = 'RdYlBu_r' if idx <= 5 else 'viridis'
            im = ax.imshow(data, cmap=cmap, origin='upper',
                          extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree())
            
            ax.coastlines(linewidth=0.3)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
            ax.set_title(layer_name, fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.02, 
                        fraction=0.05, aspect=30)
        
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
        """
        # 创建无跳跃连接的对比模型
        class UNet3DNoSkip(nn.Module):
            def __init__(self, base_channels):
                super().__init__()
                from src.models.Profile.UNet3D import DoubleConv, Down
                
                self.inc = DoubleConv(1, base_channels)
                self.down1 = Down(base_channels, base_channels * 2)
                self.down2 = Down(base_channels * 2, base_channels * 4)
                self.down3 = Down(base_channels * 4, base_channels * 8)
                self.down4 = Down(base_channels * 8, base_channels * 16)
                
                self.up1 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    DoubleConv(base_channels * 16, base_channels * 8))
                self.up2 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    DoubleConv(base_channels * 8, base_channels * 4))
                self.up3 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    DoubleConv(base_channels * 4, base_channels * 2))
                self.up4 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    DoubleConv(base_channels * 2, base_channels))
                
                self.outc = nn.Conv2d(base_channels, 10, kernel_size=1)
            
            def forward(self, x):
                x = torch.nan_to_num(x, nan=0.0)
                x = self.inc(x)
                x = self.down1(x)
                x = self.down2(x)
                x = self.down3(x)
                x = self.down4(x)
                x = self.up1(x)
                x = self.up2(x)
                x = self.up3(x)
                x = self.up4(x)
                return self.outc(x)
        
        base_channels = self.model.inc.double_conv[0].out_channels
        model_no_skip = UNet3DNoSkip(base_channels)
        
        sample_input = torch.from_numpy(sample_data).unsqueeze(0).unsqueeze(0).float()
        
        self.model.eval()
        model_no_skip.eval()
        
        with torch.no_grad():
            output_with_skip = self.model(sample_input)
            output_no_skip = model_no_skip(sample_input)
        
        # 绘制对比
        fig, axes = plt.subplots(2, 5, figsize=(14, 6),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        
        depth_indices = [0, 2, 4, 6, 8]
        depth_labels = ['Surface', '50m', '150m', '400m', '900m']
        
        for i, (depth_idx, depth_name) in enumerate(zip(depth_indices, depth_labels)):
            # With skip connections
            ax1 = axes[0, i]
            data_skip = output_with_skip[0, depth_idx].cpu().numpy()
            im1 = ax1.imshow(data_skip, cmap='RdYlBu_r', origin='upper',
                            extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                            vmin=-2, vmax=30)
            ax1.coastlines(linewidth=0.3)
            ax1.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
            
            if i == 0:
                ax1.set_ylabel('With Skip\nConnections', fontsize=10, fontweight='bold')
            ax1.set_title(f'({chr(97+i)}) {depth_name}', fontsize=10)
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Without skip connections
            ax2 = axes[1, i]
            data_no_skip = output_no_skip[0, depth_idx].cpu().numpy()
            im2 = ax2.imshow(data_no_skip, cmap='RdYlBu_r', origin='upper',
                            extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                            vmin=-2, vmax=30)
            ax2.coastlines(linewidth=0.3)
            ax2.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
            
            if i == 0:
                ax2.set_ylabel('Without Skip\nConnections', fontsize=10, fontweight='bold')
            ax2.set_title(f'({chr(102+i)}) {depth_name}', fontsize=10)
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        # 添加统一的colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im1, cax=cbar_ax, label='Temperature (°C)')
        
        plt.tight_layout(rect=[0, 0, 0.9, 1.0])
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
        fig = plt.figure(figsize=(14, 6))
        
        # 准备数据
        nan_mask = np.isnan(sample_data)
        data_replaced = np.nan_to_num(sample_data, nan=0.0)
        
        sample_input = torch.from_numpy(sample_data).unsqueeze(0).unsqueeze(0).float()
        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_input)
        
        data_raw = output[0, 0].cpu().numpy()
        data_masked = data_raw.copy()
        data_masked[nan_mask] = np.nan
        
        # 绘制
        titles = ['(a) Input SST\n(Land as NaN)', 
                 '(b) NaN Mask\n(Red=Land)', 
                 '(c) Preprocessed\n(NaN→0)',
                 '(d) Model Output\n(Raw)',
                 '(e) Final Output\n(Land Restored)']
        
        datas = [sample_data, nan_mask.astype(float), data_replaced, data_raw, data_masked]
        cmaps = ['RdYlBu_r', 'RdYlGn_r', 'RdYlBu_r', 'RdYlBu_r', 'RdYlBu_r']
        
        for i, (title, data, cmap) in enumerate(zip(titles, datas, cmaps)):
            ax = plt.subplot(1, 5, i+1, projection=ccrs.PlateCarree())
            
            im = ax.imshow(data, cmap=cmap, origin='upper',
                          extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.3)
            if i == 4:
                ax.add_feature(cfeature.LAND, facecolor='gray', alpha=0.5)
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.02,
                        fraction=0.05, aspect=20)
        
        plt.tight_layout()
        save_path = self.output_dir / 'Figure4_NaNHandling.png'
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        if verbose:
            print(f"✅ Figure 4 saved: {save_path}")
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

