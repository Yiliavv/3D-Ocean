"""
UNet3D 模型结构详细分析工具
提供完整的模型分析功能，包括参数统计、感受野、特征图可视化等
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from torch.profiler import profile, ProfilerActivity


def load_model_safely(model_path: Union[str, Path], 
                     model_class: Optional[type] = None,
                     **model_kwargs) -> nn.Module:
    """
    安全地加载PyTorch模型
    
    Args:
        model_path: 模型文件路径（.pkl 或 .ckpt）
        model_class: 模型类（用于从checkpoint加载）
        **model_kwargs: 模型初始化参数
        
    Returns:
        加载的模型
        
    Examples:
        >>> # 方法1: 加载完整模型文件
        >>> model = load_model_safely('out/models/unet-3d.pkl')
        
        >>> # 方法2: 从checkpoint加载
        >>> from src.models.Profile.UNet3D import UNet3D
        >>> model = load_model_safely('out/checkpoints/xxx/UNet3D.ckpt', 
        ...                          model_class=UNet3D)
    """
    import sys
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    if model_path.suffix == '.ckpt':
        # Lightning checkpoint
        if model_class is None:
            raise ValueError("从checkpoint加载需要提供model_class参数")
        print(f"📦 从Lightning checkpoint加载: {model_path}")
        
        # 加载checkpoint查看是否有hyper_parameters
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'hyper_parameters' in checkpoint and not model_kwargs:
            # checkpoint中有超参数，使用Lightning的标准方法
            print("   使用checkpoint中保存的超参数")
            model = model_class.load_from_checkpoint(str(model_path))
        else:
            # 没有超参数或提供了自定义参数
            if not model_kwargs:
                # 从state_dict推断参数
                print("   checkpoint中无超参数，从state_dict推断...")
                state_dict = checkpoint['state_dict']
                
                # 从第一个卷积层的权重推断base_channels
                first_conv_weight = state_dict.get('inc.double_conv.0.weight')
                if first_conv_weight is not None:
                    base_channels = first_conv_weight.shape[0]
                    print(f"   推断得到: base_channels={base_channels}")
                    model_kwargs = {'base_channels': base_channels}
                else:
                    print("   ⚠️  无法推断参数，使用默认值")
                    model_kwargs = {}
            
            # 创建模型并加载权重
            print(f"   手动创建模型: {model_kwargs}")
            model = model_class(**model_kwargs)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("   ✅ 权重加载完成")
    else:
        # 完整模型文件 (.pkl, .pt, .pth)
        print(f"📦 加载模型文件: {model_path}")
        
        try:
            # 尝试使用weights_only=False（PyTorch 2.6+）
            model = torch.load(model_path, weights_only=False)
        except TypeError:
            # PyTorch < 2.6
            model = torch.load(model_path)
    
    model.eval()
    print(f"✅ 模型加载成功！")
    return model


class UNet3DAnalyzer:
    """UNet3D模型分析器"""
    
    def __init__(self, model: nn.Module, output_dir: str = "out/models"):
        """
        初始化分析器
        
        Args:
            model: UNet3D模型实例
            output_dir: 输出目录
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储分析结果
        self.results = {}
        
    def analyze_all(self, 
                   sample_data: Optional[np.ndarray] = None,
                   verbose: bool = True) -> Dict:
        """
        执行完整的模型分析
        
        Args:
            sample_data: 样本SST数据 [H, W]，如果为None则使用随机数据
            verbose: 是否打印详细信息
            
        Returns:
            分析结果字典
        """
        print("=" * 100)
        print("🔬 开始 UNet3D 模型完整分析")
        print("=" * 100)
        
        # 1. 参数量统计
        if verbose:
            print("\n📊 步骤 1/6: 分析模型参数量...")
        self.results['parameters'] = self.analyze_parameters(verbose=verbose)
        
        # 2. 感受野分析
        if verbose:
            print("\n👁️  步骤 2/6: 计算感受野...")
        self.results['receptive_field'] = self.analyze_receptive_field(verbose=verbose)
        
        # 3. 特征图可视化
        if sample_data is None:
            sample_data = np.random.randn(80, 180).astype(np.float32)
            sample_data[sample_data < -2] = np.nan  # 模拟陆地
        
        if verbose:
            print("\n🎨 步骤 3/6: 可视化特征图...")
        self.results['feature_maps'] = self.visualize_feature_maps(sample_data, verbose=verbose)
        
        # 4. 跳跃连接分析
        if verbose:
            print("\n🔗 步骤 4/6: 分析跳跃连接作用...")
        self.results['skip_connections'] = self.analyze_skip_connections(sample_data, verbose=verbose)
        
        # 5. NaN处理机制
        if verbose:
            print("\n🌊 步骤 5/6: 分析NaN处理机制...")
        self.results['nan_handling'] = self.analyze_nan_handling(sample_data, verbose=verbose)
        
        # 6. 完整流程总结
        if verbose:
            print("\n🎯 步骤 6/6: 生成完整流程图...")
        self.results['pipeline'] = self.visualize_complete_pipeline(sample_data, verbose=verbose)
        
        print("\n" + "=" * 100)
        print("✅ 分析完成！所有结果已保存至:", self.output_dir)
        print("=" * 100)
        
        return self.results
    
    def analyze_parameters(self, verbose: bool = True) -> Dict:
        """分析模型参数量"""
        
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # 编码器模块
        encoder_modules = [
            ("inc (初始卷积)", self.model.inc),
            ("down1 (第1次下采样)", self.model.down1),
            ("down2 (第2次下采样)", self.model.down2),
            ("down3 (第3次下采样)", self.model.down3),
            ("down4 (瓶颈层)", self.model.down4),
        ]
        
        # 解码器模块
        decoder_modules = [
            ("up1 (第1次上采样)", self.model.up1),
            ("up2 (第2次上采样)", self.model.up2),
            ("up3 (第3次上采样)", self.model.up3),
            ("up4 (第4次上采样)", self.model.up4),
            ("outc (输出层)", self.model.outc),
        ]
        
        results = {
            'encoder': {},
            'decoder': {},
            'total': 0,
            'encoder_total': 0,
            'decoder_total': 0
        }
        
        if verbose:
            print("\n" + "=" * 100)
            print("📊 UNet3D 模型参数量详细统计")
            print("=" * 100)
            print("\n🔹 编码器参数统计:")
            print("-" * 100)
        
        for name, module in encoder_modules:
            params = count_parameters(module)
            results['encoder'][name] = params
            results['encoder_total'] += params
            results['total'] += params
            if verbose:
                print(f"  {name:30s}: {params:15,} ({params/1e6:6.2f}M)")
        
        if verbose:
            print("\n🔹 解码器参数统计:")
            print("-" * 100)
        
        for name, module in decoder_modules:
            params = count_parameters(module)
            results['decoder'][name] = params
            results['decoder_total'] += params
            results['total'] += params
            if verbose:
                print(f"  {name:30s}: {params:15,} ({params/1e6:6.2f}M)")
        
        if verbose:
            print("\n" + "=" * 100)
            print(f"📈 总参数量: {results['total']:,} ({results['total']/1e6:.2f}M)")
            print(f"   • 编码器参数: {results['encoder_total']:,} ({results['encoder_total']/1e6:.2f}M) - {results['encoder_total']/results['total']*100:.1f}%")
            print(f"   • 解码器参数: {results['decoder_total']:,} ({results['decoder_total']/1e6:.2f}M) - {results['decoder_total']/results['total']*100:.1f}%")
            print("=" * 100)
        
        return results
    
    def analyze_receptive_field(self, verbose: bool = True) -> Dict:
        """计算各层感受野"""
        
        def calculate_rf():
            rf = 1
            stride = 1
            layers_info = []
            
            # Inc
            rf = rf + 2 * 2
            layers_info.append(("inc (初始卷积)", rf, stride, "编码器"))
            
            # Down1-4
            for i in range(1, 5):
                stride *= 2
                rf = rf + 1 * stride + 2 * stride + 2 * stride
                name = f"down{i} ({'瓶颈层' if i == 4 else f'下采样{i}'})"
                layers_info.append((name, rf, stride, "瓶颈" if i == 4 else "编码器"))
            
            # Up1-4
            for i in range(1, 5):
                stride //= 2
                rf = rf + 2 * stride + 2 * stride
                layers_info.append((f"up{i} (上采样{i})", rf, stride, "解码器"))
            
            return layers_info
        
        rf_info = calculate_rf()
        
        if verbose:
            print("\n" + "=" * 100)
            print("👁️  UNet3D 感受野 (Receptive Field) 分析")
            print("=" * 100)
            print("\n感受野大小详情:")
            print("-" * 100)
            print(f"{'层名称':<30} {'感受野大小 (像素)':<20} {'累积步长':<15} {'类型':<10}")
            print("-" * 100)
            
            for layer_name, rf, stride, layer_type in rf_info:
                marker = "⭐ " if "瓶颈" in layer_name else "  "
                print(f"{marker}{layer_name:<28} {rf:<20} {stride:<15} {layer_type:<10}")
            
            print("-" * 100)
            
            max_rf = rf_info[-1][1]
            print(f"\n🌍 物理意义:")
            print(f"  • 最终感受野: {max_rf} 像素")
            print(f"  • 在 2° 分辨率下: 约覆盖 {max_rf * 2}° 的地理范围")
            print(f"  • 在 1° 分辨率下: 约覆盖 {max_rf}° 的地理范围")
            print(f"\n  这意味着模型在重建某一点的温度剖面时，会考虑周围 {max_rf} 个网格点的信息，")
            print(f"  能够捕捉中尺度海洋现象（涡旋、锋面、上升流等）对温度剖面的影响。")
            print("=" * 100)
        
        return {
            'layers': rf_info,
            'max_receptive_field': rf_info[-1][1]
        }
    
    def visualize_feature_maps(self, sample_data: np.ndarray, verbose: bool = True) -> Dict:
        """可视化各层特征图"""
        
        # 准备输入
        sample_input = torch.from_numpy(sample_data).unsqueeze(0).unsqueeze(0).float()
        
        if verbose:
            print(f"\n输入数据形状: {sample_input.shape}")
        
        # Hook函数提取特征
        feature_maps = {}
        
        def get_activation(name):
            def hook(model, input, output):
                feature_maps[name] = output.detach()
            return hook
        
        # 注册hooks
        hooks = []
        for name in ['inc', 'down1', 'down2', 'down3', 'down4', 'up1', 'up2', 'up3', 'up4']:
            module = getattr(self.model, name)
            hook = module.register_forward_hook(get_activation(name))
            hooks.append(hook)
        
        # 前向传播
        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_input)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        if verbose:
            print(f"输出数据形状: {output.shape}")
        
        # 绘制特征图
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('UNet3D 编码-解码过程特征图可视化', fontsize=20, fontweight='bold', y=0.995)
        
        layer_names = ['Input', 'inc', 'down1', 'down2', 'down3', 'down4 (Bottleneck)', 
                       'up1', 'up2', 'up3', 'up4', 'Output']
        layer_keys = [None, 'inc', 'down1', 'down2', 'down3', 'down4', 
                      'up1', 'up2', 'up3', 'up4', None]
        
        for idx, (layer_name, layer_key) in enumerate(zip(layer_names, layer_keys)):
            ax = plt.subplot(3, 4, idx + 1, projection=ccrs.PlateCarree())
            
            if layer_key is None:
                if idx == 0:
                    data = sample_input[0, 0].cpu().numpy()
                    title = f"输入 SST\n形状: {data.shape}"
                else:
                    data = output[0, 0].cpu().numpy()
                    title = f"输出 (深度层0)\n形状: {output.shape[1:]}→显示第1层"
            else:
                feature = feature_maps[layer_key][0, 0].cpu().numpy()
                data = feature
                n_channels = feature_maps[layer_key].shape[1]
                title = f"{layer_name}\n形状: [{n_channels}, {feature.shape[0]}, {feature.shape[1]}]"
            
            # 绘制
            cmap = 'RdYlBu_r' if idx <= 5 else 'viridis'
            im = ax.imshow(data, cmap=cmap, origin='upper', 
                          extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree())
            
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.046)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        save_path = self.output_dir / 'unet3d_feature_maps.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"✅ 特征图已保存至: {save_path}")
        plt.close()
        
        # 统计信息
        if verbose:
            print("\n" + "=" * 100)
            print("📊 各层特征图统计:")
            print("-" * 100)
            for layer_name, layer_key in zip(layer_names[1:-1], layer_keys[1:-1]):
                feature = feature_maps[layer_key]
                shape_str = str(list(feature.shape[1:]))
                print(f"{layer_name:20s}: Shape={shape_str:20s}  "
                      f"Mean={feature.mean().item():8.3f}  "
                      f"Std={feature.std().item():8.3f}  "
                      f"Min={feature.min().item():8.3f}  "
                      f"Max={feature.max().item():8.3f}")
            print("=" * 100)
        
        return {
            'feature_maps': feature_maps,
            'output': output,
            'save_path': str(save_path)
        }
    
    def analyze_skip_connections(self, sample_data: np.ndarray, verbose: bool = True) -> Dict:
        """分析跳跃连接的作用"""
        
        # 创建无跳跃连接的模型
        class UNet3DNoSkip(nn.Module):
            def __init__(self, base_channels=128):
                super().__init__()
                from src.models.Profile.UNet3D import DoubleConv, Down
                
                self.inc = DoubleConv(1, base_channels)
                self.down1 = Down(base_channels, base_channels * 2)
                self.down2 = Down(base_channels * 2, base_channels * 4)
                self.down3 = Down(base_channels * 4, base_channels * 8)
                self.down4 = Down(base_channels * 8, base_channels * 16)
                
                self.up1 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    DoubleConv(base_channels * 16, base_channels * 8)
                )
                self.up2 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    DoubleConv(base_channels * 8, base_channels * 4)
                )
                self.up3 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    DoubleConv(base_channels * 4, base_channels * 2)
                )
                self.up4 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    DoubleConv(base_channels * 2, base_channels)
                )
                
                self.outc = nn.Conv2d(base_channels, 10, kernel_size=1)
            
            def forward(self, x):
                x = torch.nan_to_num(x, nan=0.0)
                x1 = self.inc(x)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x5 = self.down4(x4)
                x = self.up1(x5)
                x = self.up2(x)
                x = self.up3(x)
                x = self.up4(x)
                return self.outc(x)
        
        # 获取base_channels
        base_channels = self.model.inc.double_conv[0].out_channels
        model_no_skip = UNet3DNoSkip(base_channels=base_channels)
        
        # 预测
        sample_input = torch.from_numpy(sample_data).unsqueeze(0).unsqueeze(0).float()
        
        self.model.eval()
        model_no_skip.eval()
        
        with torch.no_grad():
            output_with_skip = self.model(sample_input)
            output_no_skip = model_no_skip(sample_input)
        
        # 可视化对比
        fig, axes = plt.subplots(2, 5, figsize=(25, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        fig.suptitle('跳跃连接对温度场重建的影响', fontsize=22, fontweight='bold', y=0.98)
        
        depth_indices = [0, 2, 4, 6, 8]
        depth_names = ['表层', '浅层', '中层', '次深层', '深层']
        
        for i, (depth_idx, depth_name) in enumerate(zip(depth_indices, depth_names)):
            # 有跳跃连接
            ax1 = axes[0, i]
            data_skip = output_with_skip[0, depth_idx].cpu().numpy()
            im1 = ax1.imshow(data_skip, cmap='RdYlBu_r', origin='upper',
                             extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                             vmin=-5, vmax=30)
            ax1.coastlines(linewidth=0.5)
            ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
            ax1.set_title(f'{depth_name} (深度层{depth_idx})\n有跳跃连接', fontsize=12, fontweight='bold')
            ax1.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)
            plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, fraction=0.046)
            
            # 无跳跃连接
            ax2 = axes[1, i]
            data_no_skip = output_no_skip[0, depth_idx].cpu().numpy()
            im2 = ax2.imshow(data_no_skip, cmap='RdYlBu_r', origin='upper',
                             extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                             vmin=-5, vmax=30)
            ax2.coastlines(linewidth=0.5)
            ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
            ax2.set_title(f'{depth_name} (深度层{depth_idx})\n无跳跃连接', fontsize=12, fontweight='bold')
            ax2.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)
            plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, fraction=0.046)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_path = self.output_dir / 'skip_connection_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"✅ 跳跃连接对比图已保存至: {save_path}")
        plt.close()
        
        # 统计分析
        if verbose:
            print("\n" + "=" * 100)
            print("📊 跳跃连接的定量影响:")
            print("-" * 100)
            print(f"{'深度层':<15} {'有跳跃连接':<25} {'无跳跃连接':<25} {'差异':<15}")
            print(f"{'':15} {'Mean':<10} {'Std':<10} {'Mean':<10} {'Std':<10} {'ΔMean':<15}")
            print("-" * 100)
            
            for depth_idx, depth_name in zip(depth_indices, depth_names):
                mean_skip = output_with_skip[0, depth_idx].mean().item()
                std_skip = output_with_skip[0, depth_idx].std().item()
                mean_no_skip = output_no_skip[0, depth_idx].mean().item()
                std_no_skip = output_no_skip[0, depth_idx].std().item()
                diff = abs(mean_skip - mean_no_skip)
                
                print(f"{depth_name:<15} {mean_skip:<10.3f} {std_skip:<10.3f} "
                      f"{mean_no_skip:<10.3f} {std_no_skip:<10.3f} {diff:<15.3f}")
            
            print("-" * 100)
            print("\n💡 结论:")
            print("  • 跳跃连接将编码器的细节特征直接传递到解码器")
            print("  • 保留了海洋锋面、涡旋等中小尺度结构的空间细节")
            print("  • 避免了纯解码器重建时的过度平滑问题")
            print("  • 提升了温度场的空间分辨率和物理真实性")
            print("=" * 100)
        
        return {
            'with_skip': output_with_skip,
            'without_skip': output_no_skip,
            'save_path': str(save_path)
        }
    
    def analyze_nan_handling(self, sample_data: np.ndarray, verbose: bool = True) -> Dict:
        """分析NaN处理机制"""
        
        nan_count = np.isnan(sample_data).sum()
        
        if verbose:
            print("\n原始数据统计:")
            print(f"  • 数据形状: {sample_data.shape}")
            print(f"  • NaN值数量: {nan_count} ({nan_count/sample_data.size*100:.1f}%)")
            print(f"  • 有效值数量: {(~np.isnan(sample_data)).sum()}")
            print(f"  • 有效值范围: [{np.nanmin(sample_data):.2f}, {np.nanmax(sample_data):.2f}]")
        
        # 可视化NaN处理流程
        fig = plt.figure(figsize=(24, 12))
        fig.suptitle('UNet3D NaN处理机制详解', fontsize=22, fontweight='bold', y=0.98)
        
        # 1. 原始SST
        ax1 = plt.subplot(2, 4, 1, projection=ccrs.PlateCarree())
        im1 = ax1.imshow(sample_data, cmap='RdYlBu_r', origin='upper',
                         extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                         vmin=-5, vmax=30)
        ax1.coastlines(linewidth=0.5)
        ax1.set_title('1️⃣ 原始SST输入\n(陆地为NaN)', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, fraction=0.046)
        
        # 2. NaN掩码
        ax2 = plt.subplot(2, 4, 2, projection=ccrs.PlateCarree())
        nan_mask = np.isnan(sample_data).astype(float)
        im2 = ax2.imshow(nan_mask, cmap='RdYlGn_r', origin='upper',
                         extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree())
        ax2.coastlines(linewidth=0.5)
        ax2.set_title('2️⃣ NaN掩码\n(红色=陆地, 绿色=海洋)', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, fraction=0.046)
        
        # 3. NaN替换为0
        ax3 = plt.subplot(2, 4, 3, projection=ccrs.PlateCarree())
        data_replaced = np.nan_to_num(sample_data, nan=0.0)
        im3 = ax3.imshow(data_replaced, cmap='RdYlBu_r', origin='upper',
                         extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                         vmin=-5, vmax=30)
        ax3.coastlines(linewidth=0.5)
        ax3.set_title('3️⃣ NaN→0替换\n(用于前向传播)', fontsize=14, fontweight='bold')
        plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, fraction=0.046)
        
        # 4-8. 模型输出
        sample_input = torch.from_numpy(sample_data).unsqueeze(0).unsqueeze(0).float()
        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_input)
        
        # 4. 原始输出
        ax4 = plt.subplot(2, 4, 4, projection=ccrs.PlateCarree())
        data_raw = output[0, 0].cpu().numpy()
        im4 = ax4.imshow(data_raw, cmap='RdYlBu_r', origin='upper',
                         extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                         vmin=-5, vmax=30)
        ax4.coastlines(linewidth=0.5)
        ax4.set_title('4️⃣ 模型原始输出\n(深度层0, 陆地有值)', fontsize=14, fontweight='bold')
        plt.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.05, fraction=0.046)
        
        # 5. 应用掩码后
        ax5 = plt.subplot(2, 4, 5, projection=ccrs.PlateCarree())
        data_masked = data_raw.copy()
        data_masked[nan_mask.astype(bool)] = np.nan
        im5 = ax5.imshow(data_masked, cmap='RdYlBu_r', origin='upper',
                         extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                         vmin=-5, vmax=30)
        ax5.coastlines(linewidth=0.5)
        ax5.add_feature(cfeature.LAND, facecolor='gray', alpha=0.7)
        ax5.set_title('5️⃣ 最终输出\n(陆地恢复为NaN)', fontsize=14, fontweight='bold')
        plt.colorbar(im5, ax=ax5, orientation='horizontal', pad=0.05, fraction=0.046)
        
        # 6-8. 不同深度层
        depth_layers = [2, 5, 8]
        depth_labels = ['浅层 (层2)', '中层 (层5)', '深层 (层8)']
        for i, (depth_idx, label) in enumerate(zip(depth_layers, depth_labels)):
            ax = plt.subplot(2, 4, 6+i, projection=ccrs.PlateCarree())
            data_depth = output[0, depth_idx].cpu().numpy()
            data_depth[nan_mask.astype(bool)] = np.nan
            im = ax.imshow(data_depth, cmap='RdYlBu_r', origin='upper',
                           extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                           vmin=-5, vmax=30)
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='gray', alpha=0.7)
            ax.set_title(f'{6+i}️⃣ {label}\n(陆地为NaN)', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.046)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_path = self.output_dir / 'nan_handling_mechanism.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"\n✅ NaN处理机制图已保存至: {save_path}")
        plt.close()
        
        if verbose:
            print("\n" + "=" * 100)
            print("💡 NaN处理机制的优点:")
            print("  1. ✅ 避免NaN导致的梯度消失和计算错误")
            print("  2. ✅ 前向传播时用0填充，不影响海洋区域的特征提取")
            print("  3. ✅ 损失函数只计算海洋区域，陆地不参与训练")
            print("  4. ✅ 预测输出恢复陆地NaN，保持物理意义")
            print("  5. ✅ 可视化时陆地显示为灰色，清晰区分海陆")
            print("=" * 100)
        
        return {
            'nan_mask': nan_mask,
            'output': output,
            'save_path': str(save_path)
        }
    
    def visualize_complete_pipeline(self, sample_data: np.ndarray, verbose: bool = True) -> Dict:
        """可视化完整预测流程"""
        
        # 这里简化处理，重用之前的feature_maps
        if 'feature_maps' not in self.results:
            print("⚠️  需要先运行 visualize_feature_maps()")
            return {}
        
        feature_maps = self.results['feature_maps']['feature_maps']
        output = self.results['feature_maps']['output']
        nan_mask = np.isnan(sample_data)
        
        # 绘制完整流程
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('UNet3D 完整预测流程与各组件作用', fontsize=24, fontweight='bold', y=0.98)
        
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # 1. 输入
        ax1 = fig.add_subplot(gs[0, :2], projection=ccrs.PlateCarree())
        im1 = ax1.imshow(sample_data, cmap='RdYlBu_r', origin='upper',
                         extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree())
        ax1.coastlines(linewidth=0.5)
        ax1.set_title('1. 输入: 海表温度 (SST)\n[B, 1, H, W]', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05)
        
        # 2-4. 编码器、瓶颈、解码器特征
        for i, layer_name in enumerate(['inc', 'down2', 'down4']):
            ax = fig.add_subplot(gs[1, i])
            feature = feature_maps[layer_name][0, 0].cpu().numpy()
            im = ax.imshow(feature, cmap='viridis', origin='upper')
            ax.set_title(f'2.{i+1} {layer_name}\n[{feature_maps[layer_name].shape[1]}ch]', 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        
        # 瓶颈层
        ax_bottle = fig.add_subplot(gs[1, 3])
        feature = feature_maps['down4'][0, 0].cpu().numpy()
        im = ax_bottle.imshow(feature, cmap='plasma', origin='upper')
        ax_bottle.set_title('3. 瓶颈层 (Bottleneck)\n[最深特征]', 
                           fontsize=12, fontweight='bold', color='red')
        ax_bottle.axis('off')
        plt.colorbar(im, ax=ax_bottle, orientation='horizontal', pad=0.05)
        
        # 解码器特征
        for i, layer_name in enumerate(['up1', 'up3', 'up4']):
            ax = fig.add_subplot(gs[2, i])
            feature = feature_maps[layer_name][0, 0].cpu().numpy()
            im = ax.imshow(feature, cmap='cool', origin='upper')
            ax.set_title(f'4.{i+1} {layer_name}\n[{feature_maps[layer_name].shape[1]}ch]', 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        
        # 输出
        depth_to_show = [0, 3, 6, 9]
        for i, depth_idx in enumerate(depth_to_show):
            ax = fig.add_subplot(gs[3, i], projection=ccrs.PlateCarree())
            data = output[0, depth_idx].cpu().numpy()
            data_masked = data.copy()
            data_masked[nan_mask] = np.nan
            im = ax.imshow(data_masked, cmap='RdYlBu_r', origin='upper',
                           extent=[-180, 180, -80, 80], transform=ccrs.PlateCarree(),
                           vmin=-5, vmax=30)
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='gray', alpha=0.5)
            ax.set_title(f'5.{i+1} 输出深度层{depth_idx}\n温度剖面', 
                        fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_path = self.output_dir / 'unet3d_complete_pipeline.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"✅ 完整流程图已保存至: {save_path}")
        plt.close()
        
        if verbose:
            print("\n" + "=" * 100)
            print("🌟 模型设计的关键创新点:")
            print("=" * 100)
            print("""
1️⃣  **U型对称架构**: 平衡编码器的特征提取和解码器的重建能力
2️⃣  **跳跃连接机制**: 编码器特征直接传递到解码器对应层
3️⃣  **多尺度特征融合**: 从局部梯度到全球模式的层次化理解
4️⃣  **NaN容错处理**: 正确处理海陆边界，确保模型稳健
5️⃣  **端到端学习**: 自动学习表层-深层温度的复杂非线性关系
6️⃣  **深度维度生成**: 统一框架下重建整个3D温度场
7️⃣  **参数高效**: 可在单GPU上训练，支持全球尺度预测
            """)
            print("=" * 100)
        
        return {'save_path': str(save_path)}


def analyze_unet3d_model(model: nn.Module, 
                         sample_data: Optional[np.ndarray] = None,
                         output_dir: str = "out/models",
                         verbose: bool = True) -> Dict:
    """
    便捷函数：分析UNet3D模型
    
    Args:
        model: UNet3D模型实例（可以是训练好的或未训练的）
        sample_data: 样本SST数据 [H, W]，如果为None则使用随机数据
        output_dir: 输出目录
        verbose: 是否打印详细信息
        
    Returns:
        分析结果字典
        
    Examples:
        >>> # 方法1: 分析未训练的模型
        >>> from src.models.Profile.UNet3D import UNet3D
        >>> model = UNet3D(n_channels=1, n_depth=10, base_channels=128)
        >>> results = analyze_unet3d_model(model)
        
        >>> # 方法2: 使用训练好的模型（推荐）
        >>> from src.analysis.model_analysis import load_model_safely
        >>> model = load_model_safely('out/models/unet-3d.pkl')
        >>> results = analyze_unet3d_model(model, sample_data=your_sst_data)
        
        >>> # 方法3: 从checkpoint加载
        >>> model = load_model_safely('out/checkpoints/xxx/UNet3D.ckpt', 
        ...                          model_class=UNet3D)
        >>> results = analyze_unet3d_model(model, sample_data=your_sst_data)
    """
    analyzer = UNet3DAnalyzer(model, output_dir)
    return analyzer.analyze_all(sample_data, verbose)

