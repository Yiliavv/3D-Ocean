import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from src.config.params import WANDB_PROJECT, WANDB_ENTITY
from src.config.area import Area
from lightning.pytorch.loggers import WandbLogger


class AttentionVisualizer:
    """
    注意力权重可视化类
    
    专门负责注意力权重、输入输出 SST 和注意力影响因子的可视化与记录
    """
    
    def __init__(self, logger: Optional[WandbLogger] = None, enabled: bool = True):
        """
        Args:
            logger: WandbLogger 实例
            enabled: 是否启用可视化
        """
        self.logger = logger
        self.enabled = enabled
    
    def log(self, 
            attention_weights: torch.Tensor, 
            step: int = None, 
            recursion_step: int = None,
            head_idx: int = None,
            input_sst: torch.Tensor = None,
            output_sst: torch.Tensor = None,
            attention_impact: torch.Tensor = None) -> None:
        """
        将注意力权重、海表温度通过注意力后的结果和注意力影响因子可视化并记录到 wandb
        
        Args:
            attention_weights: 注意力权重张量
                - 如果来自 RGAttention: [recursion_depth, batch, num_heads, seq_len, seq_len]
                - 如果平均后: [batch, num_heads, seq_len, seq_len] 或 [batch, seq_len, seq_len]
            step: 训练步数或epoch（可选）
            recursion_step: 递归步骤索引（如果有多层递归，用于区分）
            head_idx: 注意力头索引（如果指定，只可视化该头；否则平均所有头）
            input_sst: 输入的海表温度数据 [batch, seq_len, height, width] 或 [batch, seq_len, width, height]
            output_sst: 通过注意力处理后的输出结果 [batch, 1, height, width] 或 [batch, 1, width, height]
            attention_impact: 注意力影响因子，可以是：
                - query_attention 权重 [batch, 1, seq_len]（聚合所有时间步的权重）
                - 或注意力权重对输出的贡献度
        """
        if not self.enabled or not self.logger:
            return
        
        try:
            # 转换注意力权重
            if isinstance(attention_weights, torch.Tensor):
                attn_np = attention_weights.detach().cpu().numpy()
            else:
                attn_np = attention_weights
            
            # 处理不同维度的注意力权重
            attn_np = self._process_attention_weights(attn_np, recursion_step, head_idx)
            if attn_np is None:
                return
            
            # 准备记录的数据字典
            log_dict = {}
            wandb_dpi = 150
            
            # 1. 记录注意力权重矩阵
            log_dict.update(self._visualize_attention_weights(
                attn_np, step, recursion_step, head_idx, wandb_dpi
            ))
            
            # 2. 记录输入的海表温度（如果提供）
            if input_sst is not None:
                log_dict.update(self._visualize_input_sst(
                    input_sst, step, wandb_dpi
                ))
            
            # 3. 记录通过注意力处理后的输出（如果提供）
            if output_sst is not None:
                log_dict.update(self._visualize_output_sst(
                    output_sst, step, wandb_dpi
                ))
            
            # 4. 记录注意力影响因子（如果提供）
            if attention_impact is not None:
                log_dict.update(self._visualize_attention_impact(
                    attention_impact, step, wandb_dpi
                ))
            
            # 一次性记录所有图像到 wandb
            if log_dict:
                self.logger.experiment.log(log_dict)
            
        except Exception as e:
            print(f"⚠️  记录注意力权重到 wandb 失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _process_attention_weights(self, attn_np, recursion_step, head_idx):
        """处理不同维度的注意力权重"""
        # 处理不同维度的注意力权重
        if len(attn_np.shape) == 5:
            # [recursion_depth, batch, num_heads, seq_len, seq_len]
            attn_np = attn_np[:, 0, :, :, :]  # [recursion_depth, num_heads, seq_len, seq_len]
            if recursion_step is not None:
                attn_np = attn_np[recursion_step:recursion_step+1]  # [1, num_heads, seq_len, seq_len]
            attn_np = attn_np.mean(axis=(0, 1))  # [seq_len, seq_len]
        elif len(attn_np.shape) == 4:
            # [batch, num_heads, seq_len, seq_len]
            attn_np = attn_np[0]  # [num_heads, seq_len, seq_len]
            if head_idx is not None:
                attn_np = attn_np[head_idx:head_idx+1]  # [1, seq_len, seq_len]
            attn_np = attn_np.mean(axis=0)  # [seq_len, seq_len]
        elif len(attn_np.shape) == 3:
            # [batch, seq_len, seq_len]
            attn_np = attn_np[0]  # [seq_len, seq_len]
        
        # 确保注意力权重是 2D
        if len(attn_np.shape) != 2:
            print(f"⚠️  注意力权重形状不正确: {attn_np.shape}")
            return None
        
        return attn_np
    
    def _visualize_attention_weights(self, attn_np, step, recursion_step, head_idx, wandb_dpi):
        """可视化注意力权重矩阵"""
        log_dict = {}
        
        fig_attn, ax_attn = plt.subplots(figsize=(10, 8))
        im_attn = ax_attn.imshow(attn_np, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        cbar_attn = plt.colorbar(im_attn, ax=ax_attn)
        cbar_attn.set_label('Attention Weight', rotation=270, labelpad=20)
        
        title_parts = ['Attention Weights Matrix']
        if recursion_step is not None:
            title_parts.append(f'Recursion Step {recursion_step}')
        if head_idx is not None:
            title_parts.append(f'Head {head_idx}')
        else:
            title_parts.append('Averaged Heads')
        if step is not None:
            title_parts.append(f'Step {step}')
        
        ax_attn.set_title(' / '.join(title_parts), fontsize=14, fontweight='bold')
        ax_attn.set_xlabel('Key Position (To)', fontsize=12)
        ax_attn.set_ylabel('Query Position (From)', fontsize=12)
        ax_attn.set_xticks(range(0, attn_np.shape[1], max(1, attn_np.shape[1] // 10)))
        ax_attn.set_yticks(range(0, attn_np.shape[0], max(1, attn_np.shape[0] // 10)))
        plt.tight_layout()
        
        log_key_attn = 'attention/weights_matrix'
        if recursion_step is not None:
            log_key_attn += f'/recursion_step_{recursion_step}'
        if head_idx is not None:
            log_key_attn += f'/head_{head_idx}'
        else:
            log_key_attn += '/averaged'
        if step is not None:
            log_key_attn += f'/step_{step}'
        
        fig_attn.set_dpi(wandb_dpi)
        log_dict[log_key_attn] = wandb.Image(fig_attn)
        plt.close(fig_attn)
        
        return log_dict
    
    def _visualize_input_sst(self, input_sst, step, wandb_dpi):
        """可视化输入的海表温度"""
        log_dict = {}
        
        sst_input = input_sst.detach().cpu().numpy() if isinstance(input_sst, torch.Tensor) else input_sst
        
        # 处理不同形状：取第一个batch，最后一个时间步
        if len(sst_input.shape) == 4:  # [batch, seq_len, height, width] or [batch, seq_len, width, height]
            if sst_input.shape[1] > 0:  # 有多个时间步，取最后一个
                sst_input = sst_input[0, -1, :, :]  # 最后一个时间步
            else:
                sst_input = sst_input[0, 0, :, :]
        elif len(sst_input.shape) == 3:  # [batch, height, width]
            sst_input = sst_input[0, :, :]
        elif len(sst_input.shape) == 2:  # [height, width]
            pass  # 已经是2D
        
        fig_input, ax_input = plt.subplots(figsize=(10, 8))
        im_input = ax_input.imshow(sst_input, cmap='jet', aspect='auto')
        cbar_input = plt.colorbar(im_input, ax=ax_input)
        cbar_input.set_label('SST (°C)', rotation=270, labelpad=20)
        
        title_input = 'Input SST (Before Attention)'
        if step is not None:
            title_input += f' - Step {step}'
        ax_input.set_title(title_input, fontsize=14, fontweight='bold')
        ax_input.set_xlabel('Longitude', fontsize=12)
        ax_input.set_ylabel('Latitude', fontsize=12)
        plt.tight_layout()
        
        log_key_input = 'attention/input_sst'
        if step is not None:
            log_key_input += f'/step_{step}'
        fig_input.set_dpi(wandb_dpi)
        log_dict[log_key_input] = wandb.Image(fig_input)
        plt.close(fig_input)
        
        return log_dict
    
    def _visualize_output_sst(self, output_sst, step, wandb_dpi):
        """可视化注意力处理后的输出"""
        log_dict = {}
        
        sst_output = output_sst.detach().cpu().numpy() if isinstance(output_sst, torch.Tensor) else output_sst
        
        # 处理不同形状：取第一个batch
        if len(sst_output.shape) == 4:  # [batch, 1, height, width] or [batch, 1, width, height]
            sst_output = sst_output[0, 0, :, :]
        elif len(sst_output.shape) == 3:  # [batch, height, width]
            sst_output = sst_output[0, :, :]
        elif len(sst_output.shape) == 2:  # [height, width]
            pass  # 已经是2D
        
        fig_output, ax_output = plt.subplots(figsize=(10, 8))
        im_output = ax_output.imshow(sst_output, cmap='jet', aspect='auto')
        cbar_output = plt.colorbar(im_output, ax=ax_output)
        cbar_output.set_label('SST (°C)', rotation=270, labelpad=20)
        
        title_output = 'Output SST (After Attention)'
        if step is not None:
            title_output += f' - Step {step}'
        ax_output.set_title(title_output, fontsize=14, fontweight='bold')
        ax_output.set_xlabel('Longitude', fontsize=12)
        ax_output.set_ylabel('Latitude', fontsize=12)
        plt.tight_layout()
        
        log_key_output = 'attention/output_sst'
        if step is not None:
            log_key_output += f'/step_{step}'
        fig_output.set_dpi(wandb_dpi)
        log_dict[log_key_output] = wandb.Image(fig_output)
        plt.close(fig_output)
        
        return log_dict
    
    def _visualize_attention_impact(self, attention_impact, step, wandb_dpi):
        """可视化注意力影响因子"""
        log_dict = {}
        
        impact_np = attention_impact.detach().cpu().numpy() if isinstance(attention_impact, torch.Tensor) else attention_impact
        
        # 处理不同形状的影响因子
        if len(impact_np.shape) == 3:  # [batch, 1, seq_len] - query_attention 权重
            impact_np = impact_np[0, 0, :]  # [seq_len]
        elif len(impact_np.shape) == 2:  # [batch, seq_len]
            impact_np = impact_np[0, :]  # [seq_len]
        elif len(impact_np.shape) == 1:  # [seq_len]
            pass  # 已经是1D
        
        fig_impact, ax_impact = plt.subplots(figsize=(10, 6))
        seq_len_impact = len(impact_np)
        ax_impact.bar(range(seq_len_impact), impact_np, color='steelblue', alpha=0.7)
        ax_impact.set_xlabel('Time Step Index', fontsize=12)
        ax_impact.set_ylabel('Attention Impact Factor', fontsize=12)
        ax_impact.set_title(f'Attention Impact on Each Time Step{(" - Step " + str(step)) if step is not None else ""}', 
                           fontsize=14, fontweight='bold')
        ax_impact.set_xticks(range(seq_len_impact))
        ax_impact.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        log_key_impact = 'attention/impact_factor'
        if step is not None:
            log_key_impact += f'/step_{step}'
        fig_impact.set_dpi(wandb_dpi)
        log_dict[log_key_impact] = wandb.Image(fig_impact)
        plt.close(fig_impact)
        
        return log_dict


class Wandb:
    """
    Wandb 日志记录类
    
    封装所有 wandb 相关的功能，包括初始化、配置记录、指标记录和 checkpoint 保存。
    """
    
    def __init__(self, 
                 uid: str,
                 model_class: Any,
                 dataset_class: Any,
                 area: Area,
                 model_params: dict,
                 dataset_params: dict,
                 trainer_params: dict,
                 enabled: bool = True):
        """
        Args:
            uid: 训练器唯一标识
            model_class: 模型类
            dataset_class: 数据集类
            area: 区域配置
            model_params: 模型参数
            dataset_params: 数据集参数
            trainer_params: 训练参数
            enabled: 是否启用 wandb
        """
        self.uid = uid
        self.model_class = model_class
        self.dataset_class = dataset_class
        self.area = area
        self.model_params = model_params
        self.dataset_params = dataset_params
        self.trainer_params = trainer_params
        self.enabled = enabled
        
        self.logger = None
        self.model = None
        self.attention_visualizer = AttentionVisualizer(enabled=enabled)
    
    def init(self, model: Any) -> None:
        """初始化 wandb logger"""
        if not self.enabled:
            return
        
        try:
            
            self.model = model
            
            # 构建配置字典
            config = self._build_config(model)
            
            # 创建 wandb logger
            # PyTorch Lightning 的 WandbLogger 会自动将 self.log() 记录的指标同步到 wandb
            self.logger = WandbLogger(
                id=self.uid,
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=self.model_class.__name__,
                config=config,
                save_dir=None,  # 不保存本地日志
                log_model=False,  # 不自动记录模型（我们手动记录checkpoint）
            )
            
            # 访问 experiment 属性会触发 wandb.init()
            # 确保 wandb 已初始化后再调用 watch
            _ = self.logger.experiment
            
            # 更新注意力可视化器的 logger
            self.attention_visualizer.logger = self.logger
            
            print(f"\n📊 Wandb 已启用")
            print(f"  • Project: {WANDB_PROJECT}")
            print(f"  • Run ID: {self.uid}")
            print(f"  • Run URL: {self.logger.experiment.url}")
            print(f"  • 模型监控: 已开启（梯度监控）\n")
            
        except Exception as e:
            print(f"\n⚠️  Wandb 初始化失败: {str(e)}")
            print(f"  训练将继续，但不记录到 wandb\n")
            self.enabled = False
            self.logger = None
    
    def init_for_prediction(self, model: Any = None) -> None:
        """
        为预测重新初始化 wandb logger（如果训练时已关闭）
        
        如果使用相同的 run ID，会在原来的 run 上继续记录预测结果。
        如果 run 已经 finish，wandb 会创建一个新的 run（因为已完成的 run 不能继续记录）。
        因此建议在训练时不要调用 finish()，或者确保预测时使用相同的 run ID。
        """
        if not self.enabled:
            return
        
        # 如果 logger 已存在且活跃，不需要重新初始化
        if self.logger:
            try:
                # 检查 wandb run 是否仍然活跃
                _ = self.logger.experiment
                return
            except:
                pass  # run 已关闭，需要重新初始化
        
        try:
            if model:
                self.model = model
            
            # 构建配置字典
            config = self._build_config(model) if model else {}
            
            # 创建 wandb logger（使用相同的 run ID 以保持连续性）
            # resume='allow': 如果 run 存在且未完成则恢复，如果已完成或不存在则创建新 run
            # 注意：已完成的 run 无法继续记录，wandb 会自动创建新 run
            self.logger = WandbLogger(
                id=self.uid,
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"{self.model_class.__name__}",
                config=config,
                save_dir=None,
                resume='allow',  # 如果 run 存在则恢复，否则创建新 run
            )
            
            # 访问 experiment 属性会触发 wandb.init()
            _ = self.logger.experiment
            
            # 更新注意力可视化器的 logger
            self.attention_visualizer.logger = self.logger
            
            print(f"✅ Wandb 已重新初始化用于记录预测结果")
            print(f"  • Run ID: {self.uid}")
            print(f"  • Run URL: {self.logger.experiment.url}")
            
        except Exception as e:
            print(f"⚠️  重新初始化 wandb 失败: {str(e)}")
            self.logger = None
    
    def finish(self, train_time: float, checkpoint_callback, close_run: bool = False) -> None:
        """
        训练结束后记录最终指标并保存 checkpoint
        
        Args:
            train_time: 训练时间（秒）
            checkpoint_callback: checkpoint 回调
            close_run: 是否关闭 wandb run（默认: False）
                       - False: 保持 run 活跃，以便后续预测时继续记录
                       - True: 关闭 run（已完成的 run 无法继续记录）
        """
        if not self.enabled or not self.logger:
            return
        
        try:
            # 记录最终指标
            final_metrics = self._build_final_metrics(train_time)
            if final_metrics:
                self.logger.experiment.log(final_metrics)
            
            # 保存 checkpoint 到 wandb artifacts
            self._save_checkpoint(checkpoint_callback)
            
            # 根据参数决定是否关闭 run
            if close_run:
                wandb.finish()
                print(f"📝 Wandb run 已关闭")
            else:
                # 不关闭 run，保持活跃以便后续预测时继续记录
                print(f"📝 Wandb run 保持活跃，预测时可继续记录（Run ID: {self.uid}）")
            
        except Exception as e:
            print(f"\n⚠️  训练结束时 wandb 操作失败: {str(e)}")
    
    def get_lightning_logger(self):
        """返回 WandbLogger 实例（供 PyTorch Lightning 使用）"""
        return self.logger if self.enabled else None
    
    def log_prediction_images(self, offset: int, rmse: float, r2: float,
                              nino_fig, sst_fig, diff_fig) -> None:
        """
        将预测结果图像记录到 wandb
        
        Args:
            offset: 数据偏移量
            rmse: 均方根误差
            r2: 决定系数
            nino_fig: NINO指数图的 matplotlib figure
            sst_fig: 海表温度预测图的 matplotlib figure
            diff_fig: 预测误差图的 matplotlib figure
        """
        if not self.enabled or not self.logger:
            return
        
        try:
            import io
            from PIL import Image as PILImage
            
            # 为了上传到 wandb，需要降低 DPI 以避免图像过大
            # wandb 推荐的 DPI 是 150-200，1200 DPI 会导致图像过大
            wandb_dpi = 500
            
            # 临时提高 PIL 的图像大小限制（因为我们已经降低了 DPI，图像应该不会太大）
            # 但这只是额外的安全措施
            original_max_pixels = PILImage.MAX_IMAGE_PIXELS
            PILImage.MAX_IMAGE_PIXELS = None  # 临时禁用限制
            
            try:
                # 将 figure 渲染到内存缓冲区，使用较低的 DPI
                def fig_to_wandb_image(fig, dpi):
                    """将 matplotlib figure 转换为 wandb Image，使用指定的 DPI"""
                    buf = io.BytesIO()
                    # 保存时指定较低的 DPI
                    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
                    buf.seek(0)
                    # 从缓冲区读取图像并转换为 wandb.Image
                    pil_img = PILImage.open(buf)
                    return wandb.Image(pil_img)
                
                # 使用 logger 的 experiment 对象来记录（确保使用正确的 wandb run）
                self.logger.experiment.log({
                    f"prediction/offset_{offset}/nino_index": fig_to_wandb_image(nino_fig, wandb_dpi),
                    f"prediction/offset_{offset}/sst_prediction": fig_to_wandb_image(sst_fig, wandb_dpi),
                    f"prediction/offset_{offset}/prediction_error": fig_to_wandb_image(diff_fig, wandb_dpi),
                    f"prediction/offset_{offset}/rmse": rmse,
                    f"prediction/offset_{offset}/r2": r2,
                })
            finally:
                # 恢复 PIL 的原始限制
                PILImage.MAX_IMAGE_PIXELS = original_max_pixels
            
        except Exception as e:
            print(f"⚠️  记录预测图像到 wandb 失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def log_attention_weights(self, attention_weights: torch.Tensor, 
                             step: int = None, 
                             recursion_step: int = None,
                             head_idx: int = None,
                             input_sst: torch.Tensor = None,
                             output_sst: torch.Tensor = None,
                             attention_impact: torch.Tensor = None) -> None:
        """
        将注意力权重、海表温度通过注意力后的结果和注意力影响因子可视化并记录到 wandb
        
        此方法委托给 AttentionVisualizer 类处理
        
        Args:
            attention_weights: 注意力权重张量
                - 如果来自 RGAttention: [recursion_depth, batch, num_heads, seq_len, seq_len]
                - 如果平均后: [batch, num_heads, seq_len, seq_len] 或 [batch, seq_len, seq_len]
            step: 训练步数或epoch（可选）
            recursion_step: 递归步骤索引（如果有多层递归，用于区分）
            head_idx: 注意力头索引（如果指定，只可视化该头；否则平均所有头）
            input_sst: 输入的海表温度数据 [batch, seq_len, height, width] 或 [batch, seq_len, width, height]
            output_sst: 通过注意力处理后的输出结果 [batch, 1, height, width] 或 [batch, 1, width, height]
            attention_impact: 注意力影响因子，可以是：
                - query_attention 权重 [batch, 1, seq_len]（聚合所有时间步的权重）
                - 或注意力权重对输出的贡献度
        """
        self.attention_visualizer.log(
            attention_weights=attention_weights,
            step=step,
            recursion_step=recursion_step,
            head_idx=head_idx,
            input_sst=input_sst,
            output_sst=output_sst,
            attention_impact=attention_impact
        )
    
    def _build_config(self, model: Any) -> Dict:
        """构建 wandb 配置字典"""
        config = {
            'model': self.model_class.__name__,
            'dataset': self.dataset_class.__name__,
            'area': {
                'lon': self.area.lon.tolist() if hasattr(self.area.lon, 'tolist') else self.area.lon,
                'lat': self.area.lat.tolist() if hasattr(self.area.lat, 'tolist') else self.area.lat,
                'title': self.area.title,
            },
            'model_params': self.model_params,
            'dataset_params': self.dataset_params,
            'trainer_params': self.trainer_params,
        }
    
        
        return config
    
    def _build_final_metrics(self, train_time: float) -> Dict:
        """构建最终指标字典"""
        final_metrics = {}
        
        if train_time is not None:
            final_metrics['train_time_seconds'] = train_time
        
        # 记录最终的损失值
        if self.model and hasattr(self.model, 'train_loss') and self.model.train_loss:
            final_metrics['final_train_loss'] = self.model.train_loss[-1]
        
        if self.model and hasattr(self.model, 'val_loss') and self.model.val_loss:
            final_metrics['final_val_loss'] = self.model.val_loss[-1]
            final_metrics['best_val_loss'] = min(self.model.val_loss)
        
        return final_metrics
    
    def _save_checkpoint(self, checkpoint_callback) -> None:
        """保存 checkpoint 到 wandb artifacts（使用 PyTorch Lightning 保存的路径）"""
        if not self.enabled or not checkpoint_callback or not self.logger:
            return
        
        try:
            # 使用 PyTorch Lightning 自动保存的最佳 checkpoint 路径
            checkpoint_path = checkpoint_callback.best_model_path
            
            print(f"📦 正在上传 checkpoint 到 wandb...")
            print(f"  • Checkpoint 路径: {checkpoint_path}")
            
            # 使用 logger 的 experiment 对象来记录 artifact（确保在正确的 run 上下文中）
            run = self.logger.experiment
            
            # 检查 run 是否活跃
            if not run:
                raise ValueError("Wandb run 不存在或未初始化")
            
            # 使用实际的 run.id 来命名 artifact（确保一致性）
            # 注意：如果 wandb 复用了旧 run，run.id 可能与 self.uid 不同
            actual_run_id = run.id
            artifact_name = f"{self.model_class.__name__}_{actual_run_id}"
            
            print(f"  • 正在上传 artifact 到 run: {actual_run_id}")
            print(f"  • Artifact 名称: {artifact_name}")
            
            artifact = wandb.Artifact(
                name=artifact_name,
                type='model',
                description=f"{self.model_class.__name__} on {self.area.title}"
            )
            
            artifact.add_file(checkpoint_path)
            
            # 记录 artifact 到 wandb
            run.log_artifact(artifact)

        except Exception as e:
            print(f"⚠️  保存checkpoint到 wandb 失败: {str(e)}")
            import traceback
            traceback.print_exc()