"""
张春玲温度参数模型 - 基于统计关系的上层海洋三维温度场反演

核心方法：
1. 最大角度法（Maximum Angle Method）计算混合层深度（MLD）
2. 温跃层参数计算：温跃层梯度（DT）、下边界深度（Zb）
3. 基于统计相关分析建立 SST 与次表层温度的关联

参考文献：
- 张春玲等, 利用遥感SST反演上层海洋三维温度场
- 张春玲等, 基于梯度依赖FVCOM同化模型的中尺度涡三维结构模拟
"""

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import optim
from scipy import stats
from scipy.interpolate import interp1d

from src.config.constants import deep


class Thermocline(LightningModule):
    """
    张春玲温度参数模型
    
    基于最大角度法和统计关系反演上层海洋三维温度场
    
    参数：
        target_depths: 目标深度层（米），默认使用 Argo 标准深度
        mld_ref_depth: MLD 参考深度（米），默认 10m
        mld_threshold: MLD 阈值温差（°C），默认 0.5
        learning_rate: 学习率（用于统计参数优化）
        use_gradient_dependent: 是否使用梯度依赖方法改进反演精度
    
    输入：
        SST: [batch, height, width] 海表温度
        profiles: [batch, height, width, depth] 训练用温度剖面（仅训练时需要）
    
    输出：
        三维温度场: [batch, height, width, depth]
    """
    
    def __init__(self,
                 target_depths=None,
                 mld_ref_depth=10,
                 mld_threshold=0.5,
                 learning_rate=1e-4,
                 use_gradient_dependent=True):
        super().__init__()
        
        # 目标深度层
        if target_depths is None:
            self.target_depths = deep.copy()
        else:
            self.target_depths = np.array(target_depths)
        
        self.n_depths = len(self.target_depths)
        self.mld_ref_depth = mld_ref_depth
        self.mld_threshold = mld_threshold
        self.learning_rate = learning_rate
        self.use_gradient_dependent = use_gradient_dependent
        
        # 统计回归参数（每个深度层一组参数）
        # T(z) = a(z) * SST + b(z)
        self.register_buffer('regression_slopes', torch.zeros(self.n_depths))
        self.register_buffer('regression_intercepts', torch.zeros(self.n_depths))
        self.register_buffer('correlation_coeffs', torch.zeros(self.n_depths))
        
        # 温跃层参数的空间分布（气候态）
        self.register_buffer('mld_climatology', None)
        self.register_buffer('thermocline_gradient_climatology', None)
        self.register_buffer('thermocline_bottom_climatology', None)
        
        # 训练状态
        self.is_fitted = False
        
        # 损失记录
        self.train_loss = []
        self.val_loss = []
        
        # 保存超参数
        self.save_hyperparameters()
    
    def compute_mld_maximum_angle(self, profile, depths=None):
        """
        最大角度法（Maximum Angle Method）计算混合层深度
        
        原理：
        - 在温度剖面上，从参考点（通常为10m）向下搜索
        - 计算每个深度点与参考点连线的角度
        - 角度最大的点对应混合层底部
        
        Args:
            profile: [depth] 或 [batch, depth] 温度剖面
            depths: 对应的深度值，默认使用 target_depths
        
        Returns:
            mld: 混合层深度（米）
        """
        if depths is None:
            depths = self.target_depths
        
        profile = np.atleast_2d(profile)
        batch_size = profile.shape[0]
        mld_values = np.zeros(batch_size)
        
        # 找到参考深度索引（最接近 mld_ref_depth 的深度）
        ref_idx = np.argmin(np.abs(depths - self.mld_ref_depth))
        
        for b in range(batch_size):
            temp_profile = profile[b]
            
            # 跳过全 NaN 剖面
            if np.all(np.isnan(temp_profile)):
                mld_values[b] = np.nan
                continue
            
            # 参考点温度和深度
            t_ref = temp_profile[ref_idx]
            z_ref = depths[ref_idx]
            
            if np.isnan(t_ref):
                mld_values[b] = np.nan
                continue
            
            max_angle = -np.inf
            mld_idx = ref_idx
            
            # 从参考点向下搜索
            for i in range(ref_idx + 1, len(depths)):
                t_i = temp_profile[i]
                z_i = depths[i]
                
                if np.isnan(t_i):
                    continue
                
                # 计算温差和深度差
                delta_t = t_ref - t_i  # 温度下降为正
                delta_z = z_i - z_ref  # 深度增加为正
                
                if delta_z <= 0:
                    continue
                
                # 计算角度（使用温度变化率）
                # 角度 = arctan(delta_t / delta_z)
                angle = np.arctan2(delta_t, delta_z / 100)  # 归一化深度
                
                if angle > max_angle:
                    max_angle = angle
                    mld_idx = i
            
            mld_values[b] = depths[mld_idx]
        
        return mld_values.squeeze() if batch_size == 1 else mld_values
    
    def compute_mld_threshold(self, profile, depths=None):
        """
        阈值法计算混合层深度（作为最大角度法的补充验证）
        
        Args:
            profile: [depth] 或 [batch, depth] 温度剖面
            depths: 对应的深度值
        
        Returns:
            mld: 混合层深度（米）
        """
        if depths is None:
            depths = self.target_depths
        
        profile = np.atleast_2d(profile)
        batch_size = profile.shape[0]
        mld_values = np.zeros(batch_size)
        
        ref_idx = np.argmin(np.abs(depths - self.mld_ref_depth))
        
        for b in range(batch_size):
            temp_profile = profile[b]
            t_ref = temp_profile[ref_idx]
            
            if np.isnan(t_ref):
                mld_values[b] = np.nan
                continue
            
            # 找到温差超过阈值的第一个深度
            mld_found = False
            for i in range(ref_idx + 1, len(depths)):
                if np.isnan(temp_profile[i]):
                    continue
                
                if np.abs(temp_profile[i] - t_ref) >= self.mld_threshold:
                    # 线性插值获取精确的 MLD
                    if i > 0 and not np.isnan(temp_profile[i-1]):
                        t_prev = temp_profile[i-1]
                        t_curr = temp_profile[i]
                        z_prev = depths[i-1]
                        z_curr = depths[i]
                        
                        # 插值
                        t_threshold = t_ref - self.mld_threshold if t_ref > t_curr else t_ref + self.mld_threshold
                        if t_curr != t_prev:
                            mld_values[b] = z_prev + (t_threshold - t_prev) / (t_curr - t_prev) * (z_curr - z_prev)
                        else:
                            mld_values[b] = z_curr
                    else:
                        mld_values[b] = depths[i]
                    mld_found = True
                    break
            
            if not mld_found:
                mld_values[b] = depths[-1]  # 未找到则取最大深度
        
        return mld_values.squeeze() if batch_size == 1 else mld_values
    
    def compute_thermocline_parameters(self, profile, depths=None):
        """
        计算温跃层参数：梯度（DT）和下边界深度（Zb）
        
        温跃层定义：混合层底部到温度梯度显著减小的深度
        
        Args:
            profile: [depth] 温度剖面
            depths: 对应的深度值
        
        Returns:
            dict: {
                'mld': 混合层深度,
                'thermocline_gradient': 温跃层平均梯度（°C/m）,
                'thermocline_bottom': 温跃层下边界深度,
                'thermocline_top_temp': 温跃层顶部温度,
                'thermocline_bottom_temp': 温跃层底部温度
            }
        """
        if depths is None:
            depths = self.target_depths
        
        # 计算 MLD
        mld = self.compute_mld_maximum_angle(profile, depths)
        
        profile = np.atleast_1d(profile)
        
        # 跳过无效剖面
        if np.isnan(mld) or np.all(np.isnan(profile)):
            return {
                'mld': np.nan,
                'thermocline_gradient': np.nan,
                'thermocline_bottom': np.nan,
                'thermocline_top_temp': np.nan,
                'thermocline_bottom_temp': np.nan
            }
        
        # 找到 MLD 对应的索引
        mld_idx = np.argmin(np.abs(depths - mld))
        
        # 计算温度梯度 dT/dz
        gradients = np.zeros(len(depths) - 1)
        for i in range(len(depths) - 1):
            if not np.isnan(profile[i]) and not np.isnan(profile[i+1]):
                dT = profile[i] - profile[i+1]
                dz = depths[i+1] - depths[i]
                gradients[i] = dT / dz if dz > 0 else 0
            else:
                gradients[i] = np.nan
        
        # 温跃层下边界：梯度开始显著减小的位置
        # 使用梯度阈值（气候态梯度的 10%）
        valid_gradients = gradients[mld_idx:]
        valid_gradients = valid_gradients[~np.isnan(valid_gradients)]
        
        if len(valid_gradients) == 0:
            thermocline_bottom = depths[-1]
            thermocline_gradient = np.nan
        else:
            max_gradient = np.max(valid_gradients)
            gradient_threshold = max_gradient * 0.1
            
            thermocline_bottom = depths[-1]
            for i in range(mld_idx, len(depths) - 1):
                if not np.isnan(gradients[i]) and gradients[i] < gradient_threshold:
                    thermocline_bottom = depths[i]
                    break
            
            # 计算温跃层内的平均梯度
            bottom_idx = np.argmin(np.abs(depths - thermocline_bottom))
            thermocline_gradients = gradients[mld_idx:bottom_idx]
            thermocline_gradients = thermocline_gradients[~np.isnan(thermocline_gradients)]
            thermocline_gradient = np.mean(thermocline_gradients) if len(thermocline_gradients) > 0 else np.nan
        
        # 温跃层顶部和底部温度
        thermocline_top_temp = profile[mld_idx] if not np.isnan(profile[mld_idx]) else np.nan
        bottom_idx = np.argmin(np.abs(depths - thermocline_bottom))
        thermocline_bottom_temp = profile[bottom_idx] if not np.isnan(profile[bottom_idx]) else np.nan
        
        return {
            'mld': mld,
            'thermocline_gradient': thermocline_gradient,
            'thermocline_bottom': thermocline_bottom,
            'thermocline_top_temp': thermocline_top_temp,
            'thermocline_bottom_temp': thermocline_bottom_temp
        }
    
    def fit_regression_parameters(self, sst_data, profile_data):
        """
        拟合 SST 与各深度层温度的统计回归参数
        
        建立线性关系：T(z) = a(z) * SST + b(z)
        
        Args:
            sst_data: [n_samples] 海表温度样本
            profile_data: [n_samples, n_depths] 温度剖面样本
        
        Returns:
            dict: 拟合参数和统计信息
        """
        sst_flat = sst_data.flatten()
        n_samples = len(sst_flat)
        
        slopes = np.zeros(self.n_depths)
        intercepts = np.zeros(self.n_depths)
        correlations = np.zeros(self.n_depths)
        r_squared = np.zeros(self.n_depths)
        
        for d in range(self.n_depths):
            # 获取该深度的所有温度值
            if len(profile_data.shape) == 2:
                temp_at_depth = profile_data[:, d]
            else:
                # [n_samples, height, width, depth] -> flatten
                temp_at_depth = profile_data[:, :, :, d].flatten()
            
            # 移除 NaN
            valid_mask = ~(np.isnan(sst_flat) | np.isnan(temp_at_depth))
            
            if valid_mask.sum() < 10:  # 样本太少
                slopes[d] = 1.0 if d == 0 else slopes[d-1] * 0.95
                intercepts[d] = 0.0
                correlations[d] = 0.0
                continue
            
            sst_valid = sst_flat[valid_mask]
            temp_valid = temp_at_depth[valid_mask]
            
            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(sst_valid, temp_valid)
            
            slopes[d] = slope
            intercepts[d] = intercept
            correlations[d] = r_value
            r_squared[d] = r_value ** 2
        
        # 更新模型参数
        self.regression_slopes = torch.tensor(slopes, dtype=torch.float32)
        self.regression_intercepts = torch.tensor(intercepts, dtype=torch.float32)
        self.correlation_coeffs = torch.tensor(correlations, dtype=torch.float32)
        
        self.is_fitted = True
        
        return {
            'slopes': slopes,
            'intercepts': intercepts,
            'correlations': correlations,
            'r_squared': r_squared,
            'depths': self.target_depths
        }
    
    def forward(self, sst, mld=None, thermocline_params=None):
        """
        前向传播：从 SST 反演三维温度场
        
        反演策略：
        1. 混合层内（z < MLD）：T(z) ≈ SST（均匀混合）
        2. 温跃层内（MLD < z < Zb）：T(z) = SST - gradient * (z - MLD)
        3. 温跃层下（z > Zb）：T(z) = a(z) * SST + b(z)（统计回归）
        
        Args:
            sst: [batch, height, width] 海表温度
            mld: [batch, height, width] 混合层深度（可选，默认使用气候态）
            thermocline_params: dict, 温跃层参数（可选）
        
        Returns:
            profile: [batch, height, width, depth] 三维温度场
        """
        # 处理输入
        if isinstance(sst, np.ndarray):
            sst = torch.from_numpy(sst).float()
        
        if len(sst.shape) == 2:
            sst = sst.unsqueeze(0)
        
        batch, height, width = sst.shape
        device = sst.device
        
        # 初始化输出
        profile = torch.zeros(batch, height, width, self.n_depths, device=device)
        
        # 获取 NaN mask（陆地区域）
        nan_mask = torch.isnan(sst)
        sst_clean = torch.nan_to_num(sst, nan=0.0)
        
        # 确保回归参数在正确设备上
        slopes = self.regression_slopes.to(device)
        intercepts = self.regression_intercepts.to(device)
        
        if not self.is_fitted:
            # 未拟合时使用简单的线性衰减模型
            for d in range(self.n_depths):
                depth = self.target_depths[d]
                # 简单指数衰减：T(z) = SST * exp(-z/scale)
                decay_factor = np.exp(-depth / 500)  # 500m 衰减尺度
                profile[:, :, :, d] = sst_clean * decay_factor
        else:
            # 使用拟合的统计关系
            for d in range(self.n_depths):
                profile[:, :, :, d] = slopes[d] * sst_clean + intercepts[d]
        
        # 如果提供了 MLD，应用物理约束
        if mld is not None:
            if isinstance(mld, np.ndarray):
                mld = torch.from_numpy(mld).float().to(device)
            
            if len(mld.shape) == 2:
                mld = mld.unsqueeze(0)
            
            # 混合层内温度均匀 = SST
            for d in range(self.n_depths):
                depth = self.target_depths[d]
                # 深度小于 MLD 的位置，温度等于 SST
                in_mixed_layer = torch.tensor(depth, device=device) < mld
                profile[:, :, :, d] = torch.where(
                    in_mixed_layer,
                    sst_clean,
                    profile[:, :, :, d]
                )
        
        # 应用梯度依赖修正（如果启用）
        if self.use_gradient_dependent and self.is_fitted:
            profile = self._apply_gradient_dependent_correction(profile, sst_clean)
        
        # 恢复 NaN mask
        for d in range(self.n_depths):
            profile[:, :, :, d] = torch.where(
                nan_mask,
                torch.tensor(float('nan'), device=device),
                profile[:, :, :, d]
            )
        
        return profile
    
    def _apply_gradient_dependent_correction(self, profile, sst):
        """
        梯度依赖修正：根据局部 SST 梯度调整次表层温度反演
        
        原理：SST 空间梯度大的区域（如锋面），次表层结构更复杂
        
        Args:
            profile: [batch, height, width, depth] 初始反演结果
            sst: [batch, height, width] 海表温度
        
        Returns:
            corrected_profile: 修正后的温度场
        """
        batch, height, width, n_depths = profile.shape
        
        # 计算 SST 空间梯度
        # 使用 Sobel 算子近似
        sst_padded = F.pad(sst.unsqueeze(1), (1, 1, 1, 1), mode='replicate')
        
        # 梯度核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=sst.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=sst.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(sst_padded, sobel_x)
        grad_y = F.conv2d(sst_padded, sobel_y)
        
        # 梯度幅值
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze(1)
        
        # 归一化梯度（0-1）
        grad_min = gradient_magnitude.min()
        grad_max = gradient_magnitude.max()
        if grad_max > grad_min:
            gradient_norm = (gradient_magnitude - grad_min) / (grad_max - grad_min)
        else:
            gradient_norm = torch.zeros_like(gradient_magnitude)
        
        # 根据梯度调整深层温度的相关性权重
        # 高梯度区域，统计关系权重降低
        for d in range(n_depths):
            depth = self.target_depths[d]
            
            if depth > 100:  # 仅对深层应用修正
                # 修正因子：梯度越大，修正越强
                correction_weight = 0.1 * gradient_norm * (1 - self.correlation_coeffs[d].abs().item())
                
                # 向气候态方向修正（这里简化为向 SST 相关值方向）
                profile[:, :, :, d] = profile[:, :, :, d] * (1 - correction_weight)
        
        return profile
    
    def custom_mse_loss(self, y_pred, y_true):
        """
        处理 NaN 值的 MSE 损失函数
        """
        # 确保维度匹配
        if y_pred.shape != y_true.shape:
            if y_true.shape[-1] == y_pred.shape[-1]:
                pass  # 维度已匹配
            elif len(y_true.shape) == 4 and y_true.shape[1] == y_pred.shape[-1]:
                y_true = y_true.permute(0, 2, 3, 1)
        
        # 创建有效值掩码
        valid_mask = ~torch.isnan(y_true)
        
        if valid_mask.sum() > 0:
            loss = F.mse_loss(y_pred[valid_mask], y_true[valid_mask])
            return loss
        else:
            return y_pred.sum() * 0.0
    
    def training_step(self, batch, batch_idx):
        """
        训练步骤
        
        注意：此模型主要通过 fit_regression_parameters 拟合
        training_step 用于在线微调统计参数
        """
        x, y = batch  # x: SST, y: 温度剖面
        
        y_pred = self(x)
        loss = self.custom_mse_loss(y_pred, y)
        
        if torch.isnan(loss) or torch.isinf(loss):
            return y_pred.sum() * 0.0
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_loss.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        y_pred = self(x)
        val_loss = self.custom_mse_loss(y_pred, y)
        
        # 计算 RMSE
        with torch.no_grad():
            valid_mask = ~torch.isnan(y)
            if valid_mask.sum() > 0:
                rmse = torch.sqrt(F.mse_loss(y_pred[valid_mask], y[valid_mask]))
                self.log('val_rmse', rmse, prog_bar=True)
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.val_loss.append(val_loss.item())
        
        return val_loss
    
    def configure_optimizers(self):
        """
        配置优化器
        
        注意：此模型主要使用统计拟合，优化器用于可选的参数微调
        """
        # 创建可训练的回归参数
        self.trainable_slopes = torch.nn.Parameter(self.regression_slopes.clone())
        self.trainable_intercepts = torch.nn.Parameter(self.regression_intercepts.clone())
        
        optimizer = optim.Adam(
            [self.trainable_slopes, self.trainable_intercepts],
            lr=self.learning_rate
        )
        
        return optimizer
    
    def predict(self, sst, mld=None):
        """
        预测函数
        
        Args:
            sst: [batch, height, width] 或 [height, width] 海表温度
            mld: [batch, height, width] 混合层深度（可选）
        
        Returns:
            profile: numpy array [batch, height, width, depth]
        """
        self.eval()
        
        with torch.no_grad():
            if isinstance(sst, np.ndarray):
                sst_tensor = torch.from_numpy(sst).float()
            else:
                sst_tensor = sst
            
            if len(sst_tensor.shape) == 2:
                sst_tensor = sst_tensor.unsqueeze(0)
            
            # 移到模型设备
            sst_tensor = sst_tensor.to(self.device)
            
            if mld is not None:
                if isinstance(mld, np.ndarray):
                    mld = torch.from_numpy(mld).float().to(self.device)
            
            output = self(sst_tensor, mld)
            
            return output.cpu().numpy()
    
    def fit(self, sst_data, profile_data, compute_climatology=True):
        """
        完整拟合流程
        
        Args:
            sst_data: [n_samples, height, width] 海表温度数据
            profile_data: [n_samples, height, width, depth] 温度剖面数据
            compute_climatology: 是否计算气候态参数
        
        Returns:
            fit_results: 拟合结果字典
        """
        # 1. 拟合回归参数
        fit_results = self.fit_regression_parameters(
            sst_data.flatten(),
            profile_data.reshape(-1, self.n_depths)
        )
        
        # 2. 计算气候态温跃层参数（可选）
        if compute_climatology:
            n_samples = profile_data.shape[0]
            mld_values = []
            gradient_values = []
            bottom_values = []
            
            # 对每个样本点计算温跃层参数
            flat_profiles = profile_data.reshape(-1, self.n_depths)
            
            for i in range(min(1000, len(flat_profiles))):  # 抽样计算
                if np.all(np.isnan(flat_profiles[i])):
                    continue
                
                params = self.compute_thermocline_parameters(flat_profiles[i])
                
                if not np.isnan(params['mld']):
                    mld_values.append(params['mld'])
                if not np.isnan(params['thermocline_gradient']):
                    gradient_values.append(params['thermocline_gradient'])
                if not np.isnan(params['thermocline_bottom']):
                    bottom_values.append(params['thermocline_bottom'])
            
            fit_results['climatology'] = {
                'mean_mld': np.mean(mld_values) if mld_values else np.nan,
                'std_mld': np.std(mld_values) if mld_values else np.nan,
                'mean_gradient': np.mean(gradient_values) if gradient_values else np.nan,
                'mean_bottom': np.mean(bottom_values) if bottom_values else np.nan
            }
        
        return fit_results
    
    def reconstruct_profile_physical(self, sst, mld, thermocline_gradient, thermocline_bottom):
        """
        基于物理约束的温度剖面重建
        
        使用分段模型：
        1. 混合层（0 ~ MLD）：T = SST
        2. 温跃层（MLD ~ Zb）：T = SST - gradient * (z - MLD)
        3. 深层（> Zb）：T = T(Zb) - deep_gradient * (z - Zb)
        
        Args:
            sst: 海表温度
            mld: 混合层深度
            thermocline_gradient: 温跃层梯度（°C/m）
            thermocline_bottom: 温跃层下边界深度
        
        Returns:
            profile: [n_depths] 重建的温度剖面
        """
        profile = np.zeros(self.n_depths)
        
        # 温跃层底部温度
        t_at_mld = sst
        t_at_bottom = sst - thermocline_gradient * (thermocline_bottom - mld)
        
        # 深层衰减梯度（经验值）
        deep_gradient = 0.005  # °C/m
        
        for i, z in enumerate(self.target_depths):
            if z <= mld:
                # 混合层：均匀温度
                profile[i] = sst
            elif z <= thermocline_bottom:
                # 温跃层：线性递减
                profile[i] = sst - thermocline_gradient * (z - mld)
            else:
                # 深层：缓慢递减
                profile[i] = t_at_bottom - deep_gradient * (z - thermocline_bottom)
        
        return profile


class GradientDependentOI(Thermocline):
    """
    梯度依赖最优插值（Gradient-Dependent Optimal Interpolation）
    
    在 ThermoclineModel 基础上增加最优插值校正
    用于生成高精度的网格化温度场
    """
    
    def __init__(self, 
                 correlation_scale=300,  # 相关尺度（km）
                 **kwargs):
        super().__init__(**kwargs)
        
        self.correlation_scale = correlation_scale
    
    def optimal_interpolation(self, background, observations, obs_locations, 
                               obs_errors, background_errors):
        """
        最优插值算法
        
        Args:
            background: [height, width, depth] 背景场（来自统计反演）
            observations: [n_obs, depth] 观测值（Argo 剖面）
            obs_locations: [n_obs, 2] 观测点位置 (lat, lon)
            obs_errors: [n_obs] 观测误差方差
            background_errors: [height, width] 背景场误差方差
        
        Returns:
            analysis: [height, width, depth] 分析场
        """
        height, width, n_depths = background.shape
        analysis = background.copy()
        
        # 对每个网格点进行插值
        for i in range(height):
            for j in range(width):
                if np.all(np.isnan(background[i, j, :])):
                    continue
                
                grid_lat = i  # 简化，实际应转换为真实坐标
                grid_lon = j
                
                # 计算与观测点的距离和权重
                weights = []
                for k, (obs_lat, obs_lon) in enumerate(obs_locations):
                    distance = np.sqrt((grid_lat - obs_lat)**2 + (grid_lon - obs_lon)**2)
                    
                    # 高斯相关函数
                    correlation = np.exp(-(distance / self.correlation_scale)**2)
                    
                    # 梯度依赖修正：背景场梯度大的区域，相关尺度减小
                    if i > 0 and i < height-1 and j > 0 and j < width-1:
                        local_gradient = np.nanmean(np.abs(np.gradient(background[i-1:i+2, j-1:j+2, 0])))
                        gradient_factor = 1 / (1 + local_gradient)
                        correlation *= gradient_factor
                    
                    weights.append(correlation)
                
                weights = np.array(weights)
                
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    
                    # 对每个深度层应用插值
                    for d in range(n_depths):
                        obs_values = observations[:, d]
                        valid_obs = ~np.isnan(obs_values)
                        
                        if valid_obs.sum() > 0:
                            innovation = obs_values[valid_obs] - background[i, j, d]
                            correction = np.sum(weights[valid_obs] * innovation)
                            analysis[i, j, d] = background[i, j, d] + correction
        
        return analysis

