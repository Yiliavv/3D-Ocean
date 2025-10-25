# 全球海洋温度预测模型

本项目旨在使用深度学习模型和遥感数据构建高精度的全球海洋温度预测系统，包括海表温度时空序列预测和三维海洋温度场重建。

## 项目概述

本项目分为两个主要研究方向：

1. **全球海表温度时空序列预测模型**
   - 基于历年海表温度数据（月平均、日平均）预测未来的海表温度
   - 使用多种深度学习架构：LSTM、ConvLSTM、UNetLSTM、Transformer、RG-SA Transformer

2. **全球海洋三维温度场空间模型**
   - 构建海洋三维温度分布的空间重建模型（不包含时间维度）
   - 以海表温度、海表温度异常等参数构建深度温度场
   - 使用随机森林回归器进行三维温度场预测

## 环境要求

### Python 环境
- `python` >= 3.13.6
- `pytorch` >= 2.8.0
- `lightning` >= 2.5.2 (PyTorch Lightning框架)
- `numpy`
- `pandas`
- `scipy`

### 数据处理和可视化
- `netCDF4` >= 1.7.2 (处理NetCDF格式的气象数据)
- `matplotlib` >= 3.10.5
- `cartopy` (地理空间数据可视化)
- `cmocean` (海洋数据专用颜色映射)
- `seaborn`

### 机器学习
- `scikit-learn` (随机森林模型)
- `statsmodels` (时间序列分析)

### 其他依赖
- `arrow` (时间处理)

## 项目结构

```
tensorflow/
├── src/                          # 源代码目录
│   ├── models/                   # 模型定义
│   │   ├── LSTM.py              # 基础LSTM模型
│   │   ├── ConvLSTM.py          # 卷积LSTM模型
│   │   ├── UNetLSTM.py          # UNet-LSTM混合模型
│   │   ├── Transformer.py       # 标准Transformer模型
│   │   ├── RATransformer.py     # 递归注意力Transformer (RG-SA Transformer)
│   │   ├── RDF.py               # 随机决策森林模型
│   │   ├── Attention/           # 注意力机制模块
│   │   │   └── RGAttention.py   # 递归注意力层
│   │   └── PE/                  # 位置编码模块
│   │       └── SphericalHarmonicEncoding.py  # 球谐波位置编码
│   │
│   ├── dataset/                  # 数据集处理
│   │   ├── ERA5.py              # ERA5海表温度数据集
│   │   └── Argo.py              # Argo三维温度数据集
│   │
│   ├── trainer/                  # 训练器
│   │   └── base.py              # 基础训练器类
│   │
│   ├── analysis/                 # 数据分析工具
│   │   ├── season.py            # 季节性分析
│   │   ├── rmse.py              # RMSE评估
│   │   ├── prediction.py        # 预测分析
│   │   └── SH.py                # 球谐波分析
│   │
│   ├── plot/                     # 可视化工具
│   │   ├── base.py              # 基础绘图函数
│   │   ├── sst.py               # 海表温度可视化
│   │   ├── profile.py           # 温度剖面可视化
│   │   ├── mask.py              # 掩码可视化
│   │   └── mld.py               # 混合层深度可视化
│   │
│   ├── config/                   # 配置文件
│   │   ├── params.py            # 全局参数配置
│   │   ├── area.py              # 区域定义类
│   │   ├── constants.py         # 常量定义
│   │   └── ne_10m_bathymetry_all/  # 海底地形数据
│   │
│   ├── utils/                    # 工具函数
│   │   ├── util.py              # 通用工具函数
│   │   ├── mio.py               # 模型输入输出
│   │   ├── log.py               # 日志工具
│   │   ├── mld.py               # 混合层深度计算
│   │   └── plot.py              # 绘图工具
│   │
│   ├── main.ipynb                # 主要训练和评估笔记本
│   ├── rf.ipynb                  # 随机森林模型笔记本
│   └── EI-NA.ipynb               # 额外分析笔记本
│
├── out/                          # 输出目录
│   ├── models/                   # 训练好的模型
│   ├── sst/                      # 海表温度预测结果
│   ├── error/                    # 误差分析结果
│   ├── csv/                      # CSV格式数据
│   └── profile/                  # 温度剖面结果
│
├── train_output/                 # 训练输出日志
│   ├── LSTM/
│   ├── ConvLSTM/
│   ├── UNetLSTM/
│   ├── Transformer/
│   ├── RATransformer/
│   └── rf/
│
├── doc/                          # 文档和论文
│   ├── RG-SA Transformer.docx   # 主要论文文档
│   ├── IEEE/                     # IEEE论文投稿版本
│   ├── ERSGIT/                   # 会议论文版本
│   └── 组会/                     # 组会报告
│
└── README.md                     # 项目说明文档

```

## 核心组件说明

### 1. 模型架构

#### 1.1 时空序列预测模型

**LSTM (src/models/LSTM.py)**
- 纯粹的LSTM网络，适用于时间序列建模
- 支持多层LSTM堆叠和Dropout正则化
- 自动处理NaN值的损失函数

**ConvLSTM (src/models/ConvLSTM.py)**
- 结合卷积神经网络和LSTM的混合架构
- 能够捕捉空间和时间相关性
- 使用卷积核处理空间特征

**UNetLSTM (src/models/UNetLSTM.py)**
- 基于UNet编码-解码结构的LSTM模型
- 适合保留空间细节信息
- 多尺度特征提取

**Transformer (src/models/Transformer.py)**
- 标准的Transformer编码器-解码器架构
- 使用多头自注意力机制
- 支持长距离时间依赖建模

**RG-SA Transformer (src/models/RATransformer.py)** ⭐ 核心创新模型
- **递归注意力机制**：通过递归计算增强注意力表达能力
- **球谐波位置编码**：专为地理坐标设计的位置编码网络
- **空间感知编码**：考虑地理空间特性的编码方式
- **EMA训练稳定性**：使用指数移动平均提高训练稳定性
- 特别适用于全球尺度的海洋温度预测

#### 1.2 三维温度场重建模型

**RandomForest (src/models/RDF.py)**
- 基于sklearn的随机森林回归器
- 从海表温度预测深层海洋温度
- 支持PyTorch Lightning训练框架集成
- 并行计算，高效处理大规模数据

### 2. 数据集处理

**ERA5 数据集 (src/dataset/ERA5.py)**
- **ERA5SSTDataset**: 日平均海表温度数据集
  - 时间范围：2004-01-01 至 2024-12-31
  - 空间分辨率：0.25° × 0.25°
  - 支持自定义经纬度范围和时间窗口
  
- **ERA5SSTMonthlyDataset**: 月平均海表温度数据集
  - 同样的时间和空间覆盖
  - 支持序列长度自定义
  - 内置海表温度异常(SSTA)计算

**Argo 数据集 (src/dataset/Argo.py)**
- **Argo3DTemperatureDataset**: 三维海洋温度数据
  - 深度层级：58层（0-1975米）
  - 提供海表温度和深层温度剖面
  - 自动处理缺失值和数据归一化
  
- **ArgoStackTemperatureDataset**: 用于随机森林的堆叠数据格式

### 3. 训练框架

**BaseTrainer (src/trainer/base.py)**
- 统一的训练接口，支持所有模型
- 自动数据集分割（训练/验证）
- 模型保存和加载
- 预测和评估功能
- 训练日志自动记录

主要功能：
```python
trainer = BaseTrainer(
    title='模型名称',
    uid='唯一标识符',
    area=研究区域,
    model_class=模型类,
    dataset_class=数据集类,
    save_path='模型保存路径',
    dataset_params={...},
    trainer_params={...},
    model_params={...}
)

# 训练模型
model = trainer.train()

# 预测
input, output, pred_output, rmse, r2, ssta = trainer.predict(offset=月份偏移)
```

### 4. 分析工具

**季节性分析 (src/analysis/season.py)**
- 时间序列周期性检测
- 季节性分解（STL方法）
- ANOVA显著性检验
- 可视化季节性模式

**误差评估 (src/analysis/rmse.py)**
- RMSE（均方根误差）
- MAE（平均绝对误差）
- R²（决定系数）
- 误差空间分布分析

### 5. 可视化工具

**海表温度可视化 (src/plot/sst.py)**
- 海表温度分布图（使用Cartopy地理投影）
- 预测误差分布图
- NINO指数区域可视化
- 时间序列对比图
- 核密度估计图

**温度剖面可视化 (src/plot/profile.py)**
- 三维温度场断面图
- 深度-温度剖面图
- 多时间点对比

### 6. 配置管理

**全局参数 (src/config/params.py)**
- 数据路径配置
- 模型保存路径
- 研究区域定义
- 示例区域：北太平洋、南太平洋、印度洋、北大西洋

**区域类 (src/config/area.py)**
- 灵活的地理区域定义
- 自动计算区域范围
- 可视化区域位置
- 支持海底地形叠加

## 使用指南

### 1. 数据准备

需要准备以下数据：
- **ERA5**: 月平均/日平均海表温度NetCDF文件
  - 放置在配置的 `BASE_ERA5_MONTHLY_DATA_PATH` 或 `BASE_ERA5_DAILY_DATA_PATH`
  
- **Argo**: BOA_Argo格式的三维温度数据
  - 放置在配置的 `BASE_BOA_ARGO_DATA_PATH`

### 2. 训练海表温度预测模型

参考 `src/main.ipynb`:

```python
# 1. 定义研究区域
area = Area('Global', lon=[-180, 180], lat=[-80, 80], description='全球区域')

# 2. 配置数据集参数
dataset_params = {
    "seq_len": 2,        # 序列长度
    "offset": 0,         # 时间偏移
    "resolution": 2,     # 空间分辨率（度）
}

# 3. 配置模型参数（以RATransformer为例）
model_params = {
    "width": 90,              # 纬度网格点数
    "height": 180,            # 经度网格点数
    "seq_len": 2,
    "d_model": 512,           # 模型维度
    "num_heads": 8,           # 注意力头数
    "num_layers": 3,          # Transformer层数
    "recursion_depth": 2,     # 递归深度
    "learning_rate": 1e-3,
}

# 4. 配置训练参数
trainer_params = {
    "epochs": 500,
    "batch_size": 50,
}

# 5. 创建训练器
trainer = BaseTrainer(
    title='RATransformer',
    area=area,
    uid=str(uuid.uuid4()),
    model_class=RecursiveAttentionTransformer,
    dataset_class=ERA5SSTMonthlyDataset,
    save_path='./out/models/ra_transformer.pkl',
    dataset_params=dataset_params,
    trainer_params=trainer_params,
    model_params=model_params,
)

# 6. 训练模型
trainer.train()

# 7. 预测和评估
input, output, pred, rmse, r2, ssta = trainer.predict(offset=255, plot=True)
```

### 3. 训练三维温度场重建模型

参考 `src/rf.ipynb`:

```python
# 1. 配置参数
depth = [0, 10]  # 深度范围（层索引）

dataset_params = {
    "depth": depth,
    "resolution": 2
}

model_params = {
    "n_estimators": 100,     # 决策树数量
    "random_state": 42,
    "n_jobs": 10,            # 并行作业数
}

# 2. 创建训练器
rf_trainer = BaseTrainer(
    uid="rf",
    title="rf",
    area=area,
    model_class=RDFNetwork,
    dataset_class=Argo3DTemperatureDataset,
    save_path="./out/models/rf_global.pkl",
    dataset_params=dataset_params,
    trainer_params=trainer_params,
    model_params=model_params
)

# 3. 训练模型
model = rf_trainer.train()

# 4. 预测三维温度场
sst_model = load('./out/models/seq_len-2/conv.pkl')  # 加载SST预测模型
sst, temp = next(iter(loader))
pred_sst = sst_model(sst)
profile_3d = model.predict(pred_sst)  # 预测深层温度
```

### 4. 数据分析

**季节性分析：**
```python
from src.analysis.season import SeasonalityAnalysis

dataset = ERA5SSTMonthlyDataset(seq_len=1, offset=0, lon=[-180, 180], lat=[-80, 80])
analyzer = SeasonalityAnalysis(dataset)

# 统计检验
results = analyzer.test_seasonality()
# 可视化
fig1, fig2 = analyzer.plot_seasonal_patterns()
```

**模型性能对比：**
```python
from src.analysis.rmse import plot_metrics

# 绘制所有模型的损失曲线和RMSE对比
plot_metrics()
```

## 主要特性

### 创新点

1. **递归注意力机制 (RG-SA Transformer)**
   - 通过递归计算增强注意力表达能力
   - 在保持计算效率的同时提升模型性能

2. **球谐波位置编码**
   - 专为球面（地球）坐标系统设计
   - 比传统正弦位置编码更适合地理数据

3. **两阶段建模策略**
   - 第一阶段：时空序列预测（海表温度）
   -第二阶段：空间重建（三维温度场）
   - 充分利用不同模型的优势

4. **多模型对比框架**
   - 统一的训练和评估接口
   - 便于对比不同模型的性能

### 技术优势

- **GPU加速训练**：基于PyTorch Lightning，自动支持GPU
- **NaN值处理**：自动处理海洋数据中的陆地区域（NaN值）
- **模块化设计**：各组件解耦，易于扩展和修改
- **完整的可视化**：地理投影、误差分析、统计图表
- **实验可重现**：随机种子固定、完整的日志记录

## 实验结果

模型在全球海域的预测性能：
- **LSTM**: RMSE ≈ 0.5°C
- **ConvLSTM**: RMSE ≈ 0.43°C
- **UNetLSTM**: RMSE ≈ 0.48°C
- **Transformer**: RMSE ≈ 0.45°C
- **RG-SA Transformer**: RMSE ≈ 0.40°C ⭐

三维温度场重建：
- **RandomForest**: 深度0-200米，RMSE ≈ 2.2°C

## 相关论文

本项目支持以下研究论文的实验：
- 《基于RG-SA Transformer与RF的全球海洋三维温度场预测》
- 《Global Ocean Three-Dimensional Temperature Field Prediction Based on RG-SA Transformer and RF》

论文文档位于 `doc/` 目录。

## 许可证

本项目用于学术研究，JGR期刊投稿相关。

## 联系方式

如有问题，请通过项目仓库提交Issue。

---

**更新日期**: 2025年10月

**项目状态**: 活跃开发中
