from dash import Dash, html, dcc
import dash_mantine_components as dmc
from dash_iconify import DashIconify

from src.config.params import Areas

def InputFactory(label, tooltip, type, **kwargs):
    uni_width = 240
    
    if type == "number":
        return dmc.NumberInput(
            label=dmc.Center(
                [
                    dmc.Text(label, size="md", h=30, mr=6),
                    dmc.Tooltip(
                        label=tooltip,
                        children=DashIconify(
                            icon="mdi:information-outline",
                            width=16,
                        ),
                    ),
                ],
            ),
            w=uni_width,
            **kwargs
        )
    elif type == "range":
        return dmc.Stack(
            [
                dmc.Group(
                    [
                        dmc.Text(label, size="md", h=30,),
                        dmc.Tooltip(
                            label=tooltip,
                            children=DashIconify(
                                icon="mdi:information-outline",
                                width=16,
                            ),
                        ),
                    ],
                    gap=6,
                ),
                dmc.RangeSlider(
                    w=uni_width,
                    **kwargs
                ),
            ]
        )
    elif type == "select":
        return dmc.Select(
            label=dmc.Center(
                [
                    dmc.Text(label, size="md", h=30, mr=6),
                    dmc.Tooltip(
                        label=tooltip,
                        children=DashIconify(
                            icon="mdi:information-outline",
                            width=16,
                        ),
                    ),
                ],
            ),
            w=uni_width,
            **kwargs
        )
    else:
        raise ValueError(f"Invalid input type: {type}")


## 数据集参数
def SeqLenInput():
    return InputFactory(
        "Sequence Length",
        "时间序列长度，输入为 n - 1, 输出为 n",
        "number",
        id="seq-len-input",
        min=2,
        value=2,
    )
    
def OffsetInput():
    return InputFactory(
        "Offset",
        "偏移量, 时间序列的开始位置",
        "number",
        id="offset-input",
        min=0,
        value=0,
    )
    
def AreaSelect():
    
    area_data = [
        {
            "value": str(i),
            "label": area.description,
        }
        for i, area in enumerate(Areas)
    ]
    
    return InputFactory(
        "Area",
        "快速选择区域",
        "select",
        id="area-select",
        data=area_data,
    )
    
def LongitudeInput():
    return InputFactory(
        "Longitude",
        "经度范围, 例如: -110.5 : 110.5",
        "range",
        id="longitude-input",
        min=-180,
        max=180,
        value=[-180, 180],
        marks=[
            {
                "value": -90,
                "label": "90°W",
            },
            {
                "value": 0,
                "label": "0°",
            },
            {
                "value": 90,
                "label": "90°E",
            },
        ],
        mb=20,
    )

def LatitudeInput():
    return InputFactory(
        "Latitude",
        "纬度范围, 例如: 30.5 : 40.5",
        "range",
        id="latitude-input",
        min=-90,
        max=90,
        value=[-90, 90],
        marks=[
            {
                "value": -45,
                "label": "45°S",
            },
            {
                "value": 0,
                "label": "0°",
            },
            {
                "value": 45,
                "label": "45°N",
            },
        ],
        mb=20,
    )
    
def ResolutionInput():
    return InputFactory(
        "Resolution",
        "经纬度分辨率",
        "number",
        id="resolution-input",
        max=10,
        suffix= " °",
        value=0.25,
    )
    

## 模型参数

def NHeadInput():
    return InputFactory(
        "NHead",
        "多头注意力头数",
        "number",
        id="n-head-input",
        min=1,
        value=4,
    )

def NumEncoderLayersInput():
    return InputFactory(
        "NumEncoderLayers",
        "编码器层数",
        "number",
        id="num-encoder-layers-input",
        min=1,
        value=2,
    )

def NumDecoderLayersInput():
    return InputFactory(
        "NumDecoderLayers",
        "解码器层数",
        "number",
        id="num-decoder-layers-input",
        min=1,
        value=2,
    )

def LearningRateInput():
    return InputFactory(
        "LearningRate",
        "初始学习率",
        "number",
        id="learning-rate-input",
        suffix=" * 10e-4",
        min=0,
        value=1,
    )
    
def DropoutInput():
    return InputFactory(
        "Dropout",
        "数据 dropout 比例",
        "number",
        id="dropout-input",
        min=0.05,
        max=1,
        value=0.,
    )
    
def OptimizerSelect():
    return InputFactory(
        "Optimizer",
        "选择优化器",
        "select",
        id="optimizer-select",
        data=["Adam", "SGD"],
        value="Adam",
    )


## 训练参数 

def BatchSizeInput():
    return InputFactory(
        "BatchSize",
        "批量大小",
        "number",
        id="batch-size-input",
        min=1,
        value=15,
    )

def EpochsInput():
    return InputFactory(
        "Epochs",
        "训练轮数",
        "number",
        id="epochs-input",
        min=1,
        value=100,
    )

