# 模型示意图
from dash import html, dcc
import plotly.graph_objects as go
import dash_mantine_components as dmc
import numpy as np

def create_data_blocks(fig, x0, y0, width, height, num_blocks, direction="horizontal", color="lightblue", text=None):
    """
    创建数据块序列
    
    参数:
    - fig: plotly图形对象
    - x0, y0: 起始坐标
    - width, height: 整体宽度和高度
    - num_blocks: 块的数量
    - direction: 方向，"horizontal" 或 "vertical"
    - color: 填充颜色
    - text: 可选的文本标注
    """
    if direction == "horizontal":
        block_width = width / num_blocks
        block_height = height
        for i in range(num_blocks):
            block_x0 = x0 + i * block_width
            # 添加数据块底色
            fig.add_shape(
                type="rect",
                x0=block_x0,
                y0=y0,
                x1=block_x0 + block_width * 0.9,  # 留出一点间隔
                y1=y0 + block_height,
                line=dict(color="rgba(0,0,0,0.1)", width=1),
                fillcolor=color,
            )
            # 添加上部装饰条
            fig.add_shape(
                type="rect",
                x0=block_x0,
                y0=y0 + block_height * 0.9,
                x1=block_x0 + block_width * 0.9,
                y1=y0 + block_height,
                line=dict(color="rgba(0,0,0,0)", width=0),
                fillcolor="rgba(255,255,255,0.3)",
            )
    else:  # vertical
        block_width = width
        block_height = height / num_blocks
        for i in range(num_blocks):
            block_y0 = y0 + i * block_height
            fig.add_shape(
                type="rect",
                x0=x0,
                y0=block_y0,
                x1=x0 + block_width,
                y1=block_y0 + block_height * 0.9,
                line=dict(color="rgba(0,0,0,0.1)", width=1),
                fillcolor=color,
            )
    
    if text:
        # 添加半透明背景框
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0 + height/2 - 0.15,
            x1=x0 + width,
            y1=y0 + height/2 + 0.15,
            line=dict(color="rgba(0,0,0,0)", width=0),
            fillcolor="rgba(255,255,255,0.7)",
        )
        # 添加文本
        fig.add_annotation(
            x=x0 + width/2,
            y=y0 + height/2,
            text=text,
            showarrow=False,
            font=dict(size=10, color="rgba(0,0,0,0.7)"),
        )

def create_model_diagram(n_head, num_encoder_layers, num_decoder_layers, seq_len):
    """
    根据模型参数创建模型结构示意图
    
    参数:
    - n_head: 注意力头数
    - num_encoder_layers: 编码器层数
    - num_decoder_layers: 解码器层数
    - seq_len: 序列长度
    """
    # 创建图形
    fig = go.Figure()
    
    # 设置图形布局
    fig.update_layout(
        title={
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        height=500,
        width=500,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis={
            'showgrid': False,
            'zeroline': False,
            'visible': False,
            'range': [-1, 7]  # 调整x轴范围
        },
        yaxis={
            'showgrid': False,
            'zeroline': False,
            'visible': False,
            'range': [-1, 4]  # 调整y轴范围
        }
    )
    
    # 添加输入层
    input_x0, input_y0 = 0.2, 0.5
    input_width, input_height = 1.6, 0.8
    
    # 输入层外框
    fig.add_shape(
        type="rect",
        x0=input_x0, y0=input_y0,
        x1=input_x0 + input_width, y1=input_y0 + input_height,
        line=dict(color="rgba(0,0,0,0.2)", width=2),
        fillcolor="white",
    )
    
    # 添加输入数据块
    create_data_blocks(
        fig,
        input_x0 + 0.1, input_y0 + 0.1,
        input_width - 0.2, input_height - 0.2,
        seq_len - 1,  # 输入序列长度
        "horizontal",
        "rgba(135,206,250,0.5)",  # 使用半透明的浅蓝色
        f"输入序列 (t=1~{seq_len-1})"
    )
    
    # 添加编码器层
    encoder_width = 1
    encoder_height = 0.8
    encoder_spacing = 0.3
    
    for i in range(num_encoder_layers):
        x0 = 2.2 + i * (encoder_width + encoder_spacing)
        y0 = 0.5
        x1 = x0 + encoder_width
        y1 = y0 + encoder_height
        
        # 添加阴影效果
        fig.add_shape(
            type="rect",
            x0=x0 + 0.05, y0=y0 + 0.05,
            x1=x1 + 0.05, y1=y1 + 0.05,
            line=dict(color="rgba(0,0,0,0)", width=0),
            fillcolor="rgba(0,0,0,0.1)",
        )
        
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="rgba(0,0,0,0.2)", width=2),
            fillcolor="rgba(144,238,144,0.5)",  # 使用半透明的浅绿色
        )
        
        # 添加多头注意力标注
        if i == 0:
            fig.add_annotation(
                x=x0 + encoder_width/2, y=y0 + encoder_height/2,
                text=f"编码器层 {i+1}<br>({n_head} 头注意力)",
                showarrow=False,
                font=dict(size=10, color="rgba(0,0,0,0.7)"),
            )
        else:
            fig.add_annotation(
                x=x0 + encoder_width/2, y=y0 + encoder_height/2,
                text=f"编码器层 {i+1}",
                showarrow=False,
                font=dict(size=10, color="rgba(0,0,0,0.7)"),
            )
    
    # 添加解码器层
    decoder_width = 1
    decoder_height = 0.8
    decoder_spacing = 0.3
    
    for i in range(num_decoder_layers):
        x0 = 2.2 + i * (decoder_width + decoder_spacing)
        y0 = 2
        x1 = x0 + decoder_width
        y1 = y0 + decoder_height
        
        # 添加阴影效果
        fig.add_shape(
            type="rect",
            x0=x0 + 0.05, y0=y0 + 0.05,
            x1=x1 + 0.05, y1=y1 + 0.05,
            line=dict(color="rgba(0,0,0,0)", width=0),
            fillcolor="rgba(0,0,0,0.1)",
        )
        
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="rgba(0,0,0,0.2)", width=2),
            fillcolor="rgba(255,182,193,0.5)",  # 使用半透明的浅粉色
        )
        
        # 添加多头注意力标注
        if i == 0:
            fig.add_annotation(
                x=x0 + decoder_width/2, y=y0 + decoder_height/2,
                text=f"解码器层 {i+1}<br>({n_head} 头注意力)",
                showarrow=False,
                font=dict(size=10, color="rgba(0,0,0,0.7)"),
            )
        else:
            fig.add_annotation(
                x=x0 + decoder_width/2, y=y0 + decoder_height/2,
                text=f"解码器层 {i+1}",
                showarrow=False,
                font=dict(size=10, color="rgba(0,0,0,0.7)"),
            )
    
    # 添加输出层
    output_x = 2.2 + (max(num_encoder_layers, num_decoder_layers) - 1) * (encoder_width + encoder_spacing) + encoder_width + 0.5
    output_width, output_height = 0.8, 0.8
    
    # 输出层外框
    fig.add_shape(
        type="rect",
        x0=output_x, y0=1.25,
        x1=output_x + output_width, y1=1.25 + output_height,
        line=dict(color="rgba(0,0,0,0.2)", width=2),
        fillcolor="white",
    )
    
    # 添加输出数据块
    create_data_blocks(
        fig,
        output_x + 0.1, 1.25 + 0.1,
        output_width - 0.2, output_height - 0.2,
        1,  # 输出序列长度为1
        "horizontal",
        "rgba(255,255,224,0.5)",  # 使用半透明的浅黄色
        f"预测值\n(t={seq_len})"
    )
    
    # 添加连接线
    def add_arrow(x0, y0, x1, y1):
        fig.add_shape(
            type="path",
            path=f"M {x0},{y0} L {x1},{y1}",
            line=dict(color="rgba(0,0,0,0.3)", width=2),
        )
        # 添加箭头
        angle = np.arctan2(y1-y0, x1-x0)
        arrow_length = 0.1
        arrow_angle = np.pi/6
        fig.add_shape(
            type="path",
            path=f"M {x1},{y1} L {x1-arrow_length*np.cos(angle+arrow_angle)},{y1-arrow_length*np.sin(angle+arrow_angle)} M {x1},{y1} L {x1-arrow_length*np.cos(angle-arrow_angle)},{y1-arrow_length*np.sin(angle-arrow_angle)}",
            line=dict(color="rgba(0,0,0,0.3)", width=2),
        )
    
    # 输入到第一个编码器
    add_arrow(input_x0 + input_width, input_y0 + input_height/2, 2.2, 0.9)
    
    # 编码器层之间
    for i in range(num_encoder_layers - 1):
        x0 = 2.2 + i * (encoder_width + encoder_spacing) + encoder_width
        x1 = 2.2 + (i + 1) * (encoder_width + encoder_spacing)
        add_arrow(x0, 0.9, x1, 0.9)
    
    # 编码器到解码器
    if num_encoder_layers > 0:
        last_encoder_x = 2.2 + (num_encoder_layers - 1) * (encoder_width + encoder_spacing) + encoder_width
        add_arrow(last_encoder_x, 0.9, last_encoder_x, 2)
    
    # 解码器层之间
    for i in range(num_decoder_layers - 1):
        x0 = 2.2 + i * (decoder_width + decoder_spacing) + decoder_width
        x1 = 2.2 + (i + 1) * (decoder_width + decoder_spacing)
        add_arrow(x0, 2.4, x1, 2.4)
    
    # 最后一个解码器到输出
    if num_decoder_layers > 0:
        last_decoder_x = 2.2 + (num_decoder_layers - 1) * (decoder_width + decoder_spacing) + decoder_width
        add_arrow(last_decoder_x, 2.4, output_x, 1.65)
    
    return fig

def ModelInspector():
    """
    创建模型示意图组件
    """
    return dmc.Paper(
        [
            dmc.Title("模型结构", order=3, mb="md"),
            dcc.Graph(
                id="model-diagram",
                figure=create_model_diagram(4, 2, 2, 10),  # 默认参数
                config={
                    "responsive": True
                },
                style={
                    "height": "500px"
                }
            ),
        ],
        p="md",
        withBorder=True,
        radius="md",
        shadow="sm",
        style={
            "width": "600px",
            "height": "600px",
        }
    )