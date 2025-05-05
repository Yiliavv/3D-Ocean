from arrow import Arrow
from dash import Input, Output, State, callback

from src.config.params import Areas
from src.config.area import Area
from tasks.TrainerTask import TrainerTask

sst_trainer_task = None

@callback(
    Output("dataset-start-time", "children"),
    Input("offset-input", "value"),
)
def update_dataset_start_time(offset):
    return f"数据集开始时间: {Arrow(2004, 1, 1).shift(months=offset).format('YYYY-MM-DD')}"

# 更新区域
@callback(
    Output("longitude-input", "value"),
    Output("latitude-input", "value"),
    Input("area-select", "value"),
)
def update_area_select(value):
    if value is None:
        return [-180, 180], [-90, 90]
    index = int(value)
    area = Areas[index]
    return area.lon, area.lat

# 启动训练任务
@callback(
    Input("start-training-button", "n_clicks"),
    # 数据集参数
    State("seq-len-input", "value"),
    State("offset-input", "value"),
    State("longitude-input", "value"),
    State("latitude-input", "value"),
    State("resolution-input", "value"),
    # 模型参数
    State("n-head-input", "value"),
    State("num-encoder-layers-input", "value"),
    State("num-decoder-layers-input", "value"),
    State("learning-rate-input", "value"),
    State("dropout-input", "value"),
    State("optimizer-select", "value"),
    # 训练参数
    State("batch-size-input", "value"),
    State("epochs-input", "value"),
    
)
def start_training(
    n_clicks, seq_len, offset, longitude, latitude, resolution,
    n_head, num_encoder_layers, num_decoder_layers, learning_rate, dropout, optimizer,
    batch_size, epochs
):
    print(n_clicks, seq_len, offset, longitude, latitude, resolution,
          n_head, num_encoder_layers, num_decoder_layers, learning_rate, dropout, optimizer,
          batch_size, epochs)
    
    # 创建区域
    region = Area("Test", longitude, latitude, description="模型训练区域")
    
    # 创建任务
    sst_trainer_task = TrainerTask(
        region, seq_len, offset, n_head, num_encoder_layers,
        num_decoder_layers, learning_rate, dropout, optimizer,
        batch_size, epochs)
    
    # 启动任务
    sst_trainer_task.start()


# 停止训练任务
@callback(
    Input("stop-training-button", "n_clicks"),
)
def stop_training():
    if sst_trainer_task is not None:
        sst_trainer_task._stop()
        sst_trainer_task = None

    return;
    

