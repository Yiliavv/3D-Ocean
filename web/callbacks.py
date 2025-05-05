from arrow import Arrow
from dash import Input, Output, State, callback

from src.config.area import Area
from tasks.TrainerTask import TrainerTask

sst_trainer_task = None

@callback(
    Output("dataset-start-time", "children"),
    Input("offset-input", "value"),
)
def update_dataset_start_time(offset):
    return f"数据集开始时间: {Arrow(2004, 1, 1).shift(months=offset).format('YYYY-MM-DD')}"


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
    State("optimizer-input", "value"),
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
    
@callback(
    Input("stop-training-button", "n_clicks"),
)
def stop_training():
    if sst_trainer_task is not None:
        sst_trainer_task._stop()
        sst_trainer_task = None
    

