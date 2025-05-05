# 操作按钮
from dash import html, dcc
import dash_mantine_components as dmc

def Actions():
    return dmc.Group([
        dmc.Button(
            "启动训练",
            id="start-training-button",
            variant="gradient",
            gradient={"from": "teal", "to": "blue", "deg": 60},
        ),
        dmc.Button(
            "终止训练",
            id="stop-training-button",
            variant="gradient",
            gradient={"from": "orange", "to": "red"},
        ),
    ])