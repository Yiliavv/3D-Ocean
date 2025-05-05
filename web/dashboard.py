# 操作面板，用于调参和训练模型

import dash_mantine_components as dmc

from components.Inputs import *
from components.ModelInspector import ModelInspector
from components.ParamsInspector import ParamsInspector
from components.Actions import Actions

def MainLayout():
    return dmc.MantineProvider(
        dmc.AppShell(
            [
                Main([
                    dmc.Group(
                        [
                            ModelInspector(),
                            ParamsInspector(),
                        ],
                        gap="md",
                    ),
                    dmc.Space(h=24),
                    ActionsPanel(),
                ]),
                Aside([
                    AsidePanelHeader(),
                    dmc.Space(h=24),
                    AsidePanelBody(),
                ]),
            ],
            aside={
                "width": 300,   
            }
        )
    )

def Aside(children = "Aside"):
    return dmc.AppShellAside(children, p="md")

def Main(children = "Main"):
    return dmc.AppShellMain(children, p="md")

# aside panel

def AsidePanelHeader():
    return dmc.Center(
        dmc.Title("控制面板", order=3)
    )

def AsidePanelBody():
    return dmc.ScrollArea(
        [  
            DatasetPanel(),
            ModelPanel(),
            TrainingPanel(),
        ],
    )

def DatasetPanel():
    return dmc.Stack(
        [
            dmc.Text("数据集参数"),
            dmc.Divider(variant="solid"),
            SeqLenInput(),
            OffsetInput(),
            AreaSelect(),
            LongitudeInput(),
            LatitudeInput(),
            ResolutionInput(),
            dmc.Space(h=16),
        ]
    )

def ModelPanel():
    return dmc.Stack(
        [
            dmc.Text("模型参数"),
            dmc.Divider(variant="solid"),
            NHeadInput(),
            NumEncoderLayersInput(),
            NumDecoderLayersInput(),
            LearningRateInput(),
            DropoutInput(),
            OptimizerSelect(),
            dmc.Space(h=16),
        ]
    )
    
def TrainingPanel():
    return dmc.Stack(
        [
            dmc.Text("训练参数"),
            dmc.Divider(variant="solid"),
            BatchSizeInput(),
            EpochsInput(),
            dmc.Space(h=16),
        ]
    )

def ActionsPanel():
    return dmc.Stack(
        [
            Actions(),
        ]
    )

