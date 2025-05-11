import dash_mantine_components as dmc
from dash_iconify import DashIconify
from threading import Thread

def TaskMonitor():
    return dmc.Paper(
        w="400px",
        p="md",
        shadow="sm",
        id="task-monitor",
    )
    
def TaskItem(trainer_task: Thread):
    id = trainer_task.native_id
    
    return dmc.Group(
            [
                dmc.Progress(
                    value=0,
                    size="xl",
                    radius="xl",
                    animated=True,
                    id="training-progress",
                    w="300px",
                ),
                dmc.ActionIcon(
                    DashIconify(
                        icon="fluent:caret-right-16-filled",
                        width=22,
                    ),
                    variant="light",
                    color="lime",
                    id={ "index": id, "type": "task-stop-button" },
                ),
            ],
        )