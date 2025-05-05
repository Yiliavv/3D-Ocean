import dash_mantine_components as dmc
from arrow import Arrow

def ParamsInspector():
    return dmc.Paper(
        [
            dmc.Title("预览", order=3, mb="md"),
            dmc.Divider(variant="solid"),
            dmc.Space(h=10),
            dmc.Text(
                "数据集开始时间: " + Arrow(2004, 1, 1).format("YYYY-MM-DD"),
                id="dataset-start-time",
                size="lg",
            ),
            dmc.Space(h=10),
            dmc.Text(
                "数据集结束时间: " + Arrow(2024, 12, 1).format("YYYY-MM-DD"),
                id="dataset-end-time",
                size="lg",
            ),
        ],
        p="md",
        withBorder=True,
        radius="md",
        shadow="sm",
        style={
            "width": "400px",
        }
    )

