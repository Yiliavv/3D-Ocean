import pandas as pd

from dash import Dash

from web.dashboard import MainLayout
from web.callbacks import *  # 导入所有回调函数

# Incorporate data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

app = Dash(__name__)

app.layout = [
   MainLayout() 
]

if __name__ == '__main__':
    app.run(debug=True)


