# 随机决策森林
from sklearn.ensemble import RandomForestRegressor

class RDFNetwork():
    def __init__(self):
        super(RDFNetwork).__init__()
        self.model = RandomForestRegressor(
            n_estimators=200, random_state=42,
            n_jobs=8, verbose=True
        )
        
    def get_model(self):
        return self.model

    def forward(self, input, output):
        return self.model.fit(input, output)
