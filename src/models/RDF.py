# 随机决策森林
from sklearn.ensemble import RandomForestRegressor

class RDFNetwork():
    def __init__(self):
        super(RDFNetwork).__init__()
        self.model = RandomForestRegressor(
            n_estimators=50, random_state=10, n_jobs=10, verbose=True
        )
        
    def get_model(self):
        return self.model

    def forward(self, input, output):
        return self.model.fit(input, output)
