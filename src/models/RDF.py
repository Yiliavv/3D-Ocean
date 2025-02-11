# 随机决策森林
from sklearn.ensemble import RandomForestRegressor
from torch import nn

class RDFNetwork(nn.Module):
    def __init__(self):
        super(RDFNetwork, self).__init__()
        self.model = RandomForestRegressor(
            n_estimators=500, random_state=42, max_features=5, 
            criterion='friedman_mse', verbose=1
            )
        
    def get_model(self):
        return self.model

    def forward(self, input, output):
        return self.model.fit(input, output)
