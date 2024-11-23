# 随机决策森林
from sklearn.ensemble import RandomForestRegressor
from torch import nn

class RDFNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(RDFNetwork, self).__init__(self, **kwargs)
        self.model = RandomForestRegressor(n_estimators=300, random_state=42, max_features=5, verbose=True)

    def get_config(self):
        return {"shape": self.model}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    def call(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
