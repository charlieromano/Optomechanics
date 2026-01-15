import numpy as np
from MonteCarloParameters import Parameter, ParameterSet

class MonteCarloEngine:
    def __init__(self, parameter_set, model, seed=None):
        self.parameter_set = parameter_set
        self.model = model
        self.rng = np.random.default_rng(seed)

    def run_once(self):
        params = self.parameter_set.resolve(self.rng)
        return self.model.run(params)

    def run(self, n):
        return [self.run_once() for _ in range(n)]

