import numpy as np

class MonteCarloSimulation:
    def __init__(self, sampler, model, n_runs):
        self.sampler = sampler
        self.model = model
        self.n_runs = n_runs
        self.results = []
    def run(self):
        params_list = self.sampler.sample(self.n_runs)
        for params in params_list:
            t, x, y = self.model.simulate(params)
            self.results.append({
                "params": params,
                "time": t,
                "field": y
            })
        return self.results


