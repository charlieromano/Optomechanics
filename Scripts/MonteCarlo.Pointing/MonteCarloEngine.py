# MonteCarloEngine.py
import numpy as np

class MonteCarloEngine:
    def __init__(self, model, method, *, seed=None):
        self.model = model
        self.method = method
        self.rng = np.random.default_rng(seed)

    def run(self, n):
        results = []
        for i in range(n):
            params = self.method.propose(self.rng)
            result = self.model.run(params)
            self.method.observe(result)
            results.append(result)

            if hasattr(self.method, "should_stop"):
                if self.method.should_stop(i, results):
                    break

        return results


class DirectMonteCarlo:
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set

    def propose(self, rng):
        return self.parameter_set.resolve(rng)

    def observe(self, result):
        pass


class InverseMonteCarlo:
    def __init__(self, parameter_set, objective):
        self.parameter_set = parameter_set
        self.objective = objective
        self.scores = []

    def propose(self, rng):
        return self.parameter_set.resolve(rng)

    def observe(self, result):
        score = self.objective.evaluate(result)
        self.scores.append(score)