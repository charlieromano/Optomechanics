import numpy as np
from MCParameter import Parameter
from MCSampler import JointParameterSampler
from MCEngine import MonteCarloSimulation
from MCAnalyzer import ResultAnalyzer

class ExampleModel:
    def evaluate(self, inputs):
        x = inputs["x"]
        y = inputs["y"]
        z = x**2 + np.sin(y)
        return {"z": z}


params = [
    Parameter("x", distribution="normal", mean=0, std=1),
    Parameter("y", distribution="uniform", low=0, high=2*np.pi),
]

model = ExampleModel()

mc = MonteCarloSimulation(
    parameters=params,
    model=model,
    n_samples=10000
)

results = mc.run()
analyzer = ResultAnalyzer(results)

print(analyzer.summary("z"))
print("P(z < 1.0) =", analyzer.probability_less_than("z", 1.0))
