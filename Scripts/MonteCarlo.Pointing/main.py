# main.py
import numpy as np
import matplotlib.pyplot as plt
import importlib

from MonteCarloEngine import MonteCarloEngine, DirectMonteCarlo
from MonteCarloModel import AcquisitionModel
from MonteCarloAnalyzer import AcquisitionAnalyzer
from MonteCarloParameters import Parameter, ParameterSet

# --------------------------------------------------
# Monte Carlo Parameters
# --------------------------------------------------

sigma_theta = 4e-3
theta_div = 300e-6
N_sigma = 3.0
overlap_factor = 0.05

mc_params = ParameterSet([
    Parameter("sigma_theta", kind="fixed", value=sigma_theta),
    Parameter("theta_div", kind="fixed", value=theta_div),
    Parameter("N_sigma", kind="fixed", value=N_sigma),
    Parameter("overlap_factor", kind="fixed", value=overlap_factor),
    Parameter("velocity", kind="fixed", value=0.02),
    Parameter("dwell_time", kind="fixed", value=0.01),
    Parameter("power", kind="fixed", value=10.0),
    Parameter("energy_threshold", kind="fixed", value=1e-3),
    Parameter(
        "target_position",
        kind="distribution",
        dist="Gaussian2D",
        mean=[0.0, 0.0],
        cov=[[sigma_theta**2, 0.0], [0.0, sigma_theta**2]],
    ),
])

# --------------------------------------------------
# Engine
# --------------------------------------------------

model = AcquisitionModel()

engine = MonteCarloEngine(
    model=model,
    method=DirectMonteCarlo(mc_params),
    seed=42,
)

results = engine.run(1)

analyzer = AcquisitionAnalyzer(results)
print("P(acquisition):", analyzer.probability_of_acquisition())

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots()
results[0].plot_geometry(ax)
plt.show()
