import numpy as np
import matplotlib.pyplot as plt

from MonteCarloModel import AcquisitionModel, Path, SpiralPath, ExperimentResult
from MonteCarloEngine import MonteCarloEngine
from MonteCarloSampler import Gaussian2DSampler
from MonteCarloAnalyzer import AcquisitionAnalyzer
from MonteCarloParameters import Parameter, ParameterSet

mc_params = ParameterSet([
    Parameter("spiral_step", kind="fixed", value=0.05),
    Parameter("max_radius", kind="fixed", value=3.0),
    Parameter("velocity", kind="fixed", value=0.1),
    Parameter(
        "target_position",
        kind="distribution",
        dist="Gaussian2D",
        mean=[0, 0],
        cov=[[1, 0], [0, 1]]
    ),

    Parameter("irradiance", kind="fixed", value=25e-6),
    Parameter("sensor_area", kind="fixed", value=1e-4),
])

# --------------------------------------------------
# Model & engine
# --------------------------------------------------

model = AcquisitionModel(hit_radius=0.2)

engine = MonteCarloEngine(
    parameter_set=mc_params,
    model=model,
    seed=42
)

results = engine.run(500)
analyzer = AcquisitionAnalyzer(results)

print("P(acquisition):", analyzer.probability_of_acquisition())

# --------------------------------------------------
# Plots
# --------------------------------------------------

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2)

ax0 = fig.add_subplot(gs[0, 0])
results[0].plot_geometry(ax0)

ax1 = fig.add_subplot(gs[0, 1])
analyzer.plot_spatial_map(ax1)

ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
analyzer.plot_time_pdf_cdf(
    ax2,
    ax3,
    probability_threshold=0.95  # default reliability requirement
)
plt.tight_layout()
plt.show()