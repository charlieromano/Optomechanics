# main.py
import numpy as np
import matplotlib.pyplot as plt
import importlib

from MonteCarloEngine import MonteCarloEngine, DirectMonteCarlo
from MonteCarloModel import AcquisitionModel, Unit
from MonteCarloAnalyzer import AcquisitionAnalyzer
from MonteCarloParameters import Parameter, ParameterSet


# --------------------------------------------------
# Monte Carlo Parameters
# --------------------------------------------------
# Base units
RAD = Unit("rad")
SEC = Unit("s")
WATT = Unit("W")
JOULE = Unit("J")
RAD_PER_SEC = Unit("rad/s")

mc_params = ParameterSet([
    Parameter("sigma_theta", kind="fixed", value=(4e-3, RAD)),
    Parameter("theta_div", kind="fixed", value=(350e-6, RAD)),
    Parameter("N_sigma", kind="fixed", value=3.0),
    Parameter("overlap_factor", kind="fixed", value=0.05),
    Parameter("velocity", kind="fixed", value=(0.15, RAD_PER_SEC)),
    Parameter("dwell_time", kind="fixed", value=(10e-6, SEC)),
    Parameter("power", kind="fixed", value=(1e-6, WATT)),
    Parameter("energy_threshold", kind="fixed", value=(100e-15, JOULE)),
    Parameter(
        "target_position",
        kind="distribution",
        dist="Gaussian2D",
        mean=[0.0, 0.0],
        cov=[[4e-3**2, 0.0], [0.0, 4e-3**2]],
    ),
])

# --------------------------------------------------
# Engine
# --------------------------------------------------

N_MonteCarlo = 1000

#model = AcquisitionModel()
model = AcquisitionModel(backend="numba")
#model = AcquisitionModel(backend="hdf5")

engine = MonteCarloEngine(
    model=model,
    method=DirectMonteCarlo(mc_params),
    seed=42,
)

results = engine.run(N_MonteCarlo)
model.close()


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
#plt.show()

def format_mc_parameters(param_set, n_simulations=None):
    """Return a multiline string with parameter name, value and unit."""
    lines = ["Monte Carlo parameters:"]

    params = param_set.parameters

    # Handle dict-based ParameterSet
    if isinstance(params, dict):
        iterable = params.items()
    else:
        iterable = [(p.name, p) for p in params]

    for name, p in iterable:
        if p.kind == "fixed":
            val = p.value
            if isinstance(val, tuple):
                value, unit = val
                lines.append(f"{name}: {value:g} {unit}")
            else:
                lines.append(f"{name}: {val}")
        else:
            lines.append(f"{name}: {p.dist}")

    # Add number of simulations at the end
    if n_simulations is not None:
        lines.append(f"N_simulations: {n_simulations}")

    return "\n".join(lines)

param_text = format_mc_parameters(mc_params, n_simulations=N_MonteCarlo)

fig.text(
    0.5, 0.98,                  # center-top of figure
    param_text,
    fontsize=9,
    va="top",
    ha="center",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85,
        edgecolor="gray"
    )
)

plt.show()
# --------------------------------------------------
