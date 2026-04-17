# main.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from MonteCarloEngine import MonteCarloEngine, DirectMonteCarlo
from MonteCarloModel import AcquisitionModel, Unit
from MonteCarloAnalyzer import AcquisitionAnalyzer
from MonteCarloParameters import Parameter, ParameterSet

# --------------------------------------------------
# Units
# --------------------------------------------------
RAD = Unit("rad")
SEC = Unit("s")
WATT = Unit("W")
JOULE = Unit("J")
RAD_PER_SEC = Unit("rad/s")
METER = Unit("m")

# --------------------------------------------------
# Monte Carlo Parameters
# --------------------------------------------------
mc_params = ParameterSet([
    Parameter("sigma_theta", kind="fixed", value=(4e-3, RAD)),
    Parameter("theta_div", kind="fixed", value=(100e-6, RAD)),
    Parameter("N_sigma", kind="fixed", value=3.0),
    Parameter("overlap_factor", kind="fixed", value=0.01),
    Parameter("velocity", kind="fixed", value=(0.08, RAD_PER_SEC)),
    Parameter("dwell_time", kind="fixed", value=(100e-6, SEC)),
    Parameter("power", kind="fixed", value=(0.5e-6, WATT)),
    Parameter("receiver_diameter", kind="fixed", value=(0.08, METER)),
    Parameter("energy_threshold", kind="fixed", value=(24e-12, JOULE)),
    Parameter("simulation_resolution", kind="fixed", value=(3e-5, SEC)),
    Parameter(
        "target_position",
        kind="distribution",
        dist="Gaussian2D",
        mean=[0.0, 0.0],
        cov=[[4e-3**2, 0.0], [0.0, 4e-3**2]],
    ),
    Parameter("scan_mode", kind="fixed", value="continuous")
])


# --------------------------------------------------
# ENGINE
# --------------------------------------------------
N_MonteCarlo = 10
model = AcquisitionModel(backend="python")
engine = MonteCarloEngine(
    model=model,
    method=DirectMonteCarlo(mc_params),
    seed=42,
    progress=True
)

# Run simulation
results = engine.run(N_MonteCarlo)
model.close()

analyzer = AcquisitionAnalyzer(results)
print("P(acquisition):", analyzer.probability_of_acquisition())

# --------------------------------------------------
# Filter valid results for plotting
# --------------------------------------------------
valid_results = [
    r for r in results
    if getattr(r, 'physics', None) is not None
    and getattr(r, 'target', None) is not None
]

# --------------------------------------------------
# PLOTS
# --------------------------------------------------
fig = plt.figure(figsize=(16, 9))
gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.35)

# -------------------
# Top row: text, spiral, spatial map
# -------------------
gs_top = gs_main[0].subgridspec(1, 3, wspace=0.3)
ax_text = fig.add_subplot(gs_top[0])
ax_spiral = fig.add_subplot(gs_top[1])
ax_spatial = fig.add_subplot(gs_top[2])
ax_text.axis('off')

# Parameter text
AcquisitionAnalyzer.add_parameter_text(
    fig, mc_params, n_simulations=N_MonteCarlo, x=0.15, y=0.88, fontsize=12
)

# -------------------
# Spiral plot: path, mean hit, mean target
# -------------------
if valid_results:
    # 1. Trajectory Plotting (Handle None from Python backend)
    # If Python backend was used, we need to regenerate the path for the plot
    traj_plot = valid_results[0].trajectory
    d = valid_results[0].physics
    
    if traj_plot is None:
        # Regenerate for visualization if not stored during simulation
        # Access the value from the ParameterSet object
        mode_val = mc_params["scan_mode"].value
        traj_plot = model._build_spiral(d, mode=mode_val)

    if traj_plot is not None and len(traj_plot) > 0:
        max_plot_points = 5000
        if traj_plot.shape[0] > max_plot_points:
            step = max(1, traj_plot.shape[0] // max_plot_points)
            traj_plot = traj_plot[::step]
        ax_spiral.plot(traj_plot[:,0], traj_plot[:,1], color='black', linewidth=0.4, label="Spiral path")

    # 2. Mean hit position (Filter out None/NaN)
    hits = np.array([r.first_hit_pos for r in valid_results if r.hit and r.first_hit_pos is not None])
    if hits.size > 0:
        mean_hit = np.mean(hits, axis=0)
        ax_spiral.plot(mean_hit[0], mean_hit[1], 'go', markersize=6, label="Mean hit")

    # 3. Mean target spot
    # Use the first result's target as reference if they are static, or mean if jittered
    targets = np.array([r.target for r in valid_results])
    mean_target = np.mean(targets, axis=0)
    circle = Circle(mean_target, d.spot_radius, color='blue', fill=False,
                    linestyle='--', linewidth=1.2, label="Mean target")
    ax_spiral.add_patch(circle)

# Set the window to focus on the Field of Uncertainty (FoU)
limit = d.theta_fou * 1.1 # 10% margin
ax_spiral.set_xlim(-limit, limit)
ax_spiral.set_ylim(-limit, limit)
# Add labels so you can see the scale
ax_spiral.set_xlabel("Azimuth (rad)")
ax_spiral.set_ylabel("Elevation (rad)")
ax_spiral.set_title("Spiral & Mean Hit", fontsize=12)
ax_spiral.set_aspect("equal")
ax_spiral.grid(True)
ax_spiral.legend(fontsize=10, loc='upper right')

# -------------------
# Spatial hit/miss map
# -------------------
valid_hits = [r for r in valid_results if getattr(r, 'hit', False)]
if valid_hits:
    analyzer.plot_spatial_map(ax_spatial)
else:
    ax_spatial.text(0.5, 0.5, "No hits to display", ha='center', va='center')
ax_spatial.legend(fontsize=10, loc='upper right')

# -------------------
# Bottom row: PDF and CDF
# -------------------
gs_bottom = gs_main[1].subgridspec(1, 2, wspace=0.3)
ax_pdf = fig.add_subplot(gs_bottom[0])
ax_cdf = fig.add_subplot(gs_bottom[1])

analyzer.plot_time_pdf_cdf(
    ax_pdf=ax_pdf,
    ax_cdf=ax_cdf,
    show_percentiles=[0.95, 0.99],
    percentile_line_styles={
        0.95: ('red', 'dashed'),
        0.99: ('red', 'dotted')
    }
)

ax_pdf.legend(fontsize=10, loc='upper right')
ax_cdf.legend(fontsize=10, loc='upper left')

# -------------------
# Layout adjustments
# -------------------
plt.tight_layout()
fig.subplots_adjust(top=0.92)  # leave room for parameter text
plt.show()