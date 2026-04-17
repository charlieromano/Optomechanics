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
    Parameter("sigma_theta", kind="fixed", value=(4.0e-3, RAD)),
    Parameter("theta_div", kind="fixed", value=(0.1e-3, RAD)),
    Parameter("N_sigma", kind="fixed", value=3.0),
    Parameter("overlap_factor", kind="fixed", value=0.01),
    Parameter("velocity", kind="fixed", value=(0.1, RAD_PER_SEC)),
    Parameter("dwell_time", kind="fixed", value=(100e-6, SEC)),
    Parameter("power", kind="fixed", value=(0.3e-6, WATT)),
    Parameter("receiver_diameter", kind="fixed", value=(0.08, METER)),
    Parameter("energy_threshold", kind="fixed", value=(30e-12, JOULE)),
    Parameter("simulation_resolution", kind="fixed", value=(10e-6, SEC)),
    Parameter(
        "target_position",
        kind="distribution",
        dist="Gaussian2D",
        mean=[0.0, 0.0],
        cov=[[4e-3**2, 0.0], [0.0, 4e-3**2]],
    ),
    Parameter("scan_mode", kind="fixed", value="stare_step")
])


# --------------------------------------------------
# ENGINE
# --------------------------------------------------
N_MonteCarlo = 100
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
# Spiral plot: path, representative hit, representative target
# -------------------
if valid_results and len(valid_results) > 0:
    res_ref = valid_results[0]
    d = res_ref.physics  # This object has all your derived parameters
    
    # 1. FIXED: Get mode directly from the physics object 'd' 
    # (We assigned it in your _derive function earlier!)
    mode_val = getattr(d, 'mode', 'continuous') 
    
    # Force regeneration of the path for visualization
    traj_plot = model._build_spiral(d, mode=mode_val)

    if traj_plot is not None and len(traj_plot) > 0:
        # Visibility: Don't downsample too much
        max_plot_points = 20000
        if traj_plot.shape[0] > max_plot_points:
            step = traj_plot.shape[0] // max_plot_points
            traj_plot = traj_plot[::step]
            
        ax_spiral.plot(traj_plot[:,0], traj_plot[:,1], color='black', 
                       linewidth=0.5, alpha=0.6, label="Spiral path")

    # 2. Representative Hit (Physical Geometry)
    hit_results = [r for r in valid_results if r.hit]
    if hit_results:
        best_r = hit_results[0]
        # Beam Spot (Illumination Area)
        beam_circle = Circle(best_r.first_hit_pos, d.spot_radius, color='green', 
                             fill=True, alpha=0.3, label="Beam at Hit")
        ax_spiral.add_patch(beam_circle)
        # Target location
        ax_spiral.plot(best_r.target[0], best_r.target[1], 'bx', markersize=8, label="Target")
    else:
        ax_spiral.text(0, 0, "No hits detected", color='red', ha='center')

    # 3. Reference Search Area (3-sigma)
    ax_spiral.add_patch(Circle((0, 0), d.theta_fou, color='red', fill=False, 
                               linestyle='--', linewidth=1.5, label="3σ FoU"))

    # Limits and Formatting
    limit = d.theta_fou * 1.1
    ax_spiral.set_xlim(-limit, limit)
    ax_spiral.set_ylim(-limit, limit)
    ax_spiral.set_aspect("equal")
    ax_spiral.set_title("Spiral Geometry")
    ax_spiral.legend(fontsize=8, loc='upper right')

# -------------------
# 2. SPATIAL CLOUD MAP (Restoration)
# -------------------
if valid_results:
    # This plots the Green (hits) and Red (misses) points
    analyzer.plot_spatial_map(ax_spatial)
    
    # Sync limits with the spiral plot
    ax_spatial.set_xlim(-limit, limit)
    ax_spatial.set_ylim(-limit, limit)
    ax_spatial.set_aspect("equal")
    ax_spatial.set_title("Monte Carlo Spatial Map")
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