"""
PHYSICS CONTRACT — PAT ACQUISITION MONTE CARLO MODEL
===================================================

Purpose
-------
This model simulates the probability and timing of target acquisition
for a PAT (Pointing, Acquisition, and Tracking) system performing a
SPIRAL SCAN over a Field of Uncertainty (FoU).

The target position is random (Gaussian in angle space), and acquisition
occurs when the scanning beam delivers sufficient energy while spatially
covering the target.

This document defines the physical meaning of all parameters and the
assumptions enforced by the model.

---------------------------------------------------
COORDINATE SYSTEM
---------------------------------------------------
- Angular space (θx, θy) in radians
- All distances are angular distances
- Target distribution and scan trajectory live in the same angular plane

---------------------------------------------------
SCAN GEOMETRY (SPATIAL COMPLETENESS)
---------------------------------------------------
The scan follows an Archimedean spiral:

    r(θ) = a · θ

where:
    a = spiral pitch coefficient

Spiral pitch is derived from beam divergence and overlap factor:

    a = (θ_div · (1 − overlap)) / (2π)

Definitions:
- θ_div        : full beam divergence (rad)
- spot_radius  : θ_div / 2
- overlap      : fractional spatial overlap between adjacent spiral turns

Interpretation:
- Overlap controls SPATIAL COMPLETENESS
- If overlap is too small → spatial gaps appear
- Radial spacing criterion:

    radial_spacing = 2πa ≤ beam_diameter · (1 − overlap)

---------------------------------------------------
FIELD OF UNCERTAINTY (FoU)
---------------------------------------------------
- Target angular uncertainty is Gaussian with std = σ_theta
- FoU radius is truncated at:

    θ_FoU = N_sigma · σ_theta

The spiral is generated until it fully covers θ_FoU.

---------------------------------------------------
SCAN DYNAMICS (ENERGETIC COMPLETENESS)
---------------------------------------------------
The scan is PHYSICALLY CONTINUOUS.

There is no true dwell time in hardware.
Instead, acquisition depends on energy integrated while the beam sweeps
over the target footprint.

An EFFECTIVE dwell time is defined as:

    t_eff ≈ beam_diameter · (1 − overlap) / scan_velocity

where:
- scan_velocity is angular speed (rad/s)

Energy delivered during target crossing:

    E_eff = power · t_eff

Acquisition condition (ENERGETIC COMPLETENESS):

    E_eff ≥ energy_threshold

Interpretation:
- Velocity controls ENERGETIC COMPLETENESS
- Overlap does NOT increase energy, only spatial coverage

---------------------------------------------------
SIMULATION STRATEGY
---------------------------------------------------
- The physical scan is continuous
- The simulation discretizes the trajectory for numerical purposes
- Discrete dwell_time is a NUMERICAL STEP, not a physical dwell

Validation uses continuous physics:
- Energy threshold → effective dwell time
- Spatial gaps → spiral spacing

Simulation kernels may use discrete stepping for performance, but
validation guarantees physical consistency.

---------------------------------------------------
ORTHOGONAL CONSTRAINTS
---------------------------------------------------
- Overlap factor → spatial completeness
- Scan velocity → energetic completeness

They are independent and must BOTH be satisfied for valid acquisition.

---------------------------------------------------
KNOWN LIMITATIONS
---------------------------------------------------
- Beam profile is assumed uniform (top-hat)
- No temporal modulation, jitter, or tracking feedback
- Single-pass spiral only (no revisit strategy)

===================================================
END OF PHYSICS CONTRACT
"""
