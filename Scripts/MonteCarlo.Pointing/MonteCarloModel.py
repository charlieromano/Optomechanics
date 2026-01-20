# MonteCarloModel.py
import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ============================================================
# Experiment Result
# ============================================================

class ExperimentResult:
    def __init__(
        self,
        target,
        trajectory,
        hit,
        time_to_hit,
        total_time,
        dwell_time,
        energy,
        physics=None,
        warnings=None,
        errors=None,
        valid=True,
    ):
        self.target = target
        self.trajectory = trajectory
        self.hit = hit
        self.time_to_hit = time_to_hit
        self.total_time = total_time
        self.dwell_time = dwell_time
        self.energy = energy
        self.physics = physics
        self.warnings = warnings or []
        self.errors = errors or []
        self.valid = valid

    @classmethod
    def invalid(cls, errors, warnings):
        return cls(
            target=None,
            trajectory=None,
            hit=False,
            time_to_hit=np.nan,
            total_time=0.0,
            dwell_time=0.0,
            energy=0.0,
            errors=errors,
            warnings=warnings,
            valid=False,
        )
    
    def plot_geometry(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        if self.trajectory is not None:
            ax.plot(
                self.trajectory[:, 0],
                self.trajectory[:, 1],
                "k-",
                lw=1,
            )

        if self.target is not None:
            color = "green" if self.hit else "red"
            ax.scatter(
                self.target[0],
                self.target[1],
                c=color,
                s=60,
                zorder=3,
            )

        ax.set_aspect("equal")
        ax.set_xlabel("x [rad]")
        ax.set_ylabel("y [rad]")
        ax.grid(True)
        return ax

# ============================================================
# Derived physics container
# ============================================================

@dataclass
class AcquisitionDerived:
    spot_radius: float
    theta_fou: float
    spiral_a: float
    theta_max: float
    num_turns: float
    step_length: float
    dt: float
    energy_per_dwell: float
    irradiance: float

# ============================================================
# Unified Acquisition Model
# ============================================================

class AcquisitionModel:
    def __init__(self):
        self.errors = []
        self.warnings = []

    # ----------------------------
    # Physics derivation
    # ----------------------------
    def _derive(self, p):
        theta_fou = p["N_sigma"] * p["sigma_theta"]
        spot_radius = p["theta_div"] / 2.0
        dt = p["dwell_time"]
        step_length = p["velocity"] * dt
        energy_per_dwell = p["power"] * dt

        spiral_a = (
            p["theta_div"] * (1.0 - p["overlap_factor"])
        ) / (2.0 * math.pi)

        theta_max = theta_fou / spiral_a
        num_turns = theta_max / (2.0 * math.pi)
        irradiance = p["power"] / (math.pi * spot_radius**2)

        return AcquisitionDerived(
            spot_radius=spot_radius,
            theta_fou=theta_fou,
            spiral_a=spiral_a,
            theta_max=theta_max,
            num_turns=num_turns,
            step_length=step_length,
            dt=dt,
            energy_per_dwell=energy_per_dwell,
            irradiance=irradiance,
        )

    # ----------------------------
    # Validation
    # ----------------------------
    def _validate(self, p, d):
        self.errors.clear()
        self.warnings.clear()

        if d.energy_per_dwell < p["energy_threshold"]:
            self.errors.append("Energy per dwell below threshold")

        max_step = p["theta_div"] * (1.0 - p["overlap_factor"])
        if d.step_length > max_step:
            self.warnings.append("Along-path undersampling")

        radial_spacing = 2.0 * math.pi * d.spiral_a
        if radial_spacing > max_step:
            self.warnings.append("Radial undersampling")

        if p["N_sigma"] < 3.0:
            self.warnings.append("Target distribution truncated")

        return len(self.errors) == 0

    # ----------------------------
    # Geometry
    # ----------------------------
    def _build_spiral(self, d):
        n = int(d.theta_max / d.spiral_a)
        theta = np.linspace(0.0, d.theta_max, n)
        r = d.spiral_a * theta
        return np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    # ----------------------------
    # Simulation core
    # ----------------------------
    def _simulate(self, traj, target, d):
        time = 0.0
        dwell_time = 0.0
        energy = 0.0
        hit = False
        time_to_hit = np.nan

        for p in traj:
            if np.linalg.norm(p - target) <= d.spot_radius:
                if not hit:
                    hit = True
                    time_to_hit = time
                dwell_time += d.dt
                energy += d.energy_per_dwell
            time += d.dt

        return hit, time_to_hit, time, dwell_time, energy

    # ----------------------------
    # Public API
    # ----------------------------
    def run(self, params):
        d = self._derive(params)

        if not self._validate(params, d):
            return ExperimentResult.invalid(
                errors=self.errors.copy(),
                warnings=self.warnings.copy(),
            )

        traj = self._build_spiral(d)

        hit, t_hit, t_tot, dwell, energy = self._simulate(
            traj, params["target_position"], d
        )

        return ExperimentResult(
            target=params["target_position"],
            trajectory=traj,
            hit=hit,
            time_to_hit=t_hit,
            total_time=t_tot,
            dwell_time=dwell,
            energy=energy,
            physics=d,
            warnings=self.warnings.copy(),
            valid=True,
        )
# ============================================================