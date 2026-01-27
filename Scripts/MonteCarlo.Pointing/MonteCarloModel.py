# MonteCarloModel.py
import numpy as np
import math
import matplotlib.pyplot as plt
import h5py
from dataclasses import dataclass

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ---------------------------
# Unit class
# ---------------------------
class Unit:
    def __init__(self, name):
        self.name = name
    def __eq__(self, other):
        return isinstance(other, Unit) and self.name == other.name
    def __repr__(self):
        return self.name


RAD = Unit("rad")
SEC = Unit("s")
WATT = Unit("W")
JOULE = Unit("J")
RAD_PER_SEC = Unit("rad/s")


# ---------------------------
# ExperimentResult container
# ---------------------------
class ExperimentResult:
    def __init__(self, *, valid=True, errors=None, warnings=None, **data):
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.data = data

        # Direct attribute access
        for k, v in data.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return self.data[key]

    def keys(self):
        return self.data.keys()

    @classmethod
    def invalid(cls, errors=None, warnings=None):
        return cls(valid=False, errors=errors, warnings=warnings)


# ---------------------------
# Derived physics container
# ---------------------------
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


# ---------------------------
# NUMBA kernel
# ---------------------------
if NUMBA_AVAILABLE:
    @njit
    def _simulate_numba_kernel(traj, target, dt, spot_radius, energy_per_dwell):
        time = 0.0
        dwell_time = 0.0
        energy = 0.0
        hit = False
        time_to_hit = np.nan
        r2 = spot_radius * spot_radius

        for i in range(traj.shape[0]):
            dx = traj[i, 0] - target[0]
            dy = traj[i, 1] - target[1]

            if dx*dx + dy*dy <= r2:
                if not hit:
                    hit = True
                    time_to_hit = time
                dwell_time += dt
                energy += energy_per_dwell

            time += dt

        return hit, time_to_hit, time, dwell_time, energy


# ---------------------------
# AcquisitionModel
# ---------------------------
class AcquisitionModel:
    def __init__(self, backend="python", *, hdf5_file="mc_results.h5"):
        self.errors = []
        self.warnings = []
        self.backend = backend
        self._reference_trajectory = None

        self.hdf5_file = hdf5_file
        self._h5 = None
        self._run_index = 0

        if backend == "numba" and not NUMBA_AVAILABLE:
            raise RuntimeError("Numba backend requested but Numba is not available")
        if backend == "hdf5":
            self._init_hdf5()

    def _init_hdf5(self):
        self._h5 = h5py.File(self.hdf5_file, "w")
        self._grp = self._h5.create_group("runs")

        self._grp.create_dataset("hit", (0,), maxshape=(None,), dtype=np.bool_)
        self._grp.create_dataset("time_to_hit", (0,), maxshape=(None,), dtype=np.float64)
        self._grp.create_dataset("total_time", (0,), maxshape=(None,), dtype=np.float64)
        self._grp.create_dataset("dwell_time", (0,), maxshape=(None,), dtype=np.float64)
        self._grp.create_dataset("energy", (0,), maxshape=(None,), dtype=np.float64)
        self._grp.create_dataset("target", (0,2), maxshape=(None,2), dtype=np.float64)

    # ----------------------------
    # Parameter getter
    # ----------------------------
    def _get(self, params, name, expected_unit=None):
        v = params[name]
        if isinstance(v, tuple):
            value, unit = v
            if expected_unit and unit != expected_unit:
                self.errors.append(
                    f"Parameter '{name}' must have unit {expected_unit}, got {unit}"
                )
            return value
        else:
            if expected_unit:
                self.warnings.append(
                    f"Parameter '{name}' has no unit, expected {expected_unit}"
                )
            return v

    # ----------------------------
    # Derive physics
    # ----------------------------
    def _derive(self, p):
        sigma_theta = self._get(p, "sigma_theta", RAD)
        theta_div = self._get(p, "theta_div", RAD)
        N_sigma = self._get(p, "N_sigma")
        overlap = self._get(p, "overlap_factor")
        velocity = self._get(p, "velocity", RAD_PER_SEC)
        dwell_time = self._get(p, "dwell_time", SEC)
        power = self._get(p, "power", WATT)

        theta_fou = N_sigma * sigma_theta
        spot_radius = theta_div / 2.0
        step_length = velocity * dwell_time
        energy_per_dwell = power * dwell_time

        spiral_a = (theta_div * (1.0 - overlap)) / (2.0 * math.pi)
        theta_max = theta_fou / spiral_a
        num_turns = theta_max / (2.0 * math.pi)
        irradiance = power / (math.pi * spot_radius**2)

        return AcquisitionDerived(
            spot_radius=spot_radius,
            theta_fou=theta_fou,
            spiral_a=spiral_a,
            theta_max=theta_max,
            num_turns=num_turns,
            step_length=step_length,
            dt=dwell_time,
            energy_per_dwell=energy_per_dwell,
            irradiance=irradiance,
        )

    # ----------------------------
    # Validate physics
    # ----------------------------
    def _validate(self, p, d):
        self.errors.clear()
        self.warnings.clear()

        energy_threshold = self._get(p, "energy_threshold", JOULE)
        if d.energy_per_dwell < energy_threshold:
            self.errors.append("Energy per dwell below threshold")

        theta_div = self._get(p, "theta_div", RAD)
        overlap = self._get(p, "overlap_factor")
        max_step = theta_div * (1.0 - overlap)
        if d.step_length > max_step:
            self.warnings.append("Along-path undersampling")

        radial_spacing = 2.0 * math.pi * d.spiral_a
        if radial_spacing > max_step:
            self.warnings.append("Radial undersampling")

        N_sigma = self._get(p, "N_sigma")
        if N_sigma < 3.0:
            self.warnings.append("Target distribution truncated")

        return len(self.errors) == 0

    # ----------------------------
    # Geometry
    # ----------------------------
    def _spiral_angles(self, d):
        n = int(d.theta_max / d.spiral_a)
        return n, d.theta_max / max(n - 1, 1)

    def _build_spiral(self, d):
        n, dtheta = self._spiral_angles(d)
        theta = np.arange(n) * dtheta
        r = d.spiral_a * theta
        return np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    def _spiral_generator(self, d):
        n, dtheta = self._spiral_angles(d)
        for i in range(n):
            theta = i * dtheta
            r = d.spiral_a * theta
            yield r * math.cos(theta), r * math.sin(theta)

    # ----------------------------
    # Simulation core
    # ----------------------------
    def _simulate(self, traj, target, d):
        if self.backend == "numba":
            return _simulate_numba_kernel(
                traj,
                target,
                d.dt,
                d.spot_radius,
                d.energy_per_dwell
            )
        else:
            return self._simulate_python(traj, target, d)

    def _simulate_python(self, traj, target, d):
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
    # Streaming simulation (for HDF5 / RAM control)
    # ----------------------------
    def _simulate_streaming(self, target, d):
        time = 0.0
        dwell_time = 0.0
        energy = 0.0
        hit = False
        time_to_hit = np.nan
        r2 = d.spot_radius * d.spot_radius

        for x, y in self._spiral_generator(d):
            dx = x - target[0]
            dy = y - target[1]

            if dx*dx + dy*dy <= r2:
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
    def close(self):
        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None

    def run(self, params):
        d = self._derive(params)

        if not self._validate(params, d):
            return ExperimentResult.invalid(
                errors=self.errors.copy(),
                warnings=self.warnings.copy()
            )

        # Backend dispatch
        if self.backend == "hdf5":
            hit, t_hit, t_tot, dwell, energy = self._simulate_streaming(
                params["target_position"], d
            )

            if self._reference_trajectory is None:
                self._reference_trajectory = np.array(
                    list(self._spiral_generator(d)),
                    dtype=np.float32
                )

            traj = self._reference_trajectory

            i = self._run_index
            self._run_index += 1

            for name, value in [
                ("hit", hit),
                ("time_to_hit", t_hit),
                ("total_time", t_tot),
                ("dwell_time", dwell),
                ("energy", energy)
            ]:
                ds = self._grp[name]
                ds.resize((i + 1,))
                ds[i] = value

            ds = self._grp["target"]
            ds.resize((i + 1, 2))
            ds[i] = params["target_position"]

        else:
            traj = self._build_spiral(d)
            hit, t_hit, t_tot, dwell, energy = self._simulate(
                traj, params["target_position"], d
            )

        return ExperimentResult(
            hit=hit,
            time_to_hit=t_hit,
            total_time=t_tot,
            dwell_time=dwell,
            energy=energy,
            target=params["target_position"],
            trajectory=traj,
            physics=d,
            warnings=self.warnings.copy(),
            valid=True
        )
