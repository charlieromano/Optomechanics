# MonteCarloModel.py
import math
import numpy as np
from numba import njit

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
import h5py

# MonteCarloModel.py or a separate utils.py
class Unit:
    """Simple wrapper for units (for display/validation purposes)."""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Unit({self.name!r})"

# Auxiliary class for derived physics
class AcquisitionDerived:
    def __init__(self, spot_radius, theta_fou, spiral_a, theta_max, num_turns,
                 step_length, dt, energy_per_dwell, irradiance):
        self.spot_radius = spot_radius
        self.theta_fou = theta_fou
        self.spiral_a = spiral_a
        self.theta_max = theta_max
        self.num_turns = num_turns
        self.step_length = step_length
        self.dt = dt
        self.energy_per_dwell = energy_per_dwell
        self.irradiance = irradiance

# ExperimentResult stores outputs of a single run
class ExperimentResult:
    def __init__(self, hit, time_to_hit, total_time, dwell_time, energy,
                 first_hit_pos=None, trajectory=None, physics=None, target=None):
        self.hit = hit
        self.time_to_hit = time_to_hit
        self.total_time = total_time
        self.dwell_time = dwell_time
        self.energy = energy
        self.first_hit_pos = first_hit_pos
        self.trajectory = trajectory
        self.physics = physics
        self.target = target

    @staticmethod
    def invalid(errors=None, warnings=None):
        return ExperimentResult(hit=False, time_to_hit=np.nan, total_time=0.0,
                                dwell_time=0.0, energy=0.0,
                                first_hit_pos=None, trajectory=None,
                                physics=None, target=None)

# ----------------------------

class AcquisitionModel:
    def __init__(self, backend="python", *, hdf5_file="mc_results.h5"):
        self.errors = []
        self.warnings = []
        self.backend = backend
        self._reference_trajectory = None
        self._last_d = None
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
        sigma_theta = self._get(p, "sigma_theta")
        theta_div = self._get(p, "theta_div")
        N_sigma = self._get(p, "N_sigma")
        overlap = self._get(p, "overlap_factor")
        velocity = self._get(p, "velocity")
        dwell_time = self._get(p, "dwell_time")
        power = self._get(p, "power")

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
            irradiance=irradiance
        )

    # ----------------------------
    # Validate physics
    # ----------------------------
    def _validate(self, p, d):
        self.errors.clear()
        self.warnings.clear()

        energy_threshold = self._get(p, "energy_threshold")
        if d.energy_per_dwell < energy_threshold:
            self.errors.append("Energy per dwell below threshold")

        theta_div = self._get(p, "theta_div")
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
    def _spiral_generator(self, d):
        n = int(d.theta_max / d.spiral_a)
        dtheta = d.theta_max / max(n - 1, 1)
        for i in range(n):
            theta = i * dtheta
            r = d.spiral_a * theta
            yield r * math.cos(theta), r * math.sin(theta)

    def _build_spiral(self, d, max_points=None):
        pts = np.array(list(self._spiral_generator(d)))
        if max_points and len(pts) > max_points:
            pts = pts[:: len(pts) // max_points]
        return pts
    @staticmethod
    @njit
    def _simulate_numba_kernel(traj, target, dt, spot_radius, energy_per_dwell):
        time = 0.0
        dwell = 0.0
        energy = 0.0
        hit = False
        t_hit = np.nan
        hit_index = -1
        r2 = spot_radius**2
        for idx in range(traj.shape[0]):
            dx = traj[idx,0] - target[0]
            dy = traj[idx,1] - target[1]
            if dx*dx + dy*dy <= r2:
                if not hit:
                    hit = True
                    t_hit = time
                    hit_index = idx
                dwell += dt
                energy += energy_per_dwell
            time += dt
        return hit, t_hit, time, dwell, energy, hit_index

    def _simulate_python(self, traj, target, d):
        time = 0.0
        dwell_time = 0.0
        energy = 0.0
        hit = False
        t_hit = np.nan
        first_hit_pos = None
        r2 = d.spot_radius**2
        for x, y in traj:
            dx, dy = x - target[0], y - target[1]
            if dx*dx + dy*dy <= r2:
                if not hit:
                    hit = True
                    t_hit = time
                    first_hit_pos = np.array([x, y])
                dwell_time += d.dt
                energy += d.energy_per_dwell
            time += d.dt
        return hit, t_hit, time, dwell_time, energy, first_hit_pos

    def _simulate_streaming(self, target, d):
        """Streaming simulation for HDF5 backend."""
        time = 0.0
        dwell = 0.0
        energy = 0.0
        hit = False
        time_to_hit = np.nan
        first_hit_pos = None
        r2 = d.spot_radius**2

        for x, y in self._spiral_generator(d):
            dx = x - target[0]
            dy = y - target[1]
            if dx*dx + dy*dy <= r2:
                if not hit:
                    hit = True
                    time_to_hit = time
                    first_hit_pos = np.array([x, y])
                dwell += d.dt
                energy += d.energy_per_dwell
            time += d.dt

        return hit, time_to_hit, time, dwell, energy, first_hit_pos

    def _simulate(self, traj, target, d):
        if self.backend == "numba":
            hit, t_hit, t_tot, dwell, energy, hit_idx = AcquisitionModel._simulate_numba_kernel(
                traj, target, d.dt, d.spot_radius, d.energy_per_dwell
            )
            first_hit_pos = traj[hit_idx].copy() if hit_idx >= 0 else None
            return hit, t_hit, t_tot, dwell, energy, first_hit_pos

        elif self.backend == "hdf5":
            # HDF5 streaming generates its own trajectory internally
            return self._simulate_streaming(target, d)

        else:  # python backend
            return self._simulate_python(traj, target, d)


    # ----------------------------
    # Public API
    # ----------------------------
    def run(self, params):
        d = self._derive(params)
        self._last_d = d  # store for plotting/reference

        if not self._validate(params, d):
            return ExperimentResult.invalid(
                errors=self.errors.copy(),
                warnings=self.warnings.copy()
            )

        # Build trajectory for Python / Numba backends
        traj = None
        if self.backend != "hdf5":
            traj = self._build_spiral(d)
        else:
            # Ensure reference trajectory exists for HDF5
            if self._reference_trajectory is None:
                self._reference_trajectory = np.array(
                    list(self._spiral_generator(d)), dtype=np.float32
                )

        # Dispatch simulation
        hit, t_hit, t_tot, dwell, energy, first_hit_pos = self._simulate(
            traj, params["target_position"], d
        )

        return ExperimentResult(
            hit=hit,
            time_to_hit=t_hit,
            total_time=t_tot,
            dwell_time=dwell,
            energy=energy,
            first_hit_pos=first_hit_pos,
            trajectory=traj if traj is not None else None,
            physics=d,
            target=params["target_position"]
        )


    # ----------------------------
    def close(self):
        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None

    def get_last_physics(self):
        return self._reference_trajectory, getattr(self, "_last_d", None)