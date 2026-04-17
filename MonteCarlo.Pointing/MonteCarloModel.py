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
    def __init__(self):
        # We will assign attributes dynamically in _derive
        pass

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
        d = AcquisitionDerived()
        
        # 1. Base Parameter Extraction
        theta_div = self._get(p, "theta_div")
        velocity = self._get(p, "velocity")
        rx_power = self._get(p, "power")  # P_rx in Watts
        mode = self._get(p, "scan_mode")
        sim_res = self._get(p, "simulation_resolution")
        p_dwell = self._get(p, "dwell_time")
        
        # 2. Geometry
        d.spot_radius = theta_div / 2.0
        d.theta_fou = self._get(p, "N_sigma") * self._get(p, "sigma_theta")
        d.spiral_a = (theta_div * (1.0 - self._get(p, "overlap_factor"))) / (2.0 * np.pi)
        
        # Add these to avoid AttributeErrors if your generator or analyzer needs them
        d.theta_max = d.theta_fou / d.spiral_a if d.spiral_a != 0 else 0
        d.num_turns = d.theta_max / (2.0 * np.pi)
        
        # 3. Energy Physics
        d.rx_power = rx_power
        d.energy_threshold = self._get(p, "energy_threshold")
        
        # Physical Irradiance (W/m^2)
        receiver_diameter = self._get(p, "receiver_diameter")
        receiver_area = np.pi * (receiver_diameter / 2.0)**2
        d.irradiance = rx_power / receiver_area
        
        # 4. Mode-Specific Timing Logic
        d.mode = mode
        if mode == "continuous":
            d.dwell_time = theta_div / velocity 
            d.dt = sim_res
        else: # stare_step
            d.dwell_time = p_dwell
            d.dt = p_dwell
            
        # 5. Simulation Helpers
        d.step_length = velocity * d.dt
        d.energy_per_dwell = rx_power * d.dwell_time
        d.velocity = velocity
        d.simulation_resolution = sim_res

        # JUST RETURN THE OBJECT d
        return d
    # ----------------------------
    # Validate physics
    # ----------------------------
    def _validate(self, p, d):
        self.errors.clear()
        self.warnings.clear()
        mode = self._get(p, "scan_mode")

        # 1. Energy and Power Consistency Check
        energy_threshold = self._get(p, "energy_threshold")      
        
        # Validation: Energy requirement based on integrated power over dwell time
        if d.energy_per_dwell < energy_threshold:
            self.errors.append(f"Energy link budget fail: {d.energy_per_dwell:.2e}J < {energy_threshold:.2e}J")
        
        # Physical Check: Ensure Irradiance * Area matches the Power used for energy calcs
        expected_power = d.irradiance * (math.pi * (self._get(p, "receiver_diameter") / 2.0)**2)
        if not math.isclose(expected_power, self._get(p, "power"), rel_tol=1e-5):
            self.warnings.append("Irradiance and Receiver Power are numerically inconsistent")

        # 2. Sampling Check (Mode dependent)
        theta_div = self._get(p, "theta_div")
        overlap = self._get(p, "overlap_factor")
        
        if mode == "continuous":
            # Nyquist-like check: must sample at least twice per spot diameter
            if d.step_length > (theta_div / 2.0):
                self.errors.append("Velocity too high for simulation resolution (undersampling)")
        
        # 3. Geometric Coverage
        radial_spacing = 2.0 * math.pi * d.spiral_a
        max_allowed_gap = theta_div * (1.0 - overlap)
        
        if radial_spacing > theta_div:
            self.errors.append("Radial gaps detected: Spiral pitch exceeds beam diameter")
        elif radial_spacing > max_allowed_gap + 1e-9: # tiny epsilon for float math
            self.warnings.append("Overlap requirement not met in radial direction")

        # 4. Statistical Coverage
        if self._get(p, "N_sigma") < 3.0:
            self.warnings.append("Target distribution truncated (N_sigma < 3)")

        return len(self.errors) == 0
    # ----------------------------
    # Geometry & Timing
    # ----------------------------
    def compute_continuous_scan_time(self, d):
        """Calculates total time required to traverse the spiral path at constant velocity."""
        # L = 0.5 * a * theta_max^2 (Arc length approximation for Archimedean spiral)
        total_length = 0.5 * d.spiral_a * (d.theta_max**2)
        return total_length / d.velocity

    def _generator(self, d, mode="continuous"):
        """
        Generates (x, y, time) coordinates for the beam center.
        - Continuous: Constant tangential velocity, sampled at simulation_resolution.
        - Stare-Step: Discrete jumps with velocity-limited slew time + fixed dwell.
        """
        if mode == "continuous":
            t_tot = self.compute_continuous_scan_time(d)
            n_steps = int(t_tot / d.simulation_resolution)
            for i in range(n_steps + 1):
                t = i * d.simulation_resolution
                # Solve s = v*t = 0.5 * a * theta^2  => theta = sqrt(2vt/a)
                theta = math.sqrt((2.0 * d.velocity * t) / d.spiral_a)
                r = d.spiral_a * theta
                
                yield r * math.cos(theta), r * math.sin(theta), t
        elif mode == "stare_step":
            current_time = 0.0
            prev_pos = (0.0, 0.0)
            # Distance between rings is 2*pi*a. To maintain overlap k in the step direction,
            # we move by the same distance along the arc.
            # Step size = theta_div * (1 - overlap)
            arc_step = 2.0 * math.pi * d.spiral_a 
            # Using the same arc length relation s = 0.5 * a * theta^2
            # Total distance L = 0.5 * a * theta_max^2
            total_dist = 0.5 * d.spiral_a * (d.theta_max**2)
            num_steps = int(total_dist / arc_step)
            for i in range(num_steps + 1):
                # Calculate theta for this discrete step
                s = i * arc_step
                theta = math.sqrt((2.0 * s) / d.spiral_a)
                r = d.spiral_a * theta
                curr_pos = (r * math.cos(theta), r * math.sin(theta))
                # Calculate Slew (Jump) Time
                dist = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                jump_time = dist / d.velocity # velocity represents maximum slew speed here
                # Total time = previous + movement + pause
                current_time += jump_time + d.dwell_time
                yield curr_pos[0], curr_pos[1], current_time
                prev_pos = curr_pos

    def _build_spiral(self, d, mode="continuous", max_points=None):
        """Helper to collect generator output into a numpy array for visualization or analysis."""
        # We extract only (x, y) for the trajectory array, but keep t available if needed
        data = list(self._generator(d, mode=mode))
        pts = np.array([(p[0], p[1]) for p in data])
        if max_points and len(pts) > max_points:
            step = len(pts) // max_points
            pts = pts[::step]
        return pts

    # @staticmethod
    # @njit
    # def _simulate_numba_kernel(traj, target, dt, spot_radius, energy_per_dwell):
    #     time = 0.0
    #     dwell = 0.0
    #     energy = 0.0
    #     hit = False
    #     t_hit = np.nan
    #     hit_index = -1
    #     r2 = spot_radius**2
    #     for idx in range(traj.shape[0]):
    #         dx = traj[idx,0] - target[0]
    #         dy = traj[idx,1] - target[1]
    #         if dx*dx + dy*dy <= r2:
    #             if not hit:
    #                 hit = True
    #                 t_hit = time
    #                 hit_index = idx
    #             dwell += dt
    #             energy += energy_per_dwell
    #         time += dt
    #     return hit, t_hit, time, dwell, energy, hit_index
    # @staticmethod
    # @njit
    # def _simulate_numba_kernel(traj, target, dt, spot_radius, irradiance, dwell_threshold):
    #     time = 0.0
    #     dwell = 0.0
    #     energy = 0.0
    #     hit = False
    #     t_hit = np.nan
    #     hit_index = -1
    #     r2 = spot_radius * spot_radius
    #     for idx in range(traj.shape[0]):
    #         dx = traj[idx, 0] - target[0]
    #         dy = traj[idx, 1] - target[1]
    #         inside = dx*dx + dy*dy <= r2
    #         # if inside:
    #         #     dwell += dt
    #         #     energy += irradiance * dt
    #         #     if hit_index == -1:
    #         #         hit_index = idx
    #         #     if (not hit) and (dwell >= dwell_threshold):
    #         #         hit = True
    #         #         t_hit = time
    #         if inside:
    #             current_streak += dt
    #             if current_streak >= dwell_threshold and not hit:
    #                 hit = True
    #                 t_hit = time
    #         else:
    #             current_streak = 0 # Reset if the beam leaves the target
    #         time += dt
    #     return hit, t_hit, time, dwell, energy, hit_index   
    @staticmethod
    @njit
    def _simulate_numba_kernel(traj, target, dt, spot_radius, irradiance, dwell_threshold):
        time = 0.0
        dwell = 0.0
        energy = 0.0
        hit = False
        t_hit = np.nan
        hit_index = -1
        current_streak = 0.0  # tracks continuous time inside the target
        r2 = spot_radius * spot_radius
        for idx in range(traj.shape[0]):
            dx = traj[idx, 0] - target[0]
            dy = traj[idx, 1] - target[1]
            inside = dx*dx + dy*dy <= r2
            if inside:
                dwell += dt
                energy += irradiance * dt
                if hit_index == -1:
                    hit_index = idx
                current_streak += dt
                if (not hit) and (current_streak >= dwell_threshold):
                    hit = True
                    t_hit = time
            else:
                current_streak = 0.0
            time += dt
        return hit, t_hit, time, dwell, energy, hit_index
    # def _simulate_python(self, traj, target, d):
    #     time = 0.0
    #     dwell_time = 0.0
    #     energy = 0.0
    #     hit = False
    #     t_hit = np.nan
    #     first_hit_pos = None
    #     r2 = d.spot_radius**2
    #     for x, y in traj:
    #         dx, dy = x - target[0], y - target[1]
    #         if dx*dx + dy*dy <= r2:
    #             if not hit:
    #                 hit = True
    #                 t_hit = time
    #                 first_hit_pos = np.array([x, y])
    #             dwell_time += d.dt
    #             energy += d.energy_per_dwell
    #         time += d.dt
    #     return hit, t_hit, time, dwell_time, energy, first_hit_pos
    # def _simulate_python(self, traj, target, d):
    #     def segment_circle_time(p1, p2, center, r, dt):
    #         x1, y1 = p1[0] - center[0], p1[1] - center[1]
    #         x2, y2 = p2[0] - center[0], p2[1] - center[1]
    #         dx = x2 - x1
    #         dy = y2 - y1
    #         a = dx*dx + dy*dy
    #         if a == 0:
    #             return 0.0
    #         b = 2.0 * (x1*dx + y1*dy)
    #         c = x1*x1 + y1*y1 - r*r
    #         disc = b*b - 4*a*c
    #         if disc <= 0:
    #             return 0.0
    #         sqrt_d = np.sqrt(disc)
    #         t1 = (-b - sqrt_d) / (2*a)
    #         t2 = (-b + sqrt_d) / (2*a)
    #         t_in = max(0.0, min(t1, t2))
    #         t_out = min(1.0, max(t1, t2))
    #         if t_out <= 0 or t_in >= 1:
    #             return 0.0
    #         return (t_out - t_in) * dt
    #     time = 0.0
    #     dwell_time = 0.0
    #     energy = 0.0
    #     hit = False
    #     t_hit = np.nan
    #     first_hit_pos = None
    #     r = d.spot_radius
    #     dt = d.dt
    #     threshold = d.dwell_time
    #     for i in range(len(traj) - 1):
    #         p1 = traj[i]
    #         p2 = traj[i + 1]
    #         dt_overlap = segment_circle_time(p1, p2, target, r, dt)
    #         if dt_overlap > 0.0:
    #             # accumulate physical interaction time
    #             dwell_time += dt_overlap
    #             # energy = irradiance × time overlap
    #             energy += d.irradiance * dt_overlap
    #             # first hit detection
    #             if first_hit_pos is None:
    #                 first_hit_pos = p1.copy()
    #             # hit condition (continuous)
    #             if (not hit) and (dwell_time >= threshold):
    #                 hit = True
    #                 t_hit = time
    #         time += dt
    #     return hit, t_hit, time, dwell_time, energy, first_hit_pos
    def _simulate_python(self, target, d):
        def segment_circle_time(p1, p2, center, r):
            """Calculates how much time during a segment p1->p2 the beam center is inside r."""
            # Vector from segment start to target
            x1, y1 = p1[0] - center[0], p1[1] - center[1]
            x2, y2 = p2[0] - center[0], p2[1] - center[1]
            dx, dy = x2 - x1, y2 - y1            
            # Quadratic coefficients for intersection of line and circle
            a = dx*dx + dy*dy
            if a == 0: return 0.0 # No movement
            b = 2.0 * (x1*dx + y1*dy)
            c = x1*x1 + y1*y1 - r*r
            disc = b*b - 4*a*c
            if disc <= 0: return 0.0 # No intersection
            sqrt_d = math.sqrt(disc)
            t1 = (-b - sqrt_d) / (2*a)
            t2 = (-b + sqrt_d) / (2*a)
            # Entry and exit points relative to the segment [0, 1]
            t_in = max(0.0, min(t1, t2))
            t_out = min(1.0, max(t1, t2))
            if t_out <= t_in or t_out <= 0 or t_in >= 1:
                return 0.0
            # Total segment duration
            segment_dt = p2[2] - p1[2]
            return (t_out - t_in) * segment_dt

        accumulated_energy = 0.0
        hit = False
        t_hit = np.nan
        first_hit_pos = None

        gen = self._generator(d, mode=d.mode)
        prev = next(gen)

        for curr in gen:
            # 1. Calculate how much time the beam aperture overlaps the target 
            # during this specific simulation step
            dt_overlap = segment_circle_time(prev, curr, target, d.spot_radius)
            
            # In _simulate_python
            dt_overlap = segment_circle_time(prev, curr, target, d.spot_radius)
            if dt_overlap > 0:
                accumulated_energy += d.rx_power * dt_overlap # Watts * Seconds = Joules
                
                if not hit and accumulated_energy >= d.energy_threshold:
                    hit = True
                    t_hit = curr[2]
                # 2. Record the first time the beam touches the target (for the plot)
                if first_hit_pos is None:
                    first_hit_pos = (curr[0], curr[1])
                
                # 3. Integrate Energy: Energy = Power [W] * Time [s]
                accumulated_energy += d.rx_power * dt_overlap

                # 4. Physical Trigger: Does total energy meet the detector threshold?
                if not hit and accumulated_energy >= d.energy_threshold:
                    hit = True
                    t_hit = curr[2] # Current time in simulation
                    
            prev = curr
            
        return hit, t_hit, prev[2], 0.0, accumulated_energy, first_hit_pos
    # def _simulate_streaming(self, target, d):
    #     """Streaming simulation for HDF5 backend."""
    #     time = 0.0
    #     dwell = 0.0
    #     energy = 0.0
    #     hit = False
    #     time_to_hit = np.nan
    #     first_hit_pos = None
    #     r2 = d.spot_radius**2
    #     for x, y in self._spiral_generator(d):
    #         dx = x - target[0]
    #         dy = y - target[1]
    #         if dx*dx + dy*dy <= r2:
    #             if not hit:
    #                 hit = True
    #                 time_to_hit = time
    #                 first_hit_pos = np.array([x, y])
    #             dwell += d.dt
    #             energy += d.energy_per_dwell
    #         time += d.dt
    #     return hit, time_to_hit, time, dwell, energy, first_hit_pos
    def _simulate_streaming(self, target, d):
        time = 0.0
        dwell = 0.0
        energy = 0.0
        hit = False
        t_hit = np.nan
        first_hit_pos = None
        r2 = d.spot_radius * d.spot_radius
        dt = d.dt
        threshold = d.dwell_time
        prev = None
        for x, y in self._spiral_generator(d):
            dx = x - target[0]
            dy = y - target[1]
            inside = dx*dx + dy*dy <= r2
            if inside:
                dwell += dt
                energy += d.irradiance * dt
                if first_hit_pos is None:
                    first_hit_pos = np.array([x, y])
                if (not hit) and (dwell >= threshold):
                    hit = True
                    t_hit = time
            prev = (x, y)
            time += dt
        return hit, t_hit, time, dwell, energy, first_hit_pos

    # def _simulate(self, traj, target, d):
    #     if self.backend == "numba":
    #         hit, t_hit, t_tot, dwell, energy, hit_idx = AcquisitionModel._simulate_numba_kernel(
    #             traj, target, d.dt, d.spot_radius, d.irradiance, d.dwell_time
    #         )
    #         first_hit_pos = traj[hit_idx].copy() if hit_idx >= 0 else None
    #         return hit, t_hit, t_tot, dwell, energy, first_hit_pos
    #     elif self.backend == "hdf5":
    #         # HDF5 streaming generates its own trajectory internally
    #         return self._simulate_streaming(target, d)
    #     else:  # python backend
    #         return self._simulate_python(target, d)
    def _simulate(self, traj, target, d):
        # Python backend uses its own internal generator (no traj needed)
        if self.backend == "python":
            return self._simulate_python(target, d)
            
        # Numba backend requires the pre-built trajectory array
        elif self.backend == "numba":
            hit, t_hit, t_tot, dwell, energy, hit_idx = AcquisitionModel._simulate_numba_kernel(
                traj, target, d.dt, d.spot_radius, d.irradiance, d.dwell_time
            )
            first_hit_pos = traj[hit_idx].copy() if hit_idx >= 0 else None
            return hit, t_hit, t_tot, dwell, energy, first_hit_pos
            
        # HDF5 and others
        elif self.backend == "hdf5":
            return self._simulate_streaming(target, d)
            
        else:
            return self._simulate_python(target, d)

    # ----------------------------
    # Public API
    # ----------------------------
    def run(self, params):
        d = self._derive(params)
        self._last_d = d 

        if not self._validate(params, d):
            return ExperimentResult.invalid(errors=self.errors.copy(), warnings=self.warnings.copy())

        # Build trajectory ONLY for backends that require a pre-computed array
        mode = params.get("scan_mode", "continuous")
        traj = None
        if self.backend == "numba":
            traj = self._build_spiral(d, mode=mode)

        # Dispatch simulation - notice we only pass traj as a potential argument
        hit, t_hit, t_tot, dwell, energy, first_hit_pos = self._simulate(
            traj, params["target_position"], d
        )

        return ExperimentResult(
            hit=hit, time_to_hit=t_hit, total_time=t_tot,
            dwell_time=dwell, energy=energy, first_hit_pos=first_hit_pos,
            trajectory=traj, physics=d, target=params["target_position"]
        )


    # ----------------------------
    def close(self):
        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None

    def get_last_physics(self):
        return self._reference_trajectory, getattr(self, "_last_d", None)