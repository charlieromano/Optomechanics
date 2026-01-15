import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Model:
    def simulate(self, params):
        """
        params: dict
        returns:
            time: 1D array
            space: 1D array or None
            field: ndarray [time, space] or [time]
        """
        raise NotImplementedError
# ============================================================
#  Example Model
# ============================================================

class ExampleDynamicModel:
    def simulate(self, params):
        t = np.linspace(0, 10, 400)
        A = params["A"]
        b = params["b"]
        y = A * np.exp(-b * t)
        return t, None, y
# ============================================================
#  Acquisition Model: 1. Path
# ============================================================

class Path(ABC):
    @abstractmethod
    def trajectory(self):
        pass

    @abstractmethod
    def time_step(self):
        pass


class SpiralPath(Path):
    def __init__(self, spiral_step, max_radius, velocity):
        self.spiral_step = spiral_step
        self.max_radius = max_radius
        self.velocity = velocity

    def trajectory(self):
        theta = 0.0
        points = []

        while True:
            r = self.spiral_step * theta
            if r > self.max_radius:
                break
            points.append([r * np.cos(theta), r * np.sin(theta)])
            theta += self.spiral_step

        return np.array(points)

    def time_step(self):
        return self.spiral_step / self.velocity

# ============================================================
#  Acquisition Model: Model
# ============================================================

class AcquisitionModel:
    def __init__(self, hit_radius):
        self.hit_radius = hit_radius

    def run(self, params):
        # Build path from parameters
        path = SpiralPath(
            spiral_step=params["spiral_step"],
            max_radius=params["max_radius"],
            velocity=params["velocity"]
        )

        target = params["target_position"]
        irradiance = params["irradiance"]
        sensor_area = params["sensor_area"]

        traj = path.trajectory()
        dt = path.time_step()

        time = 0.0
        dwell_time = 0.0
        energy = 0.0
        hit = False
        time_to_hit = np.nan

        for p in traj:
            if np.linalg.norm(p - target) <= self.hit_radius:
                if not hit:
                    hit = True
                    time_to_hit = time
                dwell_time += dt
                energy += irradiance * sensor_area * dt
            time += dt

        return ExperimentResult(
            target=target,
            trajectory=traj,
            hit=hit,
            time_to_hit=time_to_hit,
            total_time=time,
            dwell_time=dwell_time,
            energy=energy
        )

# ============================================================
#  Acquisition Model: Experiment Result
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
        energy
    ):
        self.target = target
        self.trajectory = trajectory
        self.hit = hit
        self.time_to_hit = time_to_hit
        self.total_time = total_time
        self.dwell_time = dwell_time
        self.energy = energy
    def plot_geometry(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], 'k-', lw=1)
        color = 'green' if self.hit else 'red'
        ax.scatter(*self.target, c=color, s=50, zorder=3)

        ax.set_aspect("equal")
        ax.set_title(
            f"{'HIT' if self.hit else 'MISS'} | "
            f"t = {self.time_to_hit:.2f}s"
        )
        ax.grid(True)
        return ax

# ============================================================
