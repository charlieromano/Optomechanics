import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import numpy as np

class Path(ABC):
    @abstractmethod
    def trajectory(self):
        pass

    @abstractmethod
    def time_step(self):
        pass

# ============================================================
# 1. Parameters
# ============================================================
sigma_open_loop = 4e-3 # rad
sigma_jitter = 0.04e-3 # rad
theta_FoU = 3*sigma_open_loop # rad

# Divergence angle limits
theta_div_min = 0.1e-3 # rad
theta_div_max = 3.0e-3 # rad
theta_div = 100e-6 # rad

# Spiral parameters
k = 1.25 # Overlap factor
theta_spiral_step = k*theta_div # rad

# Link distance
Link_Distance = 500e6 # meters

# Acquisition Energy threshold
E_threshold =  0.025 # Joules = Irradiance_threshold * sensor_area
acquistion_irradiance_min = 15e-6 # W/m^2 Eagle-1
acquistion_irradiance_max = 510e-6 # W/m^2 Eagle-1
P_Rx = 0.1e-6 # Watt

# Time parameters
T_acq_max = 300.0 # seconds
T_step = 1.0 # seconds
T_dwell_min = E_threshold / acquistion_irradiance_max # seconds
T_dwell_max = E_threshold / acquistion_irradiance_min # seconds

# Sensor size
sensor_size = 1e-4 # m^2 (1e-4m^2 = 1cm^2)
sensor_QE = 0.65 # Quantum efficiency

# Rx aperture diameter limits
D_Rx_min = 0.01 # m
D_Rx_max = 0.20 # m
# Tx Power limits
P_Tx = 2.5 # Watt
P_Tx_min = 1.0 # Watt
P_Tx_max = 4.0 # Watt
# Tx aperture diameter limits
D_Tx = 0.5 # m
D_Tx_min = 0.01 # m
D_Tx_max = 0.8 # m

# ============================================================
# 2. Model
# ============================================================
# Define parameters using MCParameter

class ScanPatternModel:
    def __init__(self, speed, radius):
        self.speed = speed
        self.radius = radius
    def generate_trajectory(self):
        theta = np.linspace(0, 2 * np.pi, 500)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        trajectory = np.vstack((x, y)).T
        length = 2 * np.pi * self.radius
        time = length / self.speed
        return trajectory, time
    
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
            points.append([r*np.cos(theta), r*np.sin(theta)])
            theta += self.spiral_step

        return np.array(points)

    def time_step(self):
        return self.spiral_step / self.velocity

class AcquisitionModel:
    def __init__(self, path, hit_radius):
        self.path = path
        self.hit_radius = hit_radius

    def run(self, target):
        traj = self.path.trajectory()
        dt = self.path.time_step()
        time = 0.0

        for p in traj:
            if np.linalg.norm(p - target) <= self.hit_radius:
                return {
                    "acquired": True,
                    "time_to_hit": time,
                    "trajectory": traj,
                    "target": target
                }
            time += dt

        return {
            "acquired": False,
            "time_to_hit": np.inf,
            "trajectory": traj,
            "target": target
        }

# ============================================================
# 3. Sampler
# ============================================================
class Gaussian2DSampler:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample(self, rng):
        return rng.multivariate_normal(self.mean, self.cov)

# ============================================================
# 4. Engine
# ============================================================

class MonteCarloEngine:
    def __init__(self, sampler, model, seed=None):
        self.sampler = sampler
        self.model = model
        self.rng = np.random.default_rng(seed)

    def run_once(self):
        target = self.sampler.sample(self.rng)
        return self.model.run(target)

    def run(self, n):
        return [self.run_once() for _ in range(n)]




# ============================================================
# 5. Analyzer
# ============================================================
class AcquisitionAnalyzer:
    def __init__(self, results):
        self.results = results

    def probability_of_acquisition(self):
        return np.mean([r["acquired"] for r in self.results])

    def acquisition_times(self):
        return np.array([
            r["time_to_hit"] for r in self.results
            if r["acquired"]
        ])


# ============================================================
# Main Execution
import matplotlib.pyplot as plt

path = SpiralPath(
    spiral_step=0.05,
    max_radius=3.0,
    velocity=0.1
)

sampler = Gaussian2DSampler(
    mean=[0, 0],
    cov=[[1, 0], [0, 1]]
)

model = AcquisitionModel(
    path=path,
    hit_radius=0.2
)

engine = MonteCarloEngine(
    sampler=sampler,
    model=model,
    seed=42
)

results = engine.run(100)
analyzer = AcquisitionAnalyzer(results)

print("P(acquisition):", analyzer.probability_of_acquisition())

# Plot one geometry
r = results[0]
plt.plot(r["trajectory"][:,0], r["trajectory"][:,1], 'k-')
plt.scatter(*r["target"], c='red')
plt.gca().set_aspect("equal")
plt.grid()
plt.show()
