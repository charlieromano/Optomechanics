import numpy as np
from MCParameter import Parameter
from MCSampler import JointParameterSampler
from MCEngine import MonteCarloSimulation
from MCAnalyzer import ResultAnalyzer

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

target_position = Parameter(
    name="target_position",
    kind="distribution",
    dist="Gaussian2D",
    mean=[0.0, 0.0],
    cov=[
        [theta_FoU**2, 0.0],
        [0.0, theta_FoU**2]
    ],
    units="rad",
    description="Target position within field of uncertainty"
)

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
    

# ============================================================
# 3. Sampler
# ============================================================
class Gaussian2DSampler:
    def __init__(self, mean, cov, n_points):
        self.mean = mean
        self.cov = cov
        self.n_points = n_points
    def sample(self):
        return np.random.multivariate_normal(
            self.mean,
            self.cov,
            self.n_points
        )


# ============================================================
# 4. Engine
# ============================================================
# Simulation sample size
N = 100
spiral_pattern = Parameter(
    name="spiral_pattern",
    kind="fixed",
    value="Archimedean",
    description="Type of spiral search pattern"
)

# ============================================================
# 5. Analyzer
# ============================================================
