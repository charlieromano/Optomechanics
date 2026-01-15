
P_tx = 1  # Transmitted Power in Watts
lambda_Tx = 1550e-9  # Wavelength in meters
D_tx = 0.8  # Transmitter Aperture Diameter in meters
D_rx = 0.1  # Receiver Aperture Diameter in meters
L = 1000e3  # Link Distance in meters
T_dw = 0.2  # Dwell time in seconds
E_th = 0.1e-9  # Energy Threshold in Joules

from math import pi, sqrt, log as ln
import numpy as np
import matplotlib.pyplot as plt


#theta_div = 1.22 * (lambda_Tx / D_tx)
theta_div = 150e-6  # Divergence Angle in radians
theta_div = np.arange(theta_div, theta_div+0.01, 0.00001)
A_rx = pi * (D_rx / 2.0)**2


#theta_eff = np.array(sqrt(0.5*theta_div**2*ln((2*T_dw*P_tx*A_rx)/(pi*L**2*theta_div**2*E_th))))
theta_eff = np.sqrt(0.5 * theta_div**2 * np.log((2*T_dw*P_tx*A_rx) / (pi*L**2 * theta_div**2 * E_th)))

plt.plot(theta_eff)
plt.xlabel('Divergence Angle (rad)')
plt.ylabel('Effective Angle (rad)')
plt.title('Effective Angle vs Divergence Angle')
plt.show()