import numpy as np
import matplotlib.pyplot as plt

# Parameters
D = 1.0       # Distance between means
Noise = 0.2   # Higher Noise -> flatter curves

# Data
distance = np.array([0, 2, 20, 200, 2000, 5000, 12000, 35000])
I_0 = np.array([2.69E+00,0.098019088,0.000609307,5.84037E-06,5.81595E-08,9.30292E-09,1.61492E-09,1.89825E-10])


# Plot
plt.figure(figsize=(6,4))
plt.semilogy(distance, I_0, 'kx', linewidth=2, label="Mean = 0")   
plt.axvline(0, color='k', linestyle=':', linewidth=1)
plt.axhline(1e-8, color='k', linestyle=':', linewidth=1)
plt.text(distance[4], I_0[3], "LEO", ha='center', va='bottom', fontsize=12)
plt.text(distance[5], I_0[7], "MEO", ha='center', va='bottom', fontsize=12)
plt.text(distance[-1], 1.8e-10, "GEO", ha='center', va='bottom', fontsize=12)
plt.xlabel("Distance [km]", fontsize=12)
plt.ylabel("on-axis irradiance ratio", fontsize=12)
plt.tight_layout()
plt.grid()
plt.show()
