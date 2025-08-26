import numpy as np
import matplotlib.pyplot as plt

# Parameters
D = 1.0       # Distance between means
Noise = 0.2   # Higher Noise -> flatter curves

# Gaussian function
def gaussian(x, mu, noise):
    sigma = 0.2 + noise   # baseline width + noise factor
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

# X axis range
x = np.linspace(-2, 3, 500)

# Two Gaussians
g1 = gaussian(x, 0, Noise)
g2 = gaussian(x, D, Noise)

# Plot
plt.figure(figsize=(6,4))

plt.plot(x, g1, 'k-', linewidth=2, label="Mean = 0")   # solid thick black
plt.plot(x, g2, 'k--', linewidth=1.5, label=f"Mean = {D}") # dashed thinner black

# Labels at means
plt.axvline(0, color='k', linestyle=':', linewidth=1)
plt.axvline(D, color='k', linestyle=':', linewidth=1)

# Annotate mean values
plt.text(0, 1.05, "0", ha='center', va='bottom', fontsize=12)
plt.text(D, 1.05, f"{D:.1f}", ha='center', va='bottom', fontsize=12)

# Axis labels
plt.xlabel("x", fontsize=12)
plt.ylabel("Probability density (a.u.)", fontsize=12)

# Style: academic sober look
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
