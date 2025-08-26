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
x = np.linspace(-2, 3, 1000)

# Two Gaussians
g1 = gaussian(x, 0, Noise)
g2 = gaussian(x, D, Noise)

# --- Find intersection (between means) ---
idx = np.argwhere(np.diff(np.sign(g1 - g2))).flatten()
if len(idx) > 0:
    i = idx[0]
    # Linear interpolation for precision
    x_cross = x[i] - (g1[i] - g2[i]) * (x[i+1]-x[i]) / ((g1[i+1]-g2[i+1]) - (g1[i]-g2[i]))
    y_cross = gaussian(x_cross, D, Noise)
else:
    x_cross, y_cross = None, None

# Plot
plt.figure(figsize=(6,4))

# Two curves
plt.plot(x, g1, 'k-', linewidth=2, label="Mean = 0")   
plt.plot(x, g2, 'k-', linewidth=1.5, label=f"Mean = {D}") 

# Vertical lines at means
plt.axvline(0, color='k', linestyle=':', linewidth=1)
plt.axvline(D, color='k', linestyle=':', linewidth=1)

# Intersection line + shading for RIGHT bell
if x_cross is not None:
    plt.axvline(x_cross, color='k', linestyle='-.', linewidth=1)
    plt.text(x_cross, 1.05, f"x = {x_cross:.2f}", ha='center', va='bottom', fontsize=10)

    # Shade right bell up to intersection
    mask = x <= x_cross
    plt.fill_between(x[mask], g2[mask], color='none',
                     hatch='////', edgecolor='k', linewidth=0.0)

# Annotations at means
plt.text(0, 1.05, "0", ha='center', va='bottom', fontsize=12)
plt.text(D, 1.05, f"{D:.1f}", ha='center', va='bottom', fontsize=12)

# Labels and style
plt.xlabel("x", fontsize=12)
plt.ylabel("Probability density (a.u.)", fontsize=12)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
