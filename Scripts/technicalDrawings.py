import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Setup figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Coordinate system
axis_len = 10
ax.quiver(0, 0, 0, axis_len, 0, 0, color='k', arrow_length_ratio=0.1)
ax.text(axis_len, 0, 0, 'x', color='k')
ax.quiver(0, 0, 0, 0, axis_len, 0, color='k', arrow_length_ratio=0.1)
ax.text(0, axis_len, 0, 'y', color='k')
ax.quiver(0, 0, 0, 0, 0, axis_len, color='k', arrow_length_ratio=0.1)
ax.text(0, 0, axis_len, 'z', color='k')

# Observer (ground station) at origin
observer_radius = 0.3
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:10j]
x_sphere = observer_radius * np.cos(u) * np.sin(v)
y_sphere = observer_radius * np.sin(u) * np.sin(v)
z_sphere = observer_radius * np.cos(v)
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='grey', alpha=0.1)

# Optical gimbal as concentric circles (simplified)
circle_radius_outer = 0.5
circle_radius_inner = 0.3
theta = np.linspace(0, 2 * np.pi, 100)
ax.plot(circle_radius_outer * np.cos(theta), circle_radius_outer * np.sin(theta), 0, 'k')
ax.plot(circle_radius_inner * np.cos(theta), circle_radius_inner * np.sin(theta), 0, 'k')

# Object positions and trajectory
v_speed = 1
t = 5
L = 6  # altitude
A = np.array([-v_speed*t, 0, L])
B = np.array([0, 0, L])
C = np.array([v_speed*t, 0, L])
path_x = np.linspace(A[0], C[0], 300)
path_y = np.zeros_like(path_x)
path_z = np.full_like(path_x, L)
ax.plot(path_x, path_y, path_z, 'k--', label='Object Path')

# Object velocity vector at B
ax.quiver(B[0], B[1], B[2], v_speed, 0, 0, color='blue', arrow_length_ratio=0.1)
ax.text(B[0]+0.5, B[1], B[2], 'v', color='blue')

# Zenith angle (arc) in XZ plane
angle_range = np.linspace(0, np.arctan(L / (v_speed * t)), 100)
arc_radius = 2
arc_x = arc_radius * np.sin(angle_range)
arc_z = arc_radius * np.cos(angle_range)
ax.plot(arc_x, np.zeros_like(arc_x), arc_z, 'r')
ax.text(arc_x[-1], 0, arc_z[-1]+0.2, r'$\theta_z$', color='r')

# Zenith angular speed (omega_z) - curved arrow near the observer
omega_arc_radius = 1
omega_angle = np.linspace(0, np.pi / 6, 50)
omega_x = omega_arc_radius * np.sin(omega_angle)
omega_z = omega_arc_radius * np.cos(omega_angle)
ax.plot(omega_x, np.zeros_like(omega_x), omega_z, 'g')
ax.text(omega_x[-1], 0, omega_z[-1]+0.2, r'$\omega_z$', color='g')

# Set limits and labels
ax.set_xlim(-7, 7)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 10)
ax.set_box_aspect([1.5, 0.6, 1])  # better aspect ratio
ax.view_init(elev=25, azim=-60)
ax.axis('off')

plt.tight_layout()
plt.show()

