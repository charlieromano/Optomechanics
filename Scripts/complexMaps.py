# import library
import matplotlib.pyplot as plt
import numpy as np

#z = [0+0j, 2j, -1+2j, 1+2j, -2+3j, 2+3j]
z = []
for r in [1, 2, 3, 5]:
	z += [r * np.exp(1j * theta) for theta in np.linspace(0, 2*np.pi, 20, endpoint=False)]

#w = [(zi - 1j) / (zi + 1j) for zi in z]
#w = [(2*zi + 1) / (zi - 3) for zi in z]
w = [(  3*zi + 2) / (zi + 1) for zi in z]

x = [ele.real for ele in z]
y = [ele.imag for ele in z]
s = [ele.real for ele in w]
t = [ele.imag for ele in w]

# plot the complex numbers
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x, y, color='black')
ax1.set_ylabel('Imaginary')
ax1.set_xlabel('Real')
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.set_aspect('equal')
ax1.grid(True)
ax2.scatter(s, t, color='blue')
ax2.set_ylabel('Imaginary')
ax2.set_xlabel('Real')
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.set_aspect('equal')
ax2.grid(True)
plt.show()
