import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class Path:
    def points(self):
        raise NotImplementedError
    def arc_length(self):
        pts = self.points()
        diffs = np.diff(pts, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))
    def sample_by_arc_length(self, s_values):
        pts = self.points()
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_len = np.concatenate([[0], np.cumsum(seg_lengths)])
        x = np.interp(s_values, cum_len, pts[:, 0])
        y = np.interp(s_values, cum_len, pts[:, 1])
        return np.column_stack((x, y))
    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            ax.grid(True)
        pts = self.points()
        ax.plot(pts[:, 0], pts[:, 1], **kwargs)
        return ax

class CirclePattern:
    def __init__(self, path, N, radius=None, overlap=0.0):
        if overlap < 0 or overlap >= 1:
            raise ValueError("overlap must be in [0, 1)")
        self.path = path
        self.N = N
        self.overlap = overlap
        self._radius = radius
    @property
    def radius(self):
        if self._radius is not None:
            return self._radius
        spacing = self.path.arc_length() / self.N
        return spacing / 2
    def centers(self):
        spacing = 2 * self.radius * (1 - self.overlap)
        s = np.linspace(0, spacing * (self.N - 1), self.N)
        return self.path.sample_by_arc_length(s)
    def noisy_centers(self, noise):
        return noise.apply(self.centers())
    def centers_as_path(self, noise=None):
        pts = self.centers() if noise is None else self.noisy_centers(noise)
        return DiscretePath(pts)
    def plot(self, ax=None, noise=None, **kwargs):
        """
        Plot circles along the path.
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            ax.grid(True)
        centers = self.centers() if noise is None else self.noisy_centers(noise)
        for c in centers:
            circle = Circle(c, self.radius, fill=False, **kwargs)
            ax.add_patch(circle)
        return ax

class Spiral(Path):
    def __init__(self, radius, num_points, num_rounds):
        self.radius = radius
        self.num_points = num_points
        self.num_rounds = num_rounds
    def points(self):
        theta = np.linspace(
            0,
            2 * np.pi * self.num_rounds,
            self.num_points
        )
        rho = (self.radius / (2 * np.pi * self.num_rounds)) * theta
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return np.column_stack((x, y))

class SpatialNoise:
    def apply(self, points):
        """
        Apply spatial noise to Nx2 points
        """
        raise NotImplementedError

class GaussianSpatialNoise(SpatialNoise):
    def __init__(self, mean=0.0, sigma=1.0):
        self.mean = mean
        self.sigma = sigma
    def apply(self, points):
        noise = np.random.normal(
            loc=self.mean,
            scale=self.sigma,
            size=points.shape
        )
        return points + noise

class NoisyPath(Path):
    def __init__(self, path, noise):
        self.path = path
        self.noise = noise
    def points(self):
        clean_pts = self.path.points()
        return self.noise.apply(clean_pts)

class DiscretePath(Path):
    def __init__(self, points):
        self._points = np.asarray(points)
    def points(self):
        return self._points



spiral = Spiral(
    radius=21,
    num_points=2000,
    num_rounds=3
)

circles = CirclePattern(
    path=spiral,
    radius=4,
    N=50,
    overlap=0.3
)

noisy = GaussianSpatialNoise(
    mean=0.0,
    sigma=0.3
)

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.grid(True)

spiral.plot(ax=ax, color="black", linewidth=1)
circles.plot(ax=ax, color="blue", noise=noisy)

plt.show()

noisy_spiral = NoisyPath(spiral, noisy)

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.grid(True)

spiral.plot(ax=ax, color="black", linewidth=1)
noisy_spiral.plot(ax=ax, color="red", alpha=0.7)

plt.show()