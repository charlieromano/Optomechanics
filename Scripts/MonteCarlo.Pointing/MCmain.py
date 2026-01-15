import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# ============================================================
# 1. Path
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
# 2. Experiment Result
# ============================================================

class ExperimentResult:
    def __init__(self, target, trajectory, hit, time_to_hit, total_time):
        self.target = target
        self.trajectory = trajectory
        self.hit = hit
        self.time_to_hit = time_to_hit
        self.total_time = total_time

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
# 3. Model
# ============================================================

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
                return ExperimentResult(
                    target=target,
                    trajectory=traj,
                    hit=True,
                    time_to_hit=time,
                    total_time=time
                )
            time += dt

        total_time = len(traj) * dt
        return ExperimentResult(
            target=target,
            trajectory=traj,
            hit=False,
            time_to_hit=total_time,
            total_time=total_time
        )


# ============================================================
# 4. Sampler
# ============================================================

class Gaussian2DSampler:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample(self, rng):
        return rng.multivariate_normal(self.mean, self.cov)


# ============================================================
# 5. Monte Carlo Engine
# ============================================================

class MonteCarloEngine:
    def __init__(self, sampler, model, seed=None):
        self.sampler = sampler
        self.model = model
        self.rng = np.random.default_rng(seed)

    def run(self, n):
        return [
            self.model.run(self.sampler.sample(self.rng))
            for _ in range(n)
        ]


# ============================================================
# 6. Analyzer
# ============================================================

class AcquisitionAnalyzer:
    def __init__(self, results):
        self.results = results

    def probability_of_acquisition(self):
        return np.mean([r.hit for r in self.results])

    def acquisition_times(self):
        return np.array([r.time_to_hit for r in self.results])

    def plot_spatial_map(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        for r in self.results:
            color = 'green' if r.hit else 'red'
            ax.scatter(r.target[0], r.target[1], c=color, alpha=0.3)

        ax.set_aspect("equal")
        ax.set_title("Spatial acquisition map")
        ax.grid(True)
        return ax

    def plot_time_pdf_cdf(self, ax_pdf=None, ax_cdf=None, bins=30):
        times = self.acquisition_times()

        if ax_pdf is None or ax_cdf is None:
            fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(10, 4))

        # PDF
        ax_pdf.hist(times, bins=bins, density=True)
        ax_pdf.set_title("PDF of acquisition time")
        ax_pdf.set_xlabel("Time [s]")
        ax_pdf.set_ylabel("Density")

        # CDF
        t_sorted = np.sort(times)
        cdf = np.arange(1, len(times) + 1) / len(times)
        ax_cdf.plot(t_sorted, cdf)
        ax_cdf.set_title("CDF of acquisition time")
        ax_cdf.set_xlabel("Time [s]")
        ax_cdf.set_ylabel("Probability")

        return ax_pdf, ax_cdf


# ============================================================
# 7. Main
# ============================================================

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

results = engine.run(500)
analyzer = AcquisitionAnalyzer(results)

print("P(acquisition):", analyzer.probability_of_acquisition())

# ---- Composite plots
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2)

ax0 = fig.add_subplot(gs[0, 0])
results[0].plot_geometry(ax0)

ax1 = fig.add_subplot(gs[0, 1])
analyzer.plot_spatial_map(ax1)

ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
analyzer.plot_time_pdf_cdf(ax2, ax3)

plt.tight_layout()
plt.show()
