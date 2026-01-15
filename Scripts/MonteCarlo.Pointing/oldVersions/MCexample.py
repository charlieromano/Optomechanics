import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Model definition
# ============================================================

class ExampleDynamicModel:
    def simulate(self, params):
        t = np.linspace(0, 10, 400)
        A = params["A"]
        b = params["b"]
        y = A * np.exp(-b * t)
        return t, None, y


# ============================================================
# Monte Carlo infrastructure
# ============================================================

class JointParameterSampler:
    def __init__(self, names, mean, cov):
        self.names = names
        self.mean = np.array(mean)
        self.cov = np.array(cov)

    def sample(self, n):
        samples = np.random.multivariate_normal(self.mean, self.cov, n)
        return [dict(zip(self.names, s)) for s in samples]


class MonteCarloSimulation:
    def __init__(self, sampler, model, n_runs):
        self.sampler = sampler
        self.model = model
        self.n_runs = n_runs
        self.results = []

    def run(self):
        params_list = self.sampler.sample(self.n_runs)
        for params in params_list:
            t, x, y = self.model.simulate(params)
            self.results.append({
                "params": params,
                "time": t,
                "field": y
            })
        return self.results


# ============================================================
# Event & probability analysis
# ============================================================

class EventDetector:
    @staticmethod
    def time_to_threshold(time, signal, threshold):
        idx = np.where(signal >= threshold)[0]
        if len(idx) == 0:
            return None
        return time[idx[-1]]  # last time above threshold


class ReliabilityAnalyzer:
    def __init__(self, results):
        self.results = results

    def probability_exceedance(self, threshold):
        hits = sum(np.any(r["field"] >= threshold) for r in self.results)
        return hits / len(self.results)

    def time_to_hit_distribution(self, threshold):
        times = []
        for r in self.results:
            t_hit = EventDetector.time_to_threshold(
                r["time"], r["field"], threshold
            )
            if t_hit is not None:
                times.append(t_hit)
        return np.array(times)


# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":

    # ----------------------------
    # Monte Carlo configuration
    # ----------------------------
    N_MC = 500
    THRESHOLD = 0.5

    # Correlated parameters: A and b
    sampler = JointParameterSampler(
        names=["A", "b"],
        mean=[1.0, 0.4],
        cov=[[0.04, -0.01],
             [-0.01, 0.02]]
    )

    model = ExampleDynamicModel()
    mc = MonteCarloSimulation(sampler, model, N_MC)
    results = mc.run()

    analyzer = ReliabilityAnalyzer(results)

    # ----------------------------
    # Probability result
    # ----------------------------
    p_exc = analyzer.probability_exceedance(THRESHOLD)
    print(f"P(y ≥ {THRESHOLD}) = {p_exc:.3f}")

    # ----------------------------
    # Plot trajectories
    # ----------------------------
    plt.figure(figsize=(8, 5))

    for r in results[:50]:
        plt.plot(r["time"], r["field"], color="gray", alpha=0.3)

    plt.axhline(THRESHOLD, color="red", linestyle="--", label="Threshold")
    plt.xlabel("Time")
    plt.ylabel("y(t)")
    plt.title("Monte Carlo trajectories")
    plt.legend()
    plt.grid(True)

    plt.show()

    # ----------------------------
    # Envelope plot (percentiles)
    # ----------------------------
    Y = np.array([r["field"] for r in results])
    t = results[0]["time"]

    p5 = np.percentile(Y, 5, axis=0)
    p50 = np.percentile(Y, 50, axis=0)
    p95 = np.percentile(Y, 95, axis=0)

    plt.figure(figsize=(8, 5))
    plt.fill_between(t, p5, p95, alpha=0.3, label="5–95%")
    plt.plot(t, p50, label="Median")
    plt.axhline(THRESHOLD, color="red", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("y(t)")
    plt.title("Monte Carlo envelope")
    plt.legend()
    plt.grid(True)

    plt.show()

    # ----------------------------
    # Time-to-hit CDF
    # ----------------------------
    t_hits = analyzer.time_to_hit_distribution(THRESHOLD)

    if len(t_hits) > 0:
        t_sorted = np.sort(t_hits)
        cdf = np.arange(1, len(t_sorted) + 1) / len(t_sorted)

        plt.figure(figsize=(8, 5))
        plt.plot(t_sorted, cdf)
        plt.xlabel("Time to hit threshold")
        plt.ylabel("Cumulative probability")
        plt.title("CDF of time-to-hit")
        plt.grid(True)
        plt.show()
