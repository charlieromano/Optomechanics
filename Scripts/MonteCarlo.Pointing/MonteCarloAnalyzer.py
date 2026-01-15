import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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


class ResultAnalyzer:
    def __init__(self, results):
        self.results = results
    def get_array(self, output_name):
        return np.array([r[output_name] for r in self.results])
    def probability_less_than(self, output_name, threshold):
        values = self.get_array(output_name)
        return np.mean(values < threshold)
    def summary(self, output_name):
        values = self.get_array(output_name)
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "p5": np.percentile(values, 5),
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
        }
    def probability_exceedance(self, threshold):
        hits = 0
        for r in self.results:
            if np.any(r["field"] >= threshold):
                hits += 1
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

        # Plot target positions
        for r in self.results:
            color = 'green' if r.hit else 'red'
            ax.scatter(
                r.target[0], r.target[1],
                c=color, alpha=0.3, s=15
            )

        # Determine scan radius from trajectory
        r0 = self.results[0]
        scan_radius = np.max(np.linalg.norm(r0.trajectory, axis=1))

        scan_circle = Circle(
            (0, 0),
            scan_radius,
            fill=False,
            linestyle='--',
            linewidth=0.5,
            color='black',
            #label='Scan area'
        )
        ax.add_patch(scan_circle)

        ax.set_aspect("equal")
        ax.set_title("Spatial acquisition map")
        ax.grid(True)
        ax.legend()

        return ax

    def plot_time_pdf_cdf(
        self,
        ax_pdf=None,
        ax_cdf=None,
        bins=30,
        probability_threshold=0.95
    ):
        times = self.acquisition_times()

        if ax_pdf is None or ax_cdf is None:
            fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(10, 4))

        # -------------------
        # PDF
        # -------------------
        ax_pdf.hist(times, bins=bins, density=True)
        mean_time = np.mean(times)

        ax_pdf.axvline(
            mean_time,
            color='blue',
            linestyle='--',
            linewidth=1,
            label=f"Mean = {mean_time:.2f}s"
        )

        ax_pdf.set_title("PDF of acquisition time")
        ax_pdf.set_xlabel("Time [s]")
        ax_pdf.set_ylabel("Density")
        ax_pdf.legend()
        ax_pdf.grid(True)

        # -------------------
        # CDF
        # -------------------
        t_sorted = np.sort(times)
        cdf = np.arange(1, len(times) + 1) / len(times)

        ax_cdf.plot(t_sorted, cdf, label="CDF")

        # ---- Probability-based threshold (NEW)
        if probability_threshold is not None:
            # Quantile time
            t_quantile = np.quantile(times, probability_threshold)

            ax_cdf.axhline(
                probability_threshold,
                color='red',
                linestyle='--',
                linewidth=1,
                label=f"P = {probability_threshold:.2f}"
            )

            ax_cdf.axvline(
                t_quantile,
                color='red',
                linestyle='--',
                linewidth=1,
                label=f"T = {t_quantile:.2f}s"
            )

            ax_cdf.scatter(
                [t_quantile],
                [probability_threshold],
                color='red',
                zorder=5
            )

        ax_cdf.set_title("CDF of acquisition time")
        ax_cdf.set_xlabel("Time [s]")
        ax_cdf.set_ylabel("Probability")
        ax_cdf.legend()
        ax_cdf.grid(True)

        return ax_pdf, ax_cdf

