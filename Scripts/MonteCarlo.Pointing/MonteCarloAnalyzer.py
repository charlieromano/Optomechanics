# MonteCarloAnalyzer.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from MonteCarloModel import ExperimentResult
import h5py

class AcquisitionAnalyzer:
    @classmethod
    def from_hdf5(cls, filename):
        with h5py.File(filename, "r") as f:
            g = f["runs"]
            results = []
            for i in range(len(g["hit"])):
                results.append(
                    ExperimentResult(
                        target=g["target"][i],
                        trajectory=None,
                        hit=bool(g["hit"][i]),
                        time_to_hit=g["time_to_hit"][i],
                        total_time=g["total_time"][i],
                        dwell_time=g["dwell_time"][i],
                        energy=g["energy"][i],
                    )
                )
        return cls(results)

    def __init__(self, results):
        self.results = results

    # -------------------
    # Statistics
    # -------------------
    def probability_of_acquisition(self):
        hits = [r.hit for r in self.results]
        return np.mean(hits) if hits else np.nan

    def acquisition_times(self):
        times = np.array([r.time_to_hit for r in self.results])
        return times

    def mean_hit_position(self):
        hits = np.array([r.target for r in self.results if r.hit])
        if len(hits) == 0:
            return None
        return np.mean(hits, axis=0)

    def median_hit_position(self):
        hits = np.array([r.target for r in self.results if r.hit])
        if len(hits) == 0:
            return None
        return np.median(hits, axis=0)

    # -------------------
    # Spatial map
    # -------------------
    def plot_spatial_map(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        hit_x, hit_y = [], []
        miss_x, miss_y = [], []

        for r in self.results:
            if r.hit:
                hit_x.append(r.target[0])
                hit_y.append(r.target[1])
            else:
                miss_x.append(r.target[0])
                miss_y.append(r.target[1])

        if hit_x:
            ax.scatter(hit_x, hit_y, c='green', alpha=0.5, s=20, label="Hit")
        if miss_x:
            ax.scatter(miss_x, miss_y, c='red', alpha=0.5, s=20, label="Miss")

        # Scan radius from first trajectory
        r0 = self.results[0]
        if r0.trajectory is not None:
            scan_radius = np.max(np.linalg.norm(r0.trajectory, axis=1))
            scan_circle = Circle((0,0), scan_radius, fill=False,
                                 linestyle='--', linewidth=0.4, color='black', label='Scan area')
            ax.add_patch(scan_circle)

        ax.set_aspect("equal")
        ax.set_title("Spatial acquisition map")
        ax.grid(True)
        ax.legend()
        return ax

    # -------------------
    # PDF / CDF plot
    # -------------------
    def plot_time_pdf_cdf(
        self,
        ax_pdf=None,
        ax_cdf=None,
        bins=30,
        probability_threshold=0.95,
        show_percentiles=None,  # e.g., [0.5, 0.95, 0.99]
        percentile_line_styles=None  # e.g., {0.5: ('green','dashdot')}
    ):
        times = self.acquisition_times()
        times = times[~np.isnan(times)]  # remove NaNs

        if ax_pdf is None or ax_cdf is None:
            fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(10, 4))

        # -------------------
        # PDF
        # -------------------
        ax_pdf.hist(times, bins=bins, density=True)
        mean_time = np.nanmean(times)
        ax_pdf.axvline(mean_time, color='red', linestyle='--', linewidth=1,
                    label=f"Mean = {mean_time:.2f}s")
        ax_pdf.set_title("PDF of acquisition time")
        ax_pdf.set_xlabel("Time [s]")
        ax_pdf.set_ylabel("Density")
        ax_pdf.grid(True)
        ax_pdf.legend()

        # -------------------
        # CDF
        # -------------------
        t_sorted = np.sort(times)
        cdf = np.arange(1, len(t_sorted)+1) / len(t_sorted)
        ax_cdf.plot(t_sorted, cdf, color='blue', label='CDF')

        # Default line styles
        default_styles = {0.5: ('green','dashdot'), 0.95: ('red','dashdot'), 0.99: ('purple','dotted')}
        if percentile_line_styles is None:
            percentile_line_styles = default_styles

        if show_percentiles is None:
            show_percentiles = []

        for p in show_percentiles:
            t_p = np.quantile(t_sorted, p)
            color, linestyle = percentile_line_styles.get(p, ('black','dashed'))
            # Vertical line
            ax_cdf.axvline(t_p, color=color, linestyle=linestyle, linewidth=1.5)
            # Horizontal line at crossing point
            ax_cdf.axhline(p, color=color, linestyle=linestyle, linewidth=1.5)
            # Dot marker at intersection
            ax_cdf.plot(t_p, p, 'o', color=color, markersize=6)
            # Legend label
            ax_cdf.plot([], [], color=color, linestyle=linestyle,
                        label=f"P{int(p*100)}% = {t_p:.4f}s")

        ax_cdf.set_title("CDF of acquisition time")
        ax_cdf.set_xlabel("Time [s]")
        ax_cdf.set_ylabel("Probability")
        ax_cdf.grid(True)
        ax_cdf.legend()

        return ax_pdf, ax_cdf


    # -------------------
    # Parameter text overlay
    # -------------------
    @staticmethod
    def add_parameter_text(fig, param_set, n_simulations=None, x=0.02, y=0.98, fontsize=12):
        lines = ["Monte Carlo parameters:"]
        params = getattr(param_set, "parameters", param_set)
        if isinstance(params, dict):
            iterable = params.items()
        else:
            iterable = [(p.name, p) for p in params]

        for name, p in iterable:
            if hasattr(p, "kind") and p.kind == "fixed":
                val = p.value
                if isinstance(val, tuple):
                    value, unit = val
                    lines.append(f"{name}: {value:g} {unit}")
                else:
                    lines.append(f"{name}: {val}")
            elif hasattr(p, "kind") and p.kind != "fixed":
                lines.append(f"{name}: {p.dist}")
            else:
                lines.append(str(p))

        if n_simulations is not None:
            lines.append(f"N_simulations: {n_simulations}")

        textstr = "\n".join(lines)

        fig.text(x, y, textstr, fontsize=fontsize, va='top', ha='left',
                 fontweight='normal', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
