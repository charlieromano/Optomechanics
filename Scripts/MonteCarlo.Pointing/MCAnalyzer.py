import numpy as np

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
    
