# MonteCarloEngine.py
import numpy as np

class MonteCarloEngine:
    def __init__(self, model, method, *, adapter=None, recorder=None, seed=None, progress=True):
        """
        model   : AcquisitionModel instance
        method  : DirectMonteCarlo or InverseMonteCarlo instance
        adapter : optional ResultAdapter
        recorder: optional storage/writer object
        seed    : RNG seed
        progress: bool, print progress
        """
        self.model = model
        self.method = method
        self.adapter = adapter
        self.recorder = recorder
        self.progress = progress
        self.rng = np.random.default_rng(seed)

    def run(self, n):
        results = []

        try:
            for i in range(n):
                # ------------------------
                # Generate parameters & run
                # ------------------------
                params = self.method.propose(self.rng)
                result = self.model.run(params)
                self.method.observe(result)
                results.append(result)

                # ------------------------
                # Optional recorder
                # ------------------------
                if self.adapter and self.recorder:
                    record = self.adapter.extract(result)
                    self.recorder.write(i, record)

                # ------------------------
                # Progress printing
                # ------------------------
                if self.progress and (i % max(1, n // 100) == 0 or i == n-1):
                    print(f"Monte Carlo simulation {i+1}/{n} ({(i+1)/n*100:.1f}%)")

                # ------------------------
                # Optional early stop
                # ------------------------
                if hasattr(self.method, "should_stop") and self.method.should_stop(i, results):
                    break

        except KeyboardInterrupt:
            print("\nMonte Carlo run interrupted by user.")
            print(f"Returning {len(results)} results collected so far.")

        finally:
            self.model.close()  # ensure HDF5 flush or cleanup

        return results


# ============================================================
# Direct Monte Carlo
# ============================================================
class DirectMonteCarlo:
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set

    def propose(self, rng):
        return self.parameter_set.resolve(rng)

    def observe(self, result):
        # For direct MC, no learning or scoring
        pass


# ============================================================
# Inverse Monte Carlo (scoring-based)
# ============================================================
class InverseMonteCarlo:
    def __init__(self, parameter_set, objective):
        self.parameter_set = parameter_set
        self.objective = objective
        self.scores = []

    def propose(self, rng):
        return self.parameter_set.resolve(rng)

    def observe(self, result):
        score = self.objective.evaluate(result)
        self.scores.append(score)
