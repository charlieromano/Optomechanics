# MonteCarloParameters.py
import numpy as np


class Parameter:
    def __init__(
        self,
        name,
        kind="fixed",
        value=None,
        bounds=None,
        dist=None,
        formula=None,
        depends_on=None,
        units=None,
        description=None,
        **kwargs
    ):
        self.name = name
        self.kind = kind
        self.value = value
        self.bounds = bounds
        self.dist = dist
        self.formula = formula
        self.depends_on = depends_on or []
        self.units = units
        self.description = description
        self.kwargs = kwargs
        self._validate()

    def _validate(self):
        if self.kind == "fixed":
            if self.value is None:
                raise ValueError(f"{self.name}: fixed needs value")

        elif self.kind == "range":
            if self.bounds is None or len(self.bounds) != 2:
                raise ValueError(f"{self.name}: range needs bounds")

        elif self.kind == "distribution":
            if self.dist is None:
                raise ValueError(f"{self.name}: distribution needs dist")

        elif self.kind == "derived":
            if self.formula is None or not callable(self.formula):
                raise ValueError(f"{self.name}: derived needs callable formula")
            if not self.depends_on:
                raise ValueError(f"{self.name}: derived needs dependencies")

        else:
            raise ValueError(f"{self.name}: unknown kind '{self.kind}'")

    def sample(self, rng):
        if self.kind == "fixed":
            return self.value

        if self.kind == "range":
            low, high = self.bounds
            return rng.uniform(low, high)

        if self.kind == "distribution":
            if self.dist == "uniform":
                low, high = self.bounds
                return rng.uniform(low, high)

            elif self.dist == "normal":
                return rng.normal(
                    self.kwargs["mean"],
                    self.kwargs["std"]
                )

            elif self.dist == "Gaussian2D":
                mean = self.kwargs.get("mean", [0.0, 0.0])
                cov = self.kwargs.get("cov", [[1, 0], [0, 1]])
                return rng.multivariate_normal(mean, cov)

            else:
                raise ValueError(f"Unsupported distribution: {self.dist}")

        raise RuntimeError("sample() called on derived parameter")


class ParameterSet:
    """
    Resolves all parameters ONCE per Monte Carlo experiment.
    """
    def __init__(self, parameters):
        self.parameters = {p.name: p for p in parameters}

    def resolve(self, rng):
        values = {}

        # First pass: fixed / range / distribution
        for p in self.parameters.values():
            if p.kind != "derived":
                values[p.name] = p.sample(rng)

        # Second pass: derived parameters
        for p in self.parameters.values():
            if p.kind == "derived":
                deps = {k: values[k] for k in p.depends_on}
                values[p.name] = p.formula(**deps)

        return values
