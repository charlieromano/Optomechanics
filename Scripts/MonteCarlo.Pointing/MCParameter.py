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
        """
        kind:
            - "fixed"
            - "range"
            - "distribution"
            - "derived"
        """
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
                raise ValueError(f"{self.name}: range needs bounds (min, max)")
        elif self.kind == "distribution":
            if self.dist is None:
                raise ValueError(f"{self.name}: distribution needs dist")
            if self.dist == "uniform" and self.bounds is None:
                raise ValueError(f"{self.name}: uniform needs bounds")
        elif self.kind == "derived":
            if self.formula is None or not callable(self.formula):
                raise ValueError(f"{self.name}: derived needs callable formula")
            if not self.depends_on:
                raise ValueError(f"{self.name}: derived needs dependencies")
        else:
            raise ValueError(f"{self.name}: unknown kind '{self.kind}'")
    def sample(self, size=1):
        if self.kind != "distribution":
            raise ValueError("Sampling only valid for kind='distribution'")
        if self.dist == "uniform":
            low, high = self.bounds
            return np.random.uniform(low, high, size)
        elif self.dist == "normal":
            return np.random.normal(
                self.kwargs["mean"],
                self.kwargs["std"],
                size
            )
        elif self.dist == "lognormal":
            return np.random.lognormal(
                self.kwargs["mean"],
                self.kwargs["sigma"],
                size
            )
        elif self.dist == "rayleigh":
            return np.random.rayleigh(
                self.kwargs["scale"],
                size
            )
        elif self.dist == "exponential":
            return np.random.exponential(
                self.kwargs["scale"],
                size
            )
        elif self.dist == "Gaussian2D":
            mean = self.kwargs.get("mean", [0.0, 0.0])
            cov = self.kwargs.get(
                "cov",
                [[1.0, 0.0], [0.0, 1.0]]
            )
            return np.random.multivariate_normal(mean, cov, size)
        elif self.dist == "Rayleigh2D":
            scale = self.kwargs.get("scale", 1.0)
            r = np.random.rayleigh(scale, size)
            theta = np.random.uniform(0, 2 * np.pi, size)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return np.column_stack((x, y))
        else:
            raise ValueError(f"Unsupported distribution: {self.dist}")

