import numpy as np

class JointParameterSampler:
    def __init__(self, names, mean, cov):
        self.names = names
        self.mean = np.array(mean)
        self.cov = np.array(cov)

    def sample(self, n=1):
        samples = np.random.multivariate_normal(self.mean, self.cov, n)
        return [
            dict(zip(self.names, s)) for s in samples
        ]


class Gaussian2DSampler:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample(self, rng):
        return rng.multivariate_normal(self.mean, self.cov)

