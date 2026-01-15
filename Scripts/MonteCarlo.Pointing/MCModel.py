import numpy as np

class Model:
    def simulate(self, params):
        """
        params: dict
        returns:
            time: 1D array
            space: 1D array or None
            field: ndarray [time, space] or [time]
        """
        raise NotImplementedError


class ExampleDynamicModel:
    def simulate(self, params):
        t = np.linspace(0, 10, 400)
        A = params["A"]
        b = params["b"]
        y = A * np.exp(-b * t)
        return t, None, y
