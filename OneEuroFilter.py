"""
One Euro Filter - Python implementation for landmark / signal jitter filtering
Based on: Casiez, G., Roussel, N. and Vogel, D. (2012)
"""

import math
import numpy as np


class OneEuroFilter:
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.001, dcutoff=1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = None
        self.last_time = None

    def _alpha(self, rate, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        x = np.asarray(x, dtype=np.float32)

        if self.last_time is not None and timestamp is not None and timestamp > self.last_time:
            rate = 1.0 / (timestamp - self.last_time)
        else:
            rate = self.freq

        self.last_time = timestamp

        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x

        dx = (x - self.x_prev) * rate
        a_d = self._alpha(rate, self.dcutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(rate, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat
