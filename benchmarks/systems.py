"""
Benchmark Dynamical Systems for Neural ODE Research.
Includes non-stiff, moderately stiff, and highly stiff differential equations.
"""

import numpy as np

class BenchmarkODE:
    """Base class for benchmark ODE systems."""
    name = "BaseODE"
    stiffness_regime = "non-stiff"
    dim = 2

    def rhs(self, t, y):
        raise NotImplementedError

    def jacobian(self, t, y):
        return None


class LotkaVolterra(BenchmarkODE):
    """
    Classic Predator-Prey dynamics (Non-stiff, 2D).
    dy1/dt = alpha * y1 - beta * y1 * y2
    dy2/dt = delta * y1 * y2 - gamma * y2
    """
    name = "Lotka-Volterra"
    stiffness_regime = "non-stiff"
    dim = 2

    def __init__(self, alpha=1.5, beta=1.0, gamma=3.0, delta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.default_y0 = np.array([1.0, 0.5], dtype=np.float64)
        self.default_t_span = (0.0, 15.0)
        self.default_num_points = 150

    def rhs(self, t, y):
        y1, y2 = y[0], y[1]
        dy1 = self.alpha * y1 - self.beta * y1 * y2
        dy2 = self.delta * y1 * y2 - self.gamma * y2
        return np.array([dy1, dy2], dtype=np.float64)

    def jacobian(self, t, y):
        y1, y2 = y[0], y[1]
        return np.array([
            [self.alpha - self.beta * y2, -self.beta * y1],
            [self.delta * y2, self.delta * y1 - self.gamma]
        ], dtype=np.float64)


class FitzHughNagumo(BenchmarkODE):
    """
    Neuronal excitation model (Non-stiff / Mild, 2D).
    dv/dt = v - (v^3)/3 - w + I
    dw/dt = (1 / tau) * (v + a - b * w)
    """
    name = "FitzHugh-Nagumo"
    stiffness_regime = "non-stiff"
    dim = 2

    def __init__(self, a=0.7, b=0.8, tau=12.5, I=0.5):
        self.a = a
        self.b = b
        self.tau = tau
        self.I = I
        self.default_y0 = np.array([-1.0, 1.0], dtype=np.float64)
        self.default_t_span = (0.0, 50.0)
        self.default_num_points = 200

    def rhs(self, t, y):
        v, w = y[0], y[1]
        dv = v - (v**3) / 3.0 - w + self.I
        dw = (v + self.a - self.b * w) / self.tau
        return np.array([dv, dw], dtype=np.float64)

    def jacobian(self, t, y):
        v, w = y[0], y[1]
        return np.array([
            [1.0 - v**2, -1.0],
            [1.0 / self.tau, -self.b / self.tau]
        ], dtype=np.float64)


class VanDerPol(BenchmarkODE):
    """
    Van der Pol Oscillator with tunable stiffness parameter mu (2D).
    dy1/dt = y2
    dy2/dt = mu * (1 - y1^2) * y2 - y1
    Stiffness increases dramatically as mu grows:
    - mu = 1: non-stiff
    - mu = 10: moderate stiffness
    - mu = 100: highly stiff relaxation oscillator
    """
    name = "VanDerPol"
    dim = 2

    def __init__(self, mu=10.0):
        self.mu = mu
        self.stiffness_regime = "moderate-stiff" if mu <= 20 else "stiff"
        self.default_y0 = np.array([2.0, 0.0], dtype=np.float64)
        self.default_t_span = (0.0, 30.0)
        self.default_num_points = 300

    def rhs(self, t, y):
        y1, y2 = y[0], y[1]
        dy1 = y2
        dy2 = self.mu * (1.0 - y1**2) * y2 - y1
        return np.array([dy1, dy2], dtype=np.float64)

    def jacobian(self, t, y):
        y1, y2 = y[0], y[1]
        return np.array([
            [0.0, 1.0],
            [-2.0 * self.mu * y1 * y2 - 1.0, self.mu * (1.0 - y1**2)]
        ], dtype=np.float64)


class Robertson(BenchmarkODE):
    """
    Classic Robertson Chemical Reaction (ROBER) (Highly stiff, 3D).
    From Kim et al. (2021) base paper:
    dy1/dt = -k1 * y1 + k3 * y2 * y3
    dy2/dt = k1 * y1 - k2 * y2^2 - k3 * y2 * y3
    dy3/dt = k2 * y2^2
    Rate constants: k1 = 0.04, k2 = 3e7, k3 = 1e4.
    """
    name = "Robertson"
    stiffness_regime = "highly-stiff"
    dim = 3

    def __init__(self, k1=0.04, k2=3e7, k3=1e4):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.default_y0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        # Log-spaced time span from 1e-5 to 1e5
        self.default_t_span = (1e-5, 1e5)
        self.default_num_points = 50

    def rhs(self, t, y):
        y1, y2, y3 = y[0], y[1], y[2]
        dy1 = -self.k1 * y1 + self.k3 * y2 * y3
        dy2 = self.k1 * y1 - self.k2 * (y2**2) - self.k3 * y2 * y3
        dy3 = self.k2 * (y2**2)
        return np.array([dy1, dy2, dy3], dtype=np.float64)

    def jacobian(self, t, y):
        y1, y2, y3 = y[0], y[1], y[2]
        return np.array([
            [-self.k1, self.k3 * y3, self.k3 * y2],
            [self.k1, -2.0 * self.k2 * y2 - self.k3 * y3, -self.k3 * y2],
            [0.0, 2.0 * self.k2 * y2, 0.0]
        ], dtype=np.float64)
