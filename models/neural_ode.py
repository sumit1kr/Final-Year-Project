"""
Neural Ordinary Differential Equation wrapper.
Interfaces with torchdiffeq.odeint and integrates call counters for NFE tracking.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint


class NeuralODE(nn.Module):
    """
    Standard Neural ODE model:
    dz/dt = f_theta(t, z)
    z(t_1) = z(t_0) + \int_{t_0}^{t_1} f_theta(t, z) dt
    """
    def __init__(self, func: nn.Module, solver="dopri5", rtol=1e-5, atol=1e-7, use_adjoint=False):
        super().__init__()
        self.func = func
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint
        self.nfe = 0

    def reset_nfe(self):
        self.nfe = 0

    def _counted_func(self, t, z):
        self.nfe += 1
        return self.func(t, z)

    def forward(self, z0: torch.Tensor, t: torch.Tensor, options=None):
        """
        Integrates z0 across time tensor t.
        """
        self.reset_nfe()
        integrate_fn = odeint_adjoint if self.use_adjoint else odeint

        # Use counted forward pass
        trajectory = integrate_fn(
            self._counted_func,
            z0,
            t,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver,
            options=options
        )
        return trajectory
