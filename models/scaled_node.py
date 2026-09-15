"""
Equation-Scaled Neural ODE.
Directly adopts the formulation from Kim et al. (2021) "Stiff Neural Ordinary Differential Equations"
to prevent scale-separation pathologies during stiff integration.

dy/dt = NN(y, t) * (y_scale / t_scale)
y_scale = y_max - y_min
t_scale = t_1 - t_0
"""

import torch
import torch.nn as nn
from models.neural_ode import NeuralODE


class ScaledVectorField(nn.Module):
    def __init__(self, base_func: nn.Module, y_scale: torch.Tensor, t_scale: float):
        super().__init__()
        self.base_func = base_func
        self.register_buffer("y_scale", y_scale)
        self.t_scale = float(t_scale)

    def forward(self, t, z):
        # Normalize time if needed, scale derivative output
        raw_deriv = self.base_func(t, z)
        scaled_deriv = raw_deriv * (self.y_scale / self.t_scale)
        return scaled_deriv


class ScaledNeuralODE(NeuralODE):
    def __init__(self, func: nn.Module, y_scale: torch.Tensor, t_scale: float, solver="dopri5", rtol=1e-5, atol=1e-7):
        scaled_func = ScaledVectorField(func, y_scale, t_scale)
        super().__init__(scaled_func, solver=solver, rtol=rtol, atol=atol)
        self.y_scale = y_scale
        self.t_scale = t_scale

    def compute_scaled_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Normalized MAE loss:
        L(theta) = MAE( y_pred / y_scale, y_target / y_scale )
        """
        scaled_pred = pred / self.y_scale
        scaled_target = target / self.y_scale
        return torch.mean(torch.abs(scaled_pred - scaled_target))
