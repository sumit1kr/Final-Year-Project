"""
Vector field neural network architectures for Neural ODEs.
Supports variable depth, width, and activation functions (Tanh, GELU).
"""

import torch
import torch.nn as nn


class MLPVectorField(nn.Module):
    """
    Multilayer Perceptron defining the continuous dynamics dz/dt = f_theta(z, t).
    """
    def __init__(self, state_dim=2, hidden_dim=32, num_layers=2, activation="gelu", augment_dim=0):
        super().__init__()
        self.state_dim = state_dim
        self.augment_dim = augment_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        total_in_dim = state_dim + augment_dim

        if activation.lower() == "gelu":
            act_cls = nn.GELU
        elif activation.lower() == "softplus":
            act_cls = nn.Softplus
        else:
            act_cls = nn.Tanh

        layers = []
        # Input layer
        layers.append(nn.Linear(total_in_dim, hidden_dim))
        layers.append(act_cls())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls())

        # Output layer
        layers.append(nn.Linear(hidden_dim, total_in_dim))

        self.net = nn.Sequential(*layers)
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, t, z=None):
        if z is None:
            z = t
        return self.net(z)


def get_vector_field(preset="small", state_dim=2, activation="gelu"):
    """
    Preset configurations to test computational scaling:
    - small: 2 layers, 16 units (~300 params) - light evaluation overhead
    - medium: 3 layers, 32 units (~1,200 params)
    - large: 5 layers, 128 units (~35,000 params) - heavy evaluation overhead
    """
    presets = {
        "small": {"num_layers": 2, "hidden_dim": 16},
        "medium": {"num_layers": 3, "hidden_dim": 32},
        "large": {"num_layers": 5, "hidden_dim": 128}
    }
    cfg = presets.get(preset, presets["small"])
    return MLPVectorField(
        state_dim=state_dim,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        activation=activation
    )
