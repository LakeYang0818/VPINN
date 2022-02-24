import numpy as np
import os
import torch
from torch import nn
from typing import Any, Sequence
import time

# Local imports
from .data_types import DataGrid, DataSet


class VPINN(nn.Module):
    """ A variational physics-informed neural net. Inherits from the keras.Model parent."""

    VAR_FORMS = {1, 2, 3}

    EQUATION_TYPES = {
        "Poisson",
        "Helmholtz",
        "Burger"
    }

    def __init__(self, architecture, activation_func):
        """Initialises the neural net.

        Args:
            architecture: the number of layers and nodes per layer
            activation_func: the activation function to use
        """
        super(VPINN, self).__init__()
        self.flatten = nn.Flatten()
        self.input_dim = architecture[0]
        self.output_dim = architecture[-1]
        self.hidden_dim = len(architecture)-2
        self.activation_func = activation_func

        # Add layers
        self.layers = nn.ModuleList()
        for i in range(len(architecture)-1):
            self.layers.append(nn.Linear(architecture[i], architecture[i+1]))

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.activation_func(self.layers[i](x))
        x = self.layers[-1](x)
        return x

    def grad(self, x):
        x = torch.autograd.grad(self.forward(x), x)
        return x

    def gradgrad(self, x):
        first_derivative = torch.autograd.grad(self.forward(x), x)
        return torch.autograd.grad(first_derivative, x)

