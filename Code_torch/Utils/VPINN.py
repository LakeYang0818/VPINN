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

    def __init__(self, *, architecture):
        super(VPINN, self).__init__()
        self.flatten = nn.Flatten()
        self.input_dim = architecture[0]
        self.output_dim = architecture[-1]
        self.hidden_dim = len(architecture)-2
        current_dim = architecture[0]
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(architecture[i], architecture[i+1]) for i in range(len(architecture)-1)])
        self.layers.append(nn.Linear(architecture[-2], 1))

    def forward(self, x):
        x = self.flatten(x)
        out = self.stack(x)
        return out
