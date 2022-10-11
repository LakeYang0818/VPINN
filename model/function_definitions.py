import sys
from os.path import dirname as up
from typing import Any

import numpy as np
import torch
from dantro._import_tools import import_module_from_path

sys.path.append(up(__file__))
sys.path.append(up(up(__file__)))

base = import_module_from_path(mod_path=up(up(__file__)), mod_str="include")


"""The exact solution 'u' and the external forcing 'f' used in the VPINN model.
Make sure the functions are defined for the dimension of the grid you are trying to evaluate.
Currently only 1D and 2D grids are supported.
"""

# A dictionary of some examples that we consider, in order to quickly access different equations.
EXAMPLES = {
    "Tanh": {
        "u": lambda x: 1 * (0.1 * np.sin(4 * np.pi * x) + np.tanh(5 * x)),
        "f": lambda x: -(
            0.1 * (4 * np.pi**2) * np.sin(4 * np.pi * x)
            + (2 * 5**2) * np.tanh(5 * x) / np.cosh(5 * x) ** 2
        ),
    },
    "SinSin2D": {
        "u": lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]),
        "f": lambda x: -2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]),
    },
    "DoubleGauss1D": {
        "u": lambda x: 3 * x**2 * np.exp(-(x**2)),
        "f": lambda x, k, f: 6 * np.exp(-(x**2)) * (2 * x**4 - 5 * x**2 + 1)
        + k * f(x),
    },
    "CubedRoot": {"u": lambda x: np.sign(x) * np.abs(x) ** (1.0 / 3), "f": lambda x: 1},
    "Burger1+1D": {"u": lambda x: 1.0 / (1 + x[0] ** 2), "f": lambda x: 0},
    "PorousMedium": {
        "u": lambda x, t: max(
            t ** (-1 / 3) * (1 - 1.0 / 12 * x**2 * t ** (-2 / 3)), 0
        ),
        "f": lambda x: 0,
    },
    "Tanh2D": {
        "u": lambda x: (0.1 * np.sin(2 * np.pi * x[0]) + np.tanh(10 * x[0]))
        * np.sin(2 * np.pi * x[1]),
        "f": lambda x: np.sin(2 * np.pi * x[1])
        * (np.tanh(10 * x[0]) * (-200 / (np.cosh(10 * x[0]) ** 2) - 4 * np.pi**2))
        - 4 * np.pi**2 / 5 * np.sin(2 * np.pi * x[0]),
    },
}

WEIGHT_FUNCTIONS = {
    "exponential": lambda x: 2 ** (-torch.sum(x)),
    "uniform": lambda x: torch.tensor(1.0),
}

# Exact solution
@base.adapt_input
def u(x: Any, *, func: str = None) -> float:

    return EXAMPLES[func]["u"](x)


# External forcing
@base.adapt_input
def f(x: Any, *, func: str = None) -> float:

    return EXAMPLES[func]["f"](x)
