import numpy as np
from typing import Any

# Function decorator
# from Utils.utils import adapt_input

"""The exact solution 'u' and the external forcing 'f' used in the VPINN model.
Make sure the functions are defined for the dimension of the grid you are trying to evaluate. 
Currently only 1D and 2D grids are supported.
"""

# A dictionary of some examples that we consider, in order to quickly access different equations.
Examples = {
    'Tanh': {'u': lambda x: 1 * (0.1 * np.sin(4 * np.pi * x) + np.tanh(5 * x)),
             'f': lambda x: - (0.1 * (4 * np.pi ** 2) * np.sin(4 * np.pi * x)
                               + (2 * 5 ** 2) * np.tanh(5 * x) / np.cosh(5 * x) ** 2)},

    'SinSin2D': {'u': lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]),
                 'f': lambda x: -2 * np.pi ** 2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])},

    'DoubleGauss1D': {'u': lambda x: 3 * x ** 2 * np.exp(-x ** 2),
                      'f': lambda x, k, f: 6 * np.exp(-x ** 2) * (2 * x ** 4 - 5 * x ** 2 + 1) + k * f(x)},

    'CubedRoot': {'u': lambda x: np.sign(x) * np.abs(x) ** (1. / 3),
                  'f': lambda x: 1},

    'Burger': {'u': lambda x: 1.0 / (1 + x[0] ** 2),
                 'f': lambda x: np.sum(x)},

    'PorousMedium': {'u': lambda x, t: max(t**(-1/3)*(1-1.0/12*x**2*t**(-2/3)), 0),
                     'f': lambda x: 0},

    'Tanh2D' : {'u': lambda x: (0.1*np.sin(2*np.pi*x[0])+np.tanh(10*x[0]))*np.sin(2*np.pi*x[1]),
                'f': lambda x: np.sin(2*np.pi*x[1])*(
                        np.tanh(10*x[0])*(-200/(np.cosh(10*x[0])**2)-4*np.pi**2))
                               -4*np.pi**2/5*np.sin(2*np.pi*x[0])}
}

# # Exact solution
# @adapt_input
# def u(x: Any, *, example: str = None) -> float:
#
#     example = 'Tanh2D'
#     # Choose from the given examples
#     if example is not None:
#         return Examples[example]['u'](x)
#
#     # Define an explicit 1D case
#     if np.shape(x) <= (1,):
#         return x
#
#     # Define an explicit 2D case
#     elif len(x) == 2:
#         return x[0] + x[1]
#
#     else:
#         raise ValueError(f"You have not configured the function 'u' to handle {len(x)}-dimensional inputs!")
#
#
# # External forcing
# @adapt_input
# def f(x: Any, *, example: str = None) -> float:
#     example = 'Tanh2D'
#     # Choose from the given examples
#     if example is not None:
#         return Examples[example]['f'](x)
#
#     # Define an explicit 1D case
#     if np.shape(x) <= (1,):
#         return x
#
#     # Define an explicit 1D case
#     elif len(x) == 2:
#         return x[0] + x[1]
#
#     else:
#         raise ValueError(f"You have not configured the function 'f' to handle {len(x)}-dimensional inputs!")
