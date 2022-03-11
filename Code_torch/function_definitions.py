import numpy as np
from typing import Any

# Function decorator
from Utils.utils import adapt_input

"""The exact solution 'u' and the external forcing 'f' used in the VPINN model.
Make sure the functions are defined for the dimension of the grid you are trying to evaluate. 
Currently only 1D and 2D grids are supported.
"""


# Exact solution
@adapt_input
def u(x: Any) -> float:
    # Define the 1D case
    if np.shape(x) <= (1,):
        # r1, omega, amp = 5, 4 * np.pi, 1
        # return amp * (0.1 * np.sin(omega * x) + np.tanh(r1 * x))
        return np.sign(x)*np.abs(x)**(1./3)

    # Define the 2D case
    elif len(x) == 2:
        #return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        return 1.0/(1+x[0]**2)

    else:
        raise ValueError(f"You have not configured the function 'u' to handle {len(x)}-dimensional inputs!")


# External forcing
@adapt_input
def f(x: Any) -> float:
    # Define the 1D case
    if len(x) == 1:
        # r1, omega, amp = 5, 4 * np.pi, 1
        # return -amp * (0.1 * (omega ** 2) * np.sin(omega * x)
        #                + (2 * r1 ** 2) * np.tanh(r1 * x) / np.cosh(r1 * x) ** 2)
        #return 2.0*np.exp(-1.0*(x**2))*(2.0*(x**4)-5.0*(x**2)+1) + 5.0*u(x)
        return 1.0
    # Define the 2D case
    elif len(x) == 2:
        return 0


    else:
        raise ValueError(f"You have not configured the function 'f' to handle {len(x)}-dimensional inputs!")
