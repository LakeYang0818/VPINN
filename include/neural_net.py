from typing import Any, List, Union

from torch import nn

# Local imports
from .Variational_forms import *

ACTIVATION_FUNCS = {
    "abs": [torch.abs, False],
    "celu": [torch.nn.CELU, True],
    "cos": [torch.cos, False],
    "cosine": [torch.cos, False],
    "elu": [torch.nn.ELU, True],
    "gelu": [torch.nn.GELU, True],
    "hardshrink": [torch.nn.Hardshrink, True],
    "hardsigmoid": [torch.nn.Hardsigmoid, True],
    "hardswish": [torch.nn.Hardswish, True],
    "hardtanh": [torch.nn.Hardtanh, True],
    "leakyrelu": [torch.nn.LeakyReLU, True],
    "linear": [None, False],
    "logsigmoid": [torch.nn.LogSigmoid, True],
    "mish": [torch.nn.Mish, True],
    "none": [None, False],
    "prelu": [torch.nn.PReLU, True],
    "relu": [torch.nn.ReLU, True],
    "rrelu": [torch.nn.RReLU, True],
    "selu": [torch.nn.SELU, True],
    "sigmoid": [torch.nn.Sigmoid, True],
    "silu": [torch.nn.SiLU, True],
    "sin": [torch.sin, False],
    "sine": [torch.sin, False],
    "softplus": [torch.nn.Softplus, True],
    "softshrink": [torch.nn.Softshrink, True],
    "swish": [torch.nn.SiLU, True],
    "tanh": [torch.nn.Tanh, True],
    "tanhshrink": [torch.nn.Tanhshrink, True],
    "threshold": [torch.nn.Threshold, True],
}

# ----------------------------------------------------------------------------------------------------------------------
# -- NN utility function -----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def get_activation_funcs(n_layers: int, cfg: Union[str, dict] = None) -> List[Any]:
    """Extracts the activation functions from the config. The config is a dictionary, with the keys representing
    the layer number, and the entry the activation function to use. Alternatively, the config can also be a single
    string, which is then applied to the entire neural net.

    Example:
        activation_funcs: abs    # applies the absolute value to the entire neural net
    Example:
        activation_funcs:        # applies the nn.Hardtanh activation function to the entire neural net
          name: HardTanh
          args:
            - -2
            - 2
    Example:
        activation_funcs:
          0: abs
          1: relu
          2: tanh
    """

    funcs = [None] * (n_layers + 1)

    if cfg is None:
        return funcs

    elif isinstance(cfg, str):
        _f = ACTIVATION_FUNCS[cfg.lower()]
        if _f[1]:
            return [_f[0]()] * (n_layers + 1)
        else:
            return [_f[0]] * (n_layers + 1)

    elif isinstance(cfg, dict):
        if "name" in cfg.keys():
            _f = ACTIVATION_FUNCS[cfg.get("name").lower()]
            if _f[1]:
                return [_f[0](*cfg.get("args", ()), **cfg.get("kwargs", {}))] * (
                    n_layers + 1
                )
            else:
                return [_f[0]] * (n_layers + 1)
        else:
            for idx, entry in cfg.items():

                if isinstance(entry, str):
                    _f = ACTIVATION_FUNCS[entry.lower()]
                    if _f[1]:
                        funcs[idx] = _f[0]()
                    else:
                        funcs[idx] = _f[0]
                elif isinstance(entry, dict):
                    funcs[idx] = ACTIVATION_FUNCS[entry.get("name").lower()][0](
                        *entry.get("args", ()), **entry.get("kwargs", {})
                    )

                else:
                    raise ValueError(
                        f"Unrecognised argument {entry} in 'activation_funcs' dictionary!"
                    )
            return funcs
    else:
        raise ValueError(f"Unrecognised argument {cfg} for 'activation_funcs'!")


# ----------------------------------------------------------------------------------------------------------------------
# -- Neural net class --------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class NeuralNet(nn.Module):

    """A variational physics-informed neural net. Inherits from the nn.Module parent."""

    VAR_FORMS = {0, 1, 2}

    EQUATION_TYPES = {
        "Burger",
        # "Helmholtz",
        "Poisson",
        # "PorousMedium",
        # "Burger",
        # "Weak1D",
    }

    # Torch optimizers
    OPTIMIZERS = {
        "Adagrad": torch.optim.Adagrad,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SparseAdam": torch.optim.SparseAdam,
        "Adamax": torch.optim.Adamax,
        "ASGD": torch.optim.ASGD,
        "LBFGS": torch.optim.LBFGS,
        "NAdam": torch.optim.NAdam,
        "RAdam": torch.optim.RAdam,
        "RMSprop": torch.optim.RMSprop,
        "Rprop": torch.optim.Rprop,
        "SGD": torch.optim.SGD,
    }

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        num_layers: int,
        nodes_per_layer: int,
        activation_funcs: dict = None,
        optimizer: str = "Adam",
        learning_rate: float = 0.001,
        bias: bool = False,
        init_bias: tuple = None,
        eq_type: str = "Poisson",
        var_form: int = 1,
        pde_constants: dict = None,
    ):

        """Initialises the neural net.
        :param input_size: the number of input values
        :param output_size: the number of output values
        :param num_layers: the number of hidden layers
        :param nodes_per_layer: the number of neurons in the hidden layers
        :param activation_funcs: a dictionary specifying the activation functions to use
        :param learning_rate: the learning rate of the optimizer
        :param bias: whether to initialise the layers with a bias
        :param init_bias: the interval from which to uniformly initialise the bias
        :param eq_type: the equation type of the PDE in question
        :param var_form: the variational form to use for the loss function
        :param pde_constants: the constants for the pde in use
        Raises:
            ValueError: if the equation type is unrecognized
            ValueError: if the variational form is unrecognized
        """

        if var_form not in self.VAR_FORMS:
            raise ValueError(
                f"Unrecognized variational_form  "
                f"'{var_form}'! "
                f"Choose from: [1, 2, 3]"
            )

        super().__init__()
        self.flatten = nn.Flatten()

        self.input_dim = input_size
        self.output_dim = output_size
        self.hidden_dim = num_layers
        self.activation_funcs = get_activation_funcs(num_layers, activation_funcs)
        architecture = [input_size] + [nodes_per_layer] * num_layers + [output_size]

        # Add the neural net layers
        self.layers = nn.ModuleList()
        for i in range(len(architecture) - 1):
            layer = nn.Linear(architecture[i], architecture[i + 1])

            # # Initialise the biases of the layers with a uniform distribution on init_bias
            # if bias and init_bias is not None:
            #     torch.nn.init.uniform_(layer.bias, init_bias[0], init_bias[1])
            self.layers.append(layer)

        # Get the optimizer
        self.optimizer = self.OPTIMIZERS[optimizer](self.parameters(), lr=learning_rate)

        # Get equation type and variational form
        self._eq_type = eq_type
        self._var_form = var_form

        # Get equation parameters
        self._pde_constants = pde_constants

    # ... Evaluation functions .........................................................................................

    # The model forward pass
    def forward(self, x):
        for i in range(len(self.layers)):
            if self.activation_funcs[i] is None:
                x = self.layers[i](x)
            else:
                x = self.activation_funcs[i](self.layers[i](x))
        return x

    # Computes the first derivative of the model output. x can be a single tensor or a stack of tensors
    def grad(self, x, *, requires_grad: bool = True):
        y = self.forward(x)
        x = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y), create_graph=requires_grad
        )[0]
        return x

    # Computes the second derivative of the model output. x can be a single tensor or a stack of tensors
    def gradgrad(self, x, *, requires_grad: bool = True):

        first_derivative = self.grad(x, requires_grad=True)
        second_derivative = torch.autograd.grad(
            first_derivative,
            x,
            grad_outputs=torch.ones_like(x),
            create_graph=requires_grad,
        )[0]

        return second_derivative

    # ... Loss functions ...............................................................................................

    def variational_loss(
        self,
        device,
        grid,
        grid_boundary,
        normals,
        f_integrated,
        test_func_vals,
        weights,
        domain_density,
        d1test_func_vals=None,
        d2test_func_vals=None,
        d1test_func_vals_bd=None,
    ) -> torch.Tensor:

        """Calculates the variational loss on the grid interior.

        :param device: the training device
        :param grid: the grid
        :param grid_boundary: the grid boundary
        :param normals: the grid boundary normals
        :param f_integrated: the external function integrated against all test functions
        :param test_func_vals: the test function values on the grid interior
        :param d1test_func_vals: the values of the test function derivatives on the grid interior
        :param d2test_func_vals: the values of the second derivatives of the test functions on the grid interior
        :param d1test_func_vals_bd: the values of the first derivatives of the test functions on the boundary
        :param weight_function: the function used to weight the test function contributions in the residual loss
        :return: the variational loss
        """

        if self._eq_type == "Burger":

            return Burger(
                device,
                self.forward,
                self.grad,
                grid,
                domain_density,
                f_integrated,
                d1test_func_vals,
                weights,
                self._pde_constants.get("Burger", 0),
            )

        # elif self._eq_type == "Helmholtz":
        #
        #     return Helmholtz(
        #         self.forward,
        #         self.grad,
        #         self.gradgrad,
        #         grid,
        #         f_integrated,
        #         test_func_vals,
        #         d1test_func_vals,
        #         d2test_func_vals,
        #         d1test_func_vals_bd,
        #         self._var_form,
        #         self._pde_constants,
        #         weight_function,
        #     )

        elif self._eq_type == "Poisson":

            return Poisson(
                device,
                self._var_form,
                self.forward,
                self.grad,
                self.gradgrad,
                grid,
                domain_density,
                f_integrated,
                test_func_vals,
                weights,
                grid_boundary,
                normals,
                d1test_func_vals,
                d2test_func_vals,
                d1test_func_vals_bd,
            )

        # elif self._eq_type == "PorousMedium":
        #
        #     return PorousMedium(
        #         self.forward,
        #         self.grad,
        #         grid,
        #         f_integrated,
        #         test_func_vals,
        #         d1test_func_vals,
        #         d2test_func_vals,
        #         d1test_func_vals_bd,
        #         self._var_form,
        #         self._pde_constants,
        #     )

        # elif self._eq_type == "Weak1D":
        #
        #     return Weak1D(
        #         self.forward,
        #         grid,
        #         f_integrated,
        #         d1test_func_vals,
        #         self._var_form,
        #         weight_function,
        #     )

        else:
            raise NotImplementedError(
                f"Equation '{self._eq_type}' not implemented! "
                f"Choose from: {', '.join(self.EQUATION_TYPES)}"
            )
