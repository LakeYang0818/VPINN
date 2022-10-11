from typing import Any, List, Union

from torch import nn

# Local imports
from .DataSet import DataSet
from .Grid import Grid
from .Variational_forms import *


def get_activation_funcs(n_layers: int, cfg: Union[str, dict] = None) -> List[Any]:
    """Extracts the activation functions from the config"""

    def return_function(name: str):
        if name in ["Linear", "linear", "lin", "None"]:
            return None
        elif name in ["sigmoid", "Sigmoid"]:
            return torch.sigmoid
        elif name in ["relu", "ReLU"]:
            return torch.relu
        elif name in ["sin", "sine"]:
            return torch.sin
        elif name in ["cos", "cosine"]:
            return torch.cos
        elif name in ["tanh"]:
            return torch.tanh
        elif name in ["abs"]:
            return torch.abs
        else:
            raise ValueError(f"Unrecognised activation function {name}!")

    funcs = [None] * (n_layers + 1)

    if cfg is None:
        return funcs
    elif isinstance(cfg, str):
        return [return_function(cfg)] * (n_layers + 1)
    elif isinstance(cfg, dict):
        for val in cfg.keys():
            if val in [0]:
                funcs[0] = return_function(cfg[0])
            elif val in [-1]:
                funcs[-1] = return_function(cfg[-1])
            else:
                funcs[val - 1] = return_function(cfg[val])

        return funcs
    else:
        raise ValueError(f"Unrecognised argument {cfg} for 'activation_funcs'!")


class NeuralNet(nn.Module):
    """A variational physics-informed neural net. Inherits from the nn.Module parent."""

    VAR_FORMS = {0, 1, 2}

    EQUATION_TYPES = {
        "Burger",
        "Helmholtz",
        "Poisson",
        "PorousMedium",
        "Burger",
        "Weak1D",
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
        eq_type,
        var_form,
        *,
        pde_constants: dict = None,
        learning_rate: float = 0.001,
        optimizer: str = "Adam",
        input_size: int,
        output_size: int,
        num_layers: int,
        nodes_per_layer: int,
        activation_funcs: dict = None,
    ):

        """Initialises the neural net.

        :param architecture: the neural net architecture
        :param eq_type: the equation type of the PDE in question
        :param var_form: the variational form to use for the loss function
        :param pde_constants: the constants for the pde in use
        :param learning_rate: the learning rate for the optimizer
        :param activation_func: the activation function to use

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

        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.input_dim = input_size
        self.output_dim = output_size
        self.hidden_dim = num_layers
        self.activation_func = get_activation_funcs(num_layers, activation_funcs)
        architecture = [input_size] + [nodes_per_layer] * num_layers + [output_size]

        # Add the neural net layers
        self.layers = nn.ModuleList()
        for i in range(len(architecture) - 1):
            self.layers.append(nn.Linear(architecture[i], architecture[i + 1]))

        # Get Adam optimizer
        self.optimizer = self.OPTIMIZERS[optimizer](self.parameters(), lr=learning_rate)

        # Get equation type and variational form
        self._eq_type = eq_type
        self._var_form = var_form

        # Get equation parameters
        self._pde_constants = pde_constants

    # ... Evaluation functions .........................................................................................

    # The model forward pass
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation_func[i](self.layers[i](x))
        x = self.layers[-1](x)
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
        y = self.forward(x)
        first_derivative = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y), create_graph=True
        )[0]
        second_derivative = torch.autograd.grad(
            first_derivative,
            x,
            grad_outputs=torch.ones_like(x),
            create_graph=requires_grad,
        )[0]

        return second_derivative

    # ... Loss functions ...............................................................................................

    def boundary_loss(self, training_data: DataSet) -> torch.Tensor:

        """Calculates the loss on the domain boundary.

        :param training_data: the model training data
        :return: the boundary loss
        """

        # Conduct a forward pass on the training data
        u = self.forward(training_data.coords)

        # Calculate the pointwise error
        return torch.nn.functional.mse_loss(u, training_data.data)

    def variational_loss(
        self,
        grid: Grid,
        f_integrated: DataSet,
        test_func_vals: DataSet,
        d1test_func_vals: DataSet = None,
        d2test_func_vals: DataSet = None,
        d1test_func_vals_bd: DataSet = None,
        weight_function=lambda x: 1,
    ) -> torch.Tensor:

        """Calculates the variational loss on the grid interior.

        :param grid: the grid
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
                self.forward,
                self.grad,
                grid,
                f_integrated,
                d1test_func_vals,
                self._pde_constants,
            )

        elif self._eq_type == "Helmholtz":

            return Helmholtz(
                self.forward,
                self.grad,
                self.gradgrad,
                grid,
                f_integrated,
                test_func_vals,
                d1test_func_vals,
                d2test_func_vals,
                d1test_func_vals_bd,
                self._var_form,
                self._pde_constants,
                weight_function,
            )

        elif self._eq_type == "Poisson":

            return Poisson(
                self.forward,
                self.grad,
                self.gradgrad,
                grid,
                f_integrated,
                test_func_vals,
                d1test_func_vals,
                d2test_func_vals,
                d1test_func_vals_bd,
                self._var_form,
                weight_function,
            )

        elif self._eq_type == "PorousMedium":

            return PorousMedium(
                self.forward,
                self.grad,
                grid,
                f_integrated,
                test_func_vals,
                d1test_func_vals,
                d2test_func_vals,
                d1test_func_vals_bd,
                self._var_form,
                self._pde_constants,
            )

        elif self._eq_type == "Weak1D":

            return Weak1D(
                self.forward,
                grid,
                f_integrated,
                d1test_func_vals,
                self._var_form,
                weight_function,
            )

        else:
            raise ValueError(
                f"Unrecognized equation type '{self._eq_type}'! "
                f"Choose from: {', '.join(self.EQUATION_TYPES)}"
            )
