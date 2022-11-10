from typing import Any, List, Union

from torch import nn

from .Variational_forms import *


class NeuralNet(nn.Module):

    # Pytorch optimizers
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

    # Pytorch activation functions.
    # Pairs of activation functions and whether they are part of the torch.nn module, in which case they must be called
    # via func(*args, **kwargs)(x).
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

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        num_layers: int,
        nodes_per_layer: dict,
        activation_funcs: dict,
        biases: dict,
        optimizer: str = "Adam",
        learning_rate: float = 0.001,
        eq_type: str = "Poisson",
        var_form: int = 0,
        pde_constants: dict = None,
        **__,
    ):
        """

        :param input_size: the number of input values
        :param output_size: the number of output values
        :param num_layers: the number of hidden layers
        :param nodes_per_layer: a dictionary specifying the number of nodes per layer
        :param activation_funcs: a dictionary specifying the activation functions to use
        :param biases: a dictionary containing the initialisation parameters for the bias
        :param optimizer: the name of the optimizer to use. Default is the torch.optim.Adam optimizer.
        :param learning_rate: the learning rate of the optimizer. Default is 1e-3.
        :param __: Additional model parameters (ignored)
        """

        super().__init__()
        self.flatten = nn.Flatten()

        self.input_dim = input_size
        self.output_dim = output_size
        self.hidden_dim = num_layers

        # Get architecture, activation functions, and layer bias
        self.architecture = self._get_architecture(
            input_size, output_size, num_layers, nodes_per_layer
        )
        self.activation_funcs = self._get_activation_funcs(num_layers, activation_funcs)
        self.bias = self._get_bias(num_layers, biases)

        # Add the neural net layers
        self.layers = nn.ModuleList()
        for i in range(len(self.architecture) - 1):
            layer = nn.Linear(
                self.architecture[i], self.architecture[i + 1], self.bias[i] is not None
            )

            # Initialise the biases of the layers with a uniform distribution
            if self.bias[i] not in [None, "default"]:
                torch.nn.init.uniform_(layer.bias, self.bias[i][0], self.bias[i][1])

            self.layers.append(layer)

        # Get the optimizer
        self.optimizer = self.OPTIMIZERS[optimizer](self.parameters(), lr=learning_rate)

        # Get equation type and variational form
        self._eq_type = eq_type
        self._var_form = var_form

        # Get equation parameters
        self._pde_constants = pde_constants

    def _get_architecture(
        self, input_size: int, output_size: int, n_layers: int, cfg: dict
    ) -> List[int]:

        # Apply default to all hidden layers
        _nodes = [cfg.get("default")] * n_layers

        # Update layer-specific settings
        _layer_specific = cfg.get("layer_specific", {})
        for layer_id, layer_size in _layer_specific.items():
            _nodes[layer_id] = layer_size

        return [input_size] + _nodes + [output_size]

    def _get_activation_funcs(self, n_layers: int, cfg: dict) -> List[callable]:

        """Extracts the activation functions from the config. The config is a dictionary containing the
        default activation function, and a layer-specific entry detailing exceptions from the default. 'None' entries
        are interpreted as linear layers.

        .. Example:
            activation_funcs:
              default: relu
              layer_specific:
                0: ~
                2: tanh
                3:
                  name: HardTanh
                  args:
                    - -2  # min_value
                    - +2  # max_value
        """

        def _single_layer_func(layer_cfg: Union[str, dict]) -> callable:

            """Return the activation function from an entry for a single layer"""

            # Entry is a single string
            if isinstance(layer_cfg, str):
                _f = self.ACTIVATION_FUNCS[layer_cfg.lower()]
                if _f[1]:
                    return _f[0]()
                else:
                    return _f[0]

            # Entry is a dictionary containing args and kwargs
            elif isinstance(layer_cfg, dict):
                _f = self.ACTIVATION_FUNCS[layer_cfg.get("name").lower()]
                if _f[1]:
                    return _f[0](
                        *layer_cfg.get("args", ()), **layer_cfg.get("kwargs", {})
                    )
                else:
                    return _f[0]

            elif layer_cfg is None:
                _f = self.ACTIVATION_FUNCS["linear"][0]

            else:
                raise ValueError(f"Unrecognized activation function {cfg}!")

        # Use default activation function on all layers
        _funcs = [_single_layer_func(cfg.get("default"))] * (n_layers + 1)

        # Change activation functions on specified layers
        _layer_specific = cfg.get("layer_specific", {})
        for layer_id, layer_cfg in _layer_specific.items():
            _funcs[layer_id] = _single_layer_func(layer_cfg)

        return _funcs

    def _get_bias(self, n_layers: int, cfg: dict) -> List[Any]:

        """Extracts the bias initialisation settings from the config. The config is a dictionary containing the
        default, and a layer-specific entry detailing exceptions from the default. 'None' entries
        are interpreted as unbiased layers. 'default' values mean the bias is initialised using the pytorch
        default, U[-1/sqrt(k), 1/sqrt(k)], with k = num in_features

        .. Example:
            biases:
              default: ~
              layer_specific:
                0: [-1, 1]
                3: [2, 3]
        """

        # Use the default value on all layers
        biases = [cfg.get("default")] * (n_layers + 1)

        # Amend bias on specified layers
        _layer_specific = cfg.get("layer_specific", {})
        for layer_id, layer_bias in _layer_specific.items():
            biases[layer_id] = layer_bias

        return biases

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

        elif self._eq_type == "dummy":

            return dummy(
                device,
                self.forward,
                grid,
                domain_density,
                f_integrated,
                test_func_vals,
                weights,
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
