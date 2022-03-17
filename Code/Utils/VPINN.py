import torch
from torch import nn

# Local imports
from .Datatypes.DataSet import DataSet
from .Datatypes.Grid import Grid
from .Variational_forms import *


class VPINN(nn.Module):
    """ A variational physics-informed neural net. Inherits from the nn.Module parent."""

    VAR_FORMS = {0, 1, 2}

    EQUATION_TYPES = {
        "Burger",
        "Helmholtz",
        "Poisson",
        "PorousMedium",
        "Burger",
        "Weak1D"
    }

    def __init__(self, architecture, eq_type, var_form, *,
                 pde_constants: dict = None,
                 learning_rate: float = 0.001,
                 activation_func=torch.sin):

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

        if eq_type not in self.EQUATION_TYPES:
            raise ValueError(f"Unrecognized equation type "
                             f"'{eq_type}'! "
                             f"Choose from: {', '.join(self.EQUATION_TYPES)}")

        if var_form not in self.VAR_FORMS:
            raise ValueError(f"Unrecognized variational_form  "
                             f"'{var_form}'! "
                             f"Choose from: [1, 2, 3]")

        super(VPINN, self).__init__()
        self.flatten = nn.Flatten()
        self.input_dim = architecture[0]
        self.output_dim = architecture[-1]
        self.hidden_dim = len(architecture) - 2
        self.activation_func = activation_func

        # Add the neural net layers
        self.layers = nn.ModuleList()
        for i in range(len(architecture) - 1):
            self.layers.append(nn.Linear(architecture[i], architecture[i + 1]))

        # Get Adam optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Initialize the loss tracker dictionary, which can be used to later evaluate the training progress
        self._loss_tracker: dict = {'iter': [],
                                    'total_loss': [],
                                    'loss_b': [],
                                    'loss_v': []}

        # Get equation type and variational form
        self._eq_type = eq_type
        self._var_form = var_form

        # Get equation parameters
        self._pde_constants = pde_constants

    # Return the model loss values
    @property
    def loss_tracker(self) -> dict:
        return self._loss_tracker

    # Updates the loss tracker with a time value and the loss values
    def update_loss_tracker(self, it, total_loss, loss_b, loss_v):
        self._loss_tracker['iter'].append(it)
        self._loss_tracker['total_loss'].append(total_loss)
        self._loss_tracker['loss_b'].append(loss_b)
        self._loss_tracker['loss_v'].append(loss_v)

    def reset_loss_tracker(self):
        self._loss_tracker = {'iter': [], 'total_loss': [], 'loss_b': [], 'loss_v': []}

    # ... Evaluation functions .........................................................................................

    # The model forward pass
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation_func(self.layers[i](x))
        x = self.layers[-1](x)
        return x

    # Computes the first derivative of the model output. x can be a single tensor or a stack of tensors
    def grad(self, x, *, requires_grad: bool = True):
        y = self.forward(x)
        x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=requires_grad)[0]
        return x

    # Computes the second derivative of the model output. x can be a single tensor or a stack of tensors
    def gradgrad(self, x, *, requires_grad: bool = True):
        y = self.forward(x)
        first_derivative = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        second_derivative = torch.autograd.grad(first_derivative, x,
                                                grad_outputs=torch.ones_like(x), create_graph=requires_grad)[0]

        return second_derivative

    # ... Loss functions ...............................................................................................

    def boundary_loss(self, training_data: DataSet):
        """Calculates the loss on the domain boundary.

        :param training_data:
        :return:
        """

        # Conduct a forward pass on the training data
        u = self.forward(training_data.coords)

        # Calculate the pointwise error
        loss_b = torch.nn.functional.mse_loss(u, training_data.data)

        return loss_b

    def variational_loss(self,
                         grid: Grid,
                         f_integrated: DataSet,
                         test_func_vals: DataSet,
                         d1test_func_vals: DataSet = None,
                         d2test_func_vals: DataSet = None,
                         d1test_func_vals_bd: DataSet = None,
                         weight_function=lambda x: 1):
        """ Calculates the variational loss on the interior.

        :param grid:
        :param f_integrated:
        :param test_func_vals:
        :param d1test_func_vals:
        :param d2test_func_vals:
        :param d1test_func_vals_bd:
        :param weight_function: 
        :return:
        """
        if self._eq_type == 'Burger':
            return Burger(self.forward,
                          self.grad,
                          grid,
                          f_integrated,
                          test_func_vals,
                          d1test_func_vals,
                          self._var_form,
                          self._pde_constants)

        elif self._eq_type == 'Helmholtz':
            return Helmholtz(self.forward,
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
                             weight_function)

        elif self._eq_type == 'Poisson':
            return Poisson(self.forward,
                           self.grad,
                           self.gradgrad,
                           grid,
                           f_integrated,
                           test_func_vals,
                           d1test_func_vals,
                           d2test_func_vals,
                           d1test_func_vals_bd,
                           self._var_form,
                           weight_function)

        elif self._eq_type == 'PorousMedium':
            return PorousMedium(self.forward,
                                self.grad,
                                self.gradgrad,
                                grid,
                                f_integrated,
                                test_func_vals,
                                d1test_func_vals,
                                d2test_func_vals,
                                d1test_func_vals_bd,
                                self._var_form,
                                self._pde_constants)

        elif self._eq_type == 'Weak1D':
            return Weak1D(self.forward,
                          grid,
                          f_integrated,
                          d1test_func_vals,
                          self._var_form,
                          weight_function)

        else:
            raise ValueError(f'Unrecognised equation type {self._eq_type}!')
