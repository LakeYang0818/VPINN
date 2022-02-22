import torch
from torch import nn
from .test_functions import test_function
# Local imports
from .data_types import Grid, DataGrid, DataSet


class VPINN(nn.Module):
    """ A variational physics-informed neural net. Inherits from the keras.Model parent."""

    VAR_FORMS = {1, 2, 3}

    EQUATION_TYPES = {
        "Poisson",
        "Helmholtz",
        "Burger"
    }

    def __init__(self, architecture, activation_func, loss_weight):
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

        # Get the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # Initialize the loss tracker
        self._loss_tracker: dict = {'iter': [],
                                    'total_loss': [],
                                    'loss_b': [],
                                    'loss_v': []}

        self._loss_weight = loss_weight


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

    # ... Evaluation functions .........................................................................................

    # The model forward pass
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.activation_func(self.layers[i](x))
        x = self.layers[-1](x)
        return x

    # Computes the first derivative of the model output. x can be a single tensor or a stack of tensors
    def grad(self, x, *, requires_grad: bool = True):
        x = torch.autograd.grad(self.forward(x), x, grad_outputs=torch.ones_like(x), create_graph=requires_grad)[0]
        return x

    # Computes the second derivative of the model output. x can be a single tensor or a stack of tensors
    def gradgrad(self, x, *, requires_grad: bool = True):
        first_derivative = torch.autograd.grad(self.forward(x), x,
                                               grad_outputs=torch.ones_like(x), create_graph=True)[0]
        second_derivative = torch.autograd.grad(first_derivative, x,
                                                grad_outputs=torch.ones_like(x), create_graph=requires_grad)[0]

        return second_derivative

    # ... Loss functions ...............................................................................................

    # Calculate the loss on the domain boundary
    def boundary_loss(self, training_data: DataSet):
        u_x = self.forward(training_data.coords)
        loss = torch.nn.functional.mse_loss(u_x, training_data.data)

        return loss

    # Calculates the variational loss on the interior (only variational form 1 for now)
    # To do: don't know if all the cloning is necessary. Also haven't checked if this works for 2d
    def variational_loss(self, grid: Grid, f_integrated, test_func_vals, n_test_functions: int):
        loss_v = torch.tensor(0.0, requires_grad=True)
        Laplace_x = self.gradgrad(grid.interior, requires_grad=True)
        for i in range(0, n_test_functions):
            q = torch.tensor(0.0, requires_grad=True)
            q = q.clone() + torch.sum(torch.mul(Laplace_x, test_func_vals.data[i]))/len(grid.interior)
            q = q.clone() + f_integrated.data[i]
            q = torch.square(q.clone())/(n_test_functions)
            loss_v = loss_v + q
            del q

        return loss_v

    # Training loop (to do: move this back to main and only have a single training step be part of the model class)
    def train_custom(self, training_data, f_integrated, test_func_vals, grid, n_test_functions, n_iterations):
        loss_glob = 0.0
        loss_b_glob = 0.0
        loss_v_glob = 0.0
        for it in range(n_iterations):

            self.optimizer.zero_grad()
            loss_v = self.variational_loss(grid, f_integrated, test_func_vals, n_test_functions)
            loss_b = self.boundary_loss(training_data)
            loss = loss_b + loss_v
            loss.backward(retain_graph=True)

            self.optimizer.step()
            loss_glob = loss.item()
            loss_b_glob = loss_b.item()
            loss_v_glob = loss_v.item()

            del loss
            del loss_b
            del loss_v

            if it % 10 == 0:
                self.update_loss_tracker(it, loss_glob, loss_b_glob, loss_v_glob)
            if it % 100 == 0:
                print(f"Iteration {it}: total loss: {loss_glob}, loss_b: {loss_b_glob}, loss_v: {loss_v_glob}")
