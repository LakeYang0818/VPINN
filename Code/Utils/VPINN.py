import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Any, Union, Tuple, Sequence

# Local imports
from .functions import diff_eq
from .data_types import DataGrid, DataSet, Grid
from .utils import rescale_quads


class VPINN(keras.Model):
    """ A variational physics-informed neural net. Inherits from the keras.Model parent."""

    VAR_FORMS = {1, 2, 3}

    EQUATION_TYPES = {
        "Poisson1D",
        "Poisson2D",
        "Helmholtz",
        "Burger"
    }

    def __init__(self,
                 f_integrated: DataSet,
                 *,
                 input_dim: int,
                 architecture: Sequence[int],
                 loss_weight: float,
                 learning_rate: float = 0.01,
                 var_form: int,
                 eq_type: str,
                 activation: str = 'relu',
                 data_type: tf.DType = tf.dtypes.float64):

        """Creates a new variational physics-informed neural network (VPINN).
        Args:
             f_integrated :DataSet: the values of the external forcing integrated against all the test functions on the grid. It is used to calculate the
                variational loss. The coordinates of the DataSet are the test function numbers,
                and the datasets are the values of the integrals.
            input_dim :int: the dimension of the input data
            architecture :Sequence: the neural net architecture
            loss_weight :float: the relative weight of the strong residual to variational residual
            var_form :int: the variational form to use
            eq_type :str: the equation type
            activation :str: the activation function of the neural net
            data_type :DType: the data type of the layer weights. Default is float64.
        Raises:
            ValueError: if an invalid `type` argument is passed.
        """

        if (var_form not in self.VAR_FORMS):
            raise ValueError(f"Unrecognized variational_form  "
                             f"'{type[0]}'! "
                             f"Choose from: {', '.join(self.VAR_FORMS)}")

        if (eq_type not in self.EQUATION_TYPES):
            raise ValueError(f"Unrecognized equation type "
                             f"'{type[1]}'! "
                             f"Choose from: {', '.join(self.EQUATION_TYPES)}")

        super().__init__()

        # The external forcing integrated over the grid against every test function
        # This a two-dimensional array, where the outer dimension is the grid size and
        # the inner dimension is the number of test functions (see below)
        self.f_integrated = f_integrated

        # Number of grid elements
        self.n_grid_elements = np.shape(self.f_integrated)[0]

        # Number of test functions
        self.n_test_functions = np.shape(self.f_integrated[0])[0]

        # Relative weight
        self.loss_weight = loss_weight

        # Variational form and equation type
        self.var_form = var_form
        self.eq_type = eq_type

        # Initialise the net
        xavier = tf.keras.initializers.GlorotUniform()
        self.neural_net = keras.Sequential()

        # Add layers with the specified architecture
        for i in range(len(architecture)):
            print(f"Adding layer {i} of {len(architecture)} ...")
            if i == 0:
                self.neural_net.add(keras.layers.Dense(architecture[i],
                                                       dtype=data_type,
                                                       kernel_initializer=xavier,
                                                       input_dim=input_dim,
                                                       activation=activation))

            else:
                self.neural_net.add(keras.layers.Dense(architecture[i],
                                                       dtype=data_type,
                                                       kernel_initializer=xavier,
                                                       activation=activation))
        self.dropout = keras.layers.Dropout(0.5)
        self.neural_net.summary()

        self.data_type = data_type

        # Get the optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Track the loss
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # ... Evaluation functions ...................................................................

    def evaluate(self, x: tf.Tensor) -> tf.Tensor:
        """Evaluates the neural net"""

        return self.neural_net(x)

    def grad_net(self, x: tf.Tensor) -> tf.Tensor:
        """Calculates the first derivative of the neural net.
        Returns a tf.Tensor of derivatives in each coordinate direction."""
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.evaluate(x)
        grad = tape.gradient(y, x)

        return grad

    def gradgrad_net(self, x: tf.Tensor) -> tf.Tensor:
        """Calculates the second derivative of the neural net. Returns a tf.Tensor with
        the values of the second derivatives in each direction"""
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.grad_net(x)
        gradgrad = tape.gradient(y, x)

        return gradgrad

    # ... Loss functions ...................................................................

    @tf.function
    def calculate_strong_residual(self, x: tf.Tensor, y_pred: tf.Tensor):
        """Calculates the strong residual on the training data, which are the values of the exact solution on the
        boundary.
        """

        return tf.math.reduce_mean(tf.math.squared_difference(self.evaluate(x), y_pred))

    @tf.function
    def calculate_variational_loss(self,
                                   grid: Grid,
                                   quadrature_data: DataGrid,
                                   n_test_functions: int):
        """Calculates the variational loss on the grid points.

        Args:
            grid :Grid: the domain of integration
            quadrature_data :DataGrid: the quadrature grid, containing grid points and corresponding weights
            n_test_functions :int: the number of test functions against which to integrate
        """
        # u_integrated = []
        # for i in range(n_test_functions):
        #     res = 0
        #     for point in grid:
        #         grid_element = ...
        #         quads_rescaled = rescale_quads(quadrature_data, grid_element)
        #         du = self.d_net(quads_rescaled)
        #         ddu = self.dd_net(quads_rescaled)
        #         res += integrate_u(du, ddu, quads_rescaled, grid_element, type)
        #     u_integrated.append(res)
        #
        # return tf.math.reduce_mean(tf.math.squared_difference(u_integrated, self.f_integrated.data))

        return tf.constant([0.5], dtype=self.data_type)

    @tf.function
    def compute_loss(self, x, y_pred, *, grid: Grid, quads: DataGrid, n_test_functions: int):
        """Calculates the total loss, which is training loss + variational loss."""
        loss_s = self.calculate_strong_residual(x, y_pred)
        loss_v = self.calculate_variational_loss(grid, quads, n_test_functions)
        loss = self.loss_weight * loss_s + loss_v

        return loss

    # ... Training function ...................................................................

    @tf.function
    def train(self, x, y, *, grid: Grid, quads: DataGrid, n_test_functions: int):
        """A single training step of the model"""
        with tf.GradientTape() as tape:

            # Logits for this minibatch, tracked by the tape
            logits = self.evaluate(x)

            # Compute the loss value for this minibatch.
            loss = self.compute_loss(x, y,grid=grid, quads=quads, n_test_functions=n_test_functions)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, self.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.train_acc_metric.update_state(y, logits)

        return loss

    @tf.function
    def test_step(self, x, y):
        val_logits = self.evaluate(x)
        self.val_acc_metric.update_state(y, val_logits)
