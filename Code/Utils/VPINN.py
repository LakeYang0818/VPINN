import tensorflow as tf
from tensorflow import keras
from typing import Sequence

# Local imports
from .data_types import DataGrid, DataSet, Grid
from .test_functions import test_function
from .var_forms import var_sum


class VPINN(keras.Model):
    """ A variational physics-informed neural net. Inherits from the keras.Model parent."""

    VAR_FORMS = {1, 2, 3}

    EQUATION_TYPES = {
        "Poisson",
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
                 var_form: int = 1,
                 eq_type: str = 'Poisson',
                 activation: str = 'relu',
                 data_type: tf.DType = tf.dtypes.float64):

        """Creates a new variational physics-informed neural network (VPINN). The loss function contains both a
        strong residual (the values of the network evaluated on training data) and a variational loss, calculated
        by integrating the network against test functions over the domain.

        Args:
             f_integrated :DataSet: the values of the external forcing integrated against all the test functions on the
                grid. It is used to calculate the
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
            ValueError: if an invalid 'var_form' argument is passed.
            ValueError: if an invalid 'eq_type' argument is passed
        """

        if var_form not in self.VAR_FORMS:
            raise ValueError(f"Unrecognized variational_form  "
                             f"'{var_form}'! "
                             f"Choose from: [1, 2, 3]")

        if eq_type not in self.EQUATION_TYPES:
            raise ValueError(f"Unrecognized equation type "
                             f"'{eq_type}'! "
                             f"Choose from: {', '.join(self.EQUATION_TYPES)}")

        super().__init__()

        # The external forcing integrated against every test function over the entire grid
        self.f_integrated = f_integrated

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
            print(f"Adding layer {i + 1} of {len(architecture)} ...")
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

    # ... Evaluation functions ...................................................................

    def evaluate(self, x: tf.Tensor) -> tf.Tensor:
        """Evaluates the neural net"""

        return self.neural_net(x)

    # @tf.function
    def grad_net(self, x: tf.Tensor) -> tf.Tensor:
        """Calculates the first derivative of the neural net.
        Returns a tf.Tensor of derivatives in each coordinate direction."""

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.evaluate(x)
        grad = tape.gradient(y, x)

        return grad

    # @tf.function
    def gradgrad_net(self, x: tf.Tensor) -> tf.Tensor:
        """Calculates the second derivative of the neural net. Returns a tf.Tensor with
        the values of the second derivatives in each direction"""

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.grad_net(x)
        gradgrad = tape.gradient(y, x)

        return gradgrad

    # ... Loss functions ...................................................................

    # @tf.function
    def calculate_strong_residual(self, x: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculates the strong residual on the training data, which are the values of the exact solution on the
        boundary.
        """

        return tf.math.reduce_mean(tf.math.squared_difference(self.evaluate(x), y_pred))

    # @tf.function
    def calculate_variational_loss(self,
                                   quadrature_data: DataGrid,
                                   quadrature_data_scaled: Sequence[DataGrid],
                                   jacobians: Sequence[float],
                                   boundary: Sequence,
                                   n_test_functions: int,
                                   pde_params: dict = None) -> tf.Tensor:
        """Calculates the total variational loss over the grid.

        Args:
            quadrature_data :DataGrid: the quadrature grid, containing grid points and corresponding weights
            quadrature_data_scaled :Sequence[DataGrid]: the quadrature data scaled to each grid element
            jacobians: the jacobians of the scaling transforms
            n_test_functions :int: the number of test functions against which to integrate
            pde_params :dict: the constants used in the differential equation
        Returns:
            the variational loss
        """

        u_integrated = tf.stack([var_sum(u=self.evaluate, du=self.grad_net, ddu=self.gradgrad_net,
                                         n_test_func=i, quads=quadrature_data, quads_scaled=quadrature_data_scaled,
                                         jacobians=jacobians, grid_boundary=boundary,
                                         var_form=self.var_form, eq_type=self.eq_type,
                                         dtype=self.data_type, pde_params=pde_params)
                                 for i in range(1, n_test_functions + 1)])

        return tf.math.reduce_mean(tf.math.squared_difference(u_integrated, self.f_integrated.data))

    #@tf.function
    def compute_loss(self, x, y_pred, *, quads: DataGrid, quads_scaled: Sequence[DataGrid], jacobians: Sequence[float],
                     grid_boundary: Sequence, n_test_functions: int, pde_params: dict = None) -> tf.Tensor:
        """Calculates the total loss, which is training loss + variational loss.

        Args:
            x: the input values
            y_pred: the corresponding values to predict
            quads: the quadrature data
            quads_scaled: the quadrature data scaled to each grid element
            jacobians: the jacobians of the scaling transforms
            grid_boundary: the grid boundary
            n_test_functions: the number of test functions to use
            pde_params: the constants used in the differential equations
        Return:
            the total loss, where strong residual and variational loss are weighted
        """

        loss_s = self.calculate_strong_residual(x, y_pred)
        loss_v = self.calculate_variational_loss(quads, quads_scaled, jacobians,
                                                 grid_boundary, n_test_functions, pde_params) if self.loss_weight > 0 \
            else tf.constant([0])

        return tf.add(tf.multiply(tf.constant([self.loss_weight], dtype=self.data_type), loss_s), loss_v)

    # ... Training function ...................................................................

    # @tf.function
    def train(self, x, y, *, quads: DataGrid, quads_scaled: Sequence[DataGrid], jacobians: Sequence[float],
              grid_boundary: Sequence, n_test_functions: int, pde_params: dict = None):
        """A single training step of the model"""

        if pde_params is None:
            pde_params = {'Helmholtz': 1, 'Burger': 0}
        with tf.GradientTape() as tape:
            # Logits for this minibatch, tracked by the tape
            logits = self.evaluate(x)

            # Compute the loss value for this minibatch.
            loss = self.compute_loss(x, y, quads=quads, quads_scaled=quads_scaled, jacobians=jacobians,
                                     grid_boundary=grid_boundary, n_test_functions=n_test_functions,
                                     pde_params=pde_params)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, self.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss

