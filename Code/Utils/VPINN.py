import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Any, Sequence

# Local imports
from .data_types import DataGrid, DataSet
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
                 quadrature_data: DataGrid,
                 quadrature_data_scaled: Sequence[DataGrid],
                 jacobians: Sequence,
                 boundary: Sequence,
                 *,
                 input_dim: int,
                 architecture: Sequence[int],
                 loss_weight: float,
                 learning_rate: float = 0.01,
                 var_form: int = 1,
                 eq_type: str = 'Poisson',
                 pde_params: dict = None,
                 activation: Any,
                 data_type: tf.DType = tf.dtypes.float64):

        """Creates a new variational physics-informed neural network (VPINN). The loss function contains both a
        strong residual (the values of the network evaluated on training data) and a variational loss, calculated
        by integrating the network against test functions over the domain.

        Args:
            f_integrated :DataSet: the values of the external forcing integrated against all the test functions on the
                grid. It is used to calculate the
                variational loss. The coordinates of the DataSet are the test function numbers,
                and the datasets are the values of the integrals.
            quadrature_data: the quadrature points
            quadrature_data_scaled: the quadrature points scaled to the grid
            jacobians: the jacobians of the coordinate transforms
            input_dim :int: the dimension of the input data
            architecture :Sequence: the neural net architecture
            loss_weight :float: the relative weight of the strong residual to variational residual
            var_form :int: the variational form to use
            eq_type :str: the equation type
            pde_params: the pde parameters
            activation: the activation function of the neural net
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

        super(VPINN, self).__init__()

        # The external forcing integrated against every test function over the entire grid
        self._f_integrated = f_integrated
        self._n_test_functions = f_integrated.size

        # The quadrature data
        self._quadrature_data = quadrature_data
        self._quadrature_data_scaled = quadrature_data_scaled

        # The coordinate transform jacobians
        self._jacobians = jacobians

        # The grid boundary
        self._boundary = boundary

        # Relative weight
        self._loss_weight = loss_weight

        # Variational form, equation type and equation params
        self._var_form = var_form
        self._eq_type = eq_type
        self._pde_params = pde_params

        # Initialise the net
        self._neural_net = keras.Sequential()

        # Add layers with the specified architecture
        for i in range(len(architecture)):
            print(f"Adding layer {i + 1} of {len(architecture)} ...")
            if i == 0:
                self._neural_net.add(tf.keras.Input(shape=(architecture[i],)))
                self._neural_net.add(keras.layers.Dense(architecture[i],
                                                        dtype=data_type,
                                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                            stddev=np.sqrt(2/(architecture[i]+architecture[i+1]), dtype=np.float64)
                                                        ),
                                                        bias_initializer=tf.keras.initializers.Zeros(),
                                                        input_dim=input_dim,
                                                        activation=activation))

            elif i < len(architecture)-1:
                self._neural_net.add(keras.layers.Dense(architecture[i],
                                                        dtype=data_type,
                                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                            stddev=np.sqrt(2/(architecture[i]+architecture[i+1]), dtype=np.float64)
                                                        ),
                                                        bias_initializer=tf.keras.initializers.Zeros(),
                                                        activation=activation))
            # Add output layer
            else:
                self._neural_net.add(keras.layers.Dense(architecture[-1],
                                                        dtype=data_type,
                                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                            stddev=np.sqrt(2/(architecture[i-1]+architecture[i]), dtype=np.float64)
                                                        ),
                                                        bias_initializer=tf.keras.initializers.Zeros(),
                                                        input_dim=architecture[-2],
                                                        activation='linear'))
        self._neural_net.summary()

        self._data_type = data_type

        # Get the optimizer
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Initialize the loss tracker
        self._loss_tracker: dict = {'iter': [],
                                    'total_loss': [],
                                    'loss_b': [],
                                    'loss_v': []}

    # ... Evaluation functions ...................................................................

    # Return the model loss values
    @property
    def loss_tracker(self) -> dict:
        return self._loss_tracker

    def update_loss_tracker(self, it, total_loss, loss_b, loss_v):
        self._loss_tracker['iter'].append(it)
        self._loss_tracker['total_loss'].append(total_loss)
        self._loss_tracker['loss_b'].append(loss_b)
        self._loss_tracker['loss_v'].append(loss_v)

    # ... Evaluation functions ...................................................................

    @tf.function
    def evaluate(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Evaluates the neural net"""

        return self._neural_net(x, training=training)

    @tf.function
    def grad_net(self, x: tf.Tensor) -> tf.Tensor:
        """Calculates the first derivative of the neural net.
        Returns a tf.Tensor of derivatives in each coordinate direction."""

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(x)
            y = self.evaluate(x)
        grad = tape.gradient(y, x)

        return grad

    @tf.function
    def gradgrad_net(self, x: tf.Tensor) -> tf.Tensor:
        """Calculates the second derivative of the neural net. Returns a tf.Tensor with
        the values of the second derivatives in each direction"""

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(x)
            y = self.grad_net(x)
        gradgrad = tape.gradient(y, x)

        return gradgrad

    # ... Loss function ...................................................................
    def calculate_variational_loss(self, *, n_test_func) -> tf.Tensor:
        """Calculates the total variational loss over the grid."""

        u_integrated = var_sum(u=self.evaluate, du=self.grad_net, ddu=self.gradgrad_net,
                                         n_test_func=n_test_func, quads=self._quadrature_data,
                                         quads_scaled=self._quadrature_data_scaled,
                                         jacobians=self._jacobians, grid_boundary=self._boundary,
                                         var_form=self._var_form, eq_type=self._eq_type,
                                         dtype=self._data_type, pde_params=self._pde_params)

        return tf.math.squared_difference(u_integrated, self._f_integrated.data[n_test_func])

    def calculate_boundary_loss(self, training_data: DataSet) -> tf.Tensor:
        total_loss = tf.stack([
            tf.math.squared_difference(self.evaluate(x_train, training=True), y_train)
        for _, (x_train, y_train) in enumerate(training_data)])
        return tf.math.reduce_mean(total_loss)

    def calculate_loss(self, *, x, y):

        loss_b = self.calculate_boundary_loss(x=x, y=y)
        loss_v = self.calculate_variational_loss()
        loss = tf.add(tf.multiply(tf.constant([self._loss_weight], dtype=self._data_type), loss_b), loss_v)

        return loss

    # ... Model training ...................................................................
    def train(self, training_data: DataSet, n_iterations: int):
        """Trains the model using a training DataSet, for n_iterations."""

        for it in range(n_iterations):
            with tf.GradientTape() as tape:
                loss_b = self.calculate_boundary_loss(training_data)
                loss_v = 0
                for i in range(1, self._n_test_functions):
                    loss_v += self.calculate_variational_loss(n_test_func=i)
                loss = loss_b + self._loss_weight*loss_v
            grads = tape.gradient(loss, self.trainable_variables)

            self._optimizer.apply_gradients(zip(grads, self.trainable_variables))
            if it % 10==0:
                self.update_loss_tracker(it, loss, loss_b, loss_v)
                print(f"Current iteration: {it}; loss: {loss}")
            # Update the variational loss every 10 iterations
            # if it % 10 == 0:
            #     with tf.GradientTape() as tape:
            #
            #         loss_v = self.calculate_variational_loss()
            #
            #     grads = tape.gradient(loss_v, self.trainable_variables)
            #     loss = tf.add(tf.multiply(tf.constant([self._loss_weight], dtype=self._data_type),
            #                               loss_b), loss_v)
            #     self._optimizer.apply_gradients(zip(grads, self.trainable_variables))
            #     self.update_loss_tracker(it, loss, loss_b, loss_v)

            if it % 100 == 0:
                print(f"Current iteration: {it}; loss: {loss}")
