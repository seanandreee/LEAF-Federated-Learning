"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

from baseline_constants import ACCURACY_KEY
from utils.model_utils import batch_data

class Model(ABC):

    def __init__(self, seed, lr, optimizer=None):
        self.lr = lr
        self.seed = seed
        self._optimizer = optimizer
        self.model = self.create_model()
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    @property
    def optimizer(self):
        """Optimizer to be used by the model."""
        if self._optimizer is None:
            self._optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)
        return self._optimizer

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.

        Returns:
            A Keras model.
        """
        pass

    def set_params(self, model_params):
        self.model.set_weights(model_params)

    def get_params(self):
        return self.model.get_weights()

    def train(self, data, num_epochs=1, batch_size=10):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
        x = self.process_x(data['x'])
        y = self.process_y(data['y'])
        self.model.fit(x, y, epochs=num_epochs, batch_size=batch_size)
        update = self.get_params()
        flops = self.compute_flops()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * flops
        return comp, update

    def compute_flops(self):
        # A placeholder implementation to calculate FLOPs
        return sum([tf.reduce_prod(var.shape).numpy() for var in self.model.trainable_variables])

    def test(self, data):
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        x = self.process_x(data['x'])
        y = self.process_y(data['y'])
        results = self.model.evaluate(x, y)
        return {ACCURACY_KEY: results[1], 'loss': results[0]}

    def close(self):
        # No need to close session in TensorFlow 2.x
        pass

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        pass


class ServerModel:
    def __init__(self, model):
        self.model = model

    @property
    def size(self):
        return self.model.size

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        var_vals = {v.name: v.numpy() for v in self.model.trainable_variables}
        for c in clients:
            for v in c.model.trainable_variables:
                v.assign(var_vals[v.name])

    def save(self, path='checkpoints/model.ckpt'):
        self.model.save_weights(path)

    def close(self):
        pass  # No need to close in TensorFlow 2.x
