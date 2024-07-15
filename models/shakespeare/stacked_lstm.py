import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from model import Model
from utils.language_utils import letter_to_vec, word_to_indices

class ClientModel(Model):
    def __init__(self, seed, lr, seq_len, num_classes, n_hidden):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        # Define the input layer
        features = tf.keras.Input(shape=(self.seq_len,), dtype=tf.int32)
        embedding = layers.Embedding(input_dim=self.num_classes, output_dim=8)(features)
        
        # Define the LSTM layers using Keras
        lstm_layer = layers.RNN([layers.LSTMCell(self.n_hidden) for _ in range(2)], return_sequences=False)(embedding)
        
        # Define the dense layer using Keras
        pred = layers.Dense(units=self.num_classes)(lstm_layer)

        # Define the model
        model = tf.keras.Model(inputs=features, outputs=pred)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])

        return model

    def process_x(self, raw_x_batch):
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [letter_to_vec(c) for c in raw_y_batch]
        y_batch = np.array(y_batch)
        return y_batch
