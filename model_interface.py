"""Model interfaces."""

import tensorflow as tf


class Model(tf.keras.Model):
    """Model interfaces for all generative models.

    Attributes:
        dtst_name:
        compression_size:
        noise_size:
        batch_size:
        use_condition:
    """

    def __init__(self, dtst_name, compression_size, noise_size, batch_size, use_condition):
        super().__init__()
        """Initializes the object.

        Args:
            dtst_name: Dataset name.
            compression_size: Size to which the embedding vectors will be projected to.
            noise_size: Size of the noise vectors that are used to generate images.
            batch_size:
            use_condition:
        """
        self.dtst_name = dtst_name
        self.compression_rate = compression_size
        self.noise_size = noise_size
        self.batch_size = batch_size
        self.use_condition = use_condition

    def train_step(self, inputs, loss_function, optimizer):
        raise NotImplementedError

    def generate(self, embedding, num, num_per_caption):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
