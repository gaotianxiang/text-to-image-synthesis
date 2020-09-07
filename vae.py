"""Definition of the VAE model."""

import tensorflow as tf

from global_string import VAE_KL_LOSS
from global_string import VAE_RECONSTRUCTION_LOSS
from global_string import VAE_TOTAL_LOSS
from model_interface import Model


class ClassConditionedEncoder(tf.keras.layers.Layer):
    """Class conditioned generator.

    This generator is used by MNIST and FMNIST dataset.

    Attributes:
        _use_condition:
        _model:
        _mu_layer:
        _logvar_layer:
    """

    def __init__(self, use_condition, noise_size):
        """Initializes the object.

        Args:
            use_condition:
            noise_size:
        """
        super().__init__()
        self._use_condition = use_condition
        self._model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, [5, 5], strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(128, [5, 5], strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(noise_size),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()
        ])
        self._mu_layer = tf.keras.layers.Dense(noise_size)
        self._logvar_layer = tf.keras.layers.Dense(noise_size)

    def call(self, image, embedding):
        """Apples the model to the inputs.

        Args:
            image:
            embedding:

        Returns:

        """
        x = self._model(image)
        if self._use_condition:
            x = tf.concat([x, embedding], axis=-1)
        else:
            x = x
        mu = self._mu_layer(x)
        logvar = self._logvar_layer(x)
        return mu, logvar


class EmbeddingConditionedEncoder(tf.keras.layers.Layer):
    """Embedding conditioned generator.

    This generator is used by UCB, Flowers, and MSCOCO datasets.

    Attributes:
        _use_condition:
        _model:
        _mu_layer:
        _logvar_layer:
        _compression (if use_condition is true):
    """

    def __init__(self, use_condition, noise_size, compression_size):
        """Initializes the object.

        Args:
            use_condition:
            noise_size
            compression_size:
        """
        super().__init__()
        self._use_condition = use_condition
        self._model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, [5, 5], strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(256, [5, 5], strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(512, [5, 5], strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(1024, [5, 5], strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(noise_size),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()
        ])
        self._mu_layer = tf.keras.layers.Dense(noise_size)
        self._logvar_layer = tf.keras.layers.Dense(noise_size)

        if use_condition:
            self._compression = tf.keras.layers.Dense(compression_size, use_bias=False)

    def call(self, image, embedding):
        x = self._model(image)
        if self._use_condition:
            x = tf.concat([x, self._compression(embedding)], axis=-1)
        else:
            x = x
        mu = self._mu_layer(x)
        logvar = self._logvar_layer(x)
        return mu, logvar


def _get_encoder(dtst_name, use_condition, noise_sie, compression_size):
    """Returns an encoder.

    Args:
        dtst_name:
        use_condition:
        noise_sie:
        compression_size:

    Returns:
        A generator.
    """
    if dtst_name in ['mnist', 'fmnist']:
        return ClassConditionedEncoder(use_condition, noise_sie)
    return EmbeddingConditionedEncoder(use_condition, noise_sie, compression_size)


class ClassConditionedDecoder(tf.keras.layers.Layer):
    """Class conditioned discriminator.

    This discriminator is used by MNIST and FMNIST datasets.

    Attributes:
        _use_condition:
        _model:
    """

    def __init__(self, use_condition):
        """Initializes the object.

        Args:
            use_condition:
        """
        super().__init__()
        self._use_condition = use_condition
        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(7 * 7 * 256, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Reshape([7, 7, 256]),

            tf.keras.layers.Conv2DTranspose(128, [5, 5], padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2DTranspose(64, [5, 5], strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2DTranspose(1, [5, 5], strides=2, padding='same', use_bias=False, activation='tanh'),
        ])

    def call(self, noise, embedding):
        """Applies the model to the inputs.

        Args:
            noise:
            embedding:

        Returns:

        """
        if self._use_condition:
            x = tf.concat([noise, embedding], axis=-1)
        else:
            x = noise
        return self._model(x)


class EmbeddingConditionedDecoder(tf.keras.layers.Layer):
    """Embedding conditioned discriminator.

    This discriminator is used by CUB, Flowers, MSCOCO datasets.

    Attributes:

    """

    def __init__(self, use_condition, compression_size):
        """Initializes the object.

        Args:
            use_condition:
            compression_size:
        """
        super().__init__()
        self._use_condition = use_condition
        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * 4 * 1024, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Reshape([4, 4, 1024]),

            tf.keras.layers.Conv2DTranspose(512, [5, 5], strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2DTranspose(256, [5, 5], strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2DTranspose(128, [5, 5], strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2DTranspose(3, [5, 5], strides=2, padding='same', use_bias=False, activation='tanh')
        ])

        if use_condition:
            self._compression = tf.keras.layers.Dense(compression_size, use_bias=False)

    def call(self, noise, embedding):
        """Applies the model to the inputs.

        Args:
            noise:
            embedding:

        Returns:

        """
        if self._use_condition:
            x = tf.concat([noise, self._compression(embedding)], axis=-1)
        else:
            x = noise
        return self._model(x)


def _get_decoder(dtst_name, use_condition, compression_size):
    """Returns discriminator.

    Args:
        dtst_name:
        use_condition:
        compression_size:

    Returns:
        A discriminator.
    """
    if dtst_name in ['mnist', 'fmnist']:
        return ClassConditionedDecoder(use_condition)
    return EmbeddingConditionedDecoder(use_condition, compression_size)


class VAE(Model):
    """Generative Adversarial Network.

    Attributes:
        _encoder:
        _decoder:
    """

    def __init__(self, dtst_name, compression_size, noise_size, batch_size, use_condition):
        """Initialize the object.

        Args:
            dtst_name:
            compression_size:
            noise_size:
            batch_size:
            use_condition:
        """
        super().__init__(dtst_name, compression_size, noise_size, batch_size, use_condition)
        self._encoder = _get_encoder(dtst_name, use_condition, noise_size, compression_size)
        self._decoder = _get_decoder(dtst_name, use_condition, compression_size)

    @tf.function
    def train_step(self, inputs, loss_function, optimizer):
        """Trains the model for one step.

        Args:
            inputs:
            loss_function:
            optimizer:

        Returns:

        """
        image, embedding, _ = inputs
        noise = tf.random.normal(shape=[self.batch_size, self.noise_size])

        with tf.GradientTape() as tape:
            mu, logvar = self._encoder(image, embedding, training=True)
            noise = noise * tf.exp(0.5 * logvar) + mu
            image_reconstructed = self._decoder(noise, embedding, training=True)

            reconstruction_loss, kl_loss, total_loss = loss_function(image, image_reconstructed, mu, logvar)

        trainable_variables = self._encoder.trainable_variables + self._decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))

        return {VAE_RECONSTRUCTION_LOSS: reconstruction_loss, VAE_KL_LOSS: kl_loss, VAE_TOTAL_LOSS: total_loss}

    def generate(self, embedding, num, num_per_caption):
        """Generates images.

        Args:
            embedding: Text embeddings.
            num: Number of samplings. Ignored if the use_condition is true.
            num_per_caption: Number of samplings per caption. Ignored if the use_condition is false.

        Returns:

        """
        if not self.use_condition:
            noise = tf.random.normal(shape=[num, self.noise_size])
            fake_img = self._decoder(noise, embedding, training=False)
        else:
            num = embedding.shape[0]
            noise = tf.random.normal(shape=[num_per_caption * num, self.noise_size])
            embedding = tf.tile(embedding, multiples=[num_per_caption, 1])
            fake_img = self._decoder(noise, embedding, training=False)
        return fake_img

    def __str__(self):
        return 'vae'
