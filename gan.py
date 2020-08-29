"""Definition of the GAN model."""

import tensorflow as tf

from model_factory import Model


class ClassConditionedGenerator(tf.keras.layers.Layer):
    """Class conditioned generator.

    This generator is used by MNIST and FMNIST dataset.

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
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Reshape([7, 7, 256]),

            tf.keras.layers.Conv2DTranspose(128, [5, 5], padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2DTranspose(64, [5, 5], strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2DTranspose(1, [5, 5], strides=2, padding='same', use_bias=False, activation='tanh'),
        ])

    def call(self, noise, embedding):
        """Apples the model to the inputs.

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


class EmbeddingConditionedGenerator(tf.keras.layers.Layer):
    """Embedding conditioned generator.

    This generator is used by UCB, Flowers, and MSCOCO datasets.

    Attributes:
        _use_condition:
        _compression_size:
        _model:
        _compression (if use_condition is true):
    """

    def __init__(self, use_condition, compression_size):
        """Initializes the object.

        Args:
            use_condition:
            compression_size:
        """
        super().__init__()
        self._use_condition = use_condition
        self._compression_size = compression_size
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
        if self._use_condition:
            x = tf.concat([noise, self._compression(embedding)], axis=-1)
        else:
            x = noise
        return self._model(x)


def _get_generator(dtst_name, use_condition, compression_size):
    """Returns a generator.

    Args:
        dtst_name:
        use_condition:
        compression_size:

    Returns:
        A generator.
    """
    if dtst_name in ['mnist', 'fmnist']:
        return ClassConditionedGenerator(use_condition)
    return EmbeddingConditionedGenerator(use_condition, compression_size)


class ClassConditionedDiscriminator(tf.keras.layers.Layer):
    """Class conditioned discriminator.

    This discriminator is used by MNIST and FMNIST datasets.

    Attributes:
        _use_condition:
        _model:
        _final_layer
    """

    def __init__(self, use_condition):
        """Initializes the object.

        Args:
            use_condition:
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
        ])
        self._final_layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, [1, 1], strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(1, [7, 7], strides=1, padding='valid', activation='tanh'),
            tf.keras.layers.Flatten()
        ])

    def call(self, image, embedding):
        """Applies the model to the inputs.

        Args:
            image:
            embedding:

        Returns:

        """
        x = self._model(image)
        if self._use_condition:
            embedding = tf.expand_dims(embedding, 1)
            embedding = tf.expand_dims(embedding, 1)
            embedding = tf.tile(embedding, multiples=[1, 7, 7, 1])
            x = tf.concat([x, embedding], axis=-1)
        x = self._final_layer(x)
        return x


class EmbeddingConditionedDiscriminator(tf.keras.layers.Layer):
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
        ])
        self._final_layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(1024, [1, 1], strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(1, [4, 4], strides=1, padding='valid', activation='tanh'),
            tf.keras.layers.Flatten()
        ])
        if use_condition:
            self._compression = tf.keras.layers.Dense(compression_size, use_bias=False)

    def call(self, image, embedding):
        """Applies the model to the inputs.

        Args:
            image:
            embedding:

        Returns:

        """
        x = self._model(image)
        if self._use_condition:
            embedding = self._compression(embedding)
            embedding = tf.expand_dims(embedding, 1)
            embedding = tf.expand_dims(embedding, 1)
            embedding = tf.tile(embedding, multiples=[1, 4, 4, 1])
            x = tf.concat([x, embedding], axis=-1)
        x = self._final_layer(x)
        return x


def _get_discriminator(dtst_name, use_condition, compression_size):
    """Returns discriminator.

    Args:
        dtst_name:
        use_condition:
        compression_size:

    Returns:
        A discriminator.
    """
    if dtst_name in ['mnist', 'fmnist']:
        return ClassConditionedDiscriminator(use_condition)
    return EmbeddingConditionedDiscriminator(use_condition, compression_size)


class GAN(Model):
    """Generative Adversarial Network.

    Attributes:
        dtst_name:
        compression_size:
        noise_size:
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
        self._generator = _get_generator(dtst_name, use_condition, compression_size)
        self._discriminator = _get_discriminator(dtst_name, use_condition, compression_size)

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
        generator_loss_func = loss_function['generator_loss']
        discriminator_loss_func = loss_function['discriminator_loss']
        generator_optimizer = optimizer['generator']
        discriminator_optimizer = optimizer['discriminator']
        noise = tf.random.normal(shape=[self.batch_size, self.noise_size])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_image = self._generator(noise, embedding)

            real_img_real_caption = self._discriminator(image, embedding)
            real_img_fake_caption = self._discriminator(image, embedding[::-1])
            fake_img_real_caption = self._discriminator(fake_image, embedding)

            gen_loss = generator_loss_func(fake_img_real_caption, self.use_condition)
            disc_loss = discriminator_loss_func(real_img_real_caption, real_img_fake_caption, fake_img_real_caption,
                                                self.use_condition)
        grads_generator = gen_tape.gradient(gen_loss, self._generator.trainable_variables)
        grads_discriminator = disc_tape.gradient(disc_loss, self._discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(grads_generator, self._generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(grads_discriminator, self._discriminator.trainable_variables))

        return {'gen_loss': gen_loss, 'disc_loss': disc_loss}

    def generate(self, use_condition, ground_truth_img, embedding, caption, num, num_per_caption, visualize_tool,
                 output_dir):
        """Generates images.

        Args:
            use_condition: Whether to do conditional generation.
            ground_truth_img: Ground truth images.
            embedding: Text embeddings.
            caption: Captions.
            num: Number of samplings. Ignored if the use_condition is true.
            num_per_caption: Number of samplings per caption. Ignored if the use_condition is false.
            visualize_tool: A visualize tool.
            output_dir: Directory to which the generated images will be stored.

        Returns:

        """
        if not use_condition:
            noise = tf.random.normal(shape=[num, self.noise_size])
            img = self._generator(noise, None)
            visualize_tool(img, None, ground_truth_img, use_condition, output_dir)
        else:
            num = embedding.shape[0]
            noise = tf.random.normal(shape=[num_per_caption * num, self.noise_size])
            embedding = tf.tile(embedding, multiples=[num_per_caption, 1])
            fake_img = self._generator(noise, embedding)
            visualize_tool(fake_img, caption, ground_truth_img, use_condition, output_dir)