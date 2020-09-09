"""Definition of the Flow model."""

import tensorflow as tf

from global_string import FLOW_BPD
from global_string import FLOW_LOG_DET
from global_string import FLOW_LOG_PROB
from global_string import FLOW_TOTAL_LOSS
from model_interface import Model


class ClassConditionedConvAffineCouplingLayer(tf.keras.layers.Layer):
    """Class conditioned convolutional affine coupling layer.

    Attributes:
        _mask
        _use_condition:
        _hidden_conv:
        _log_scale:
        _shift:
    """

    def __init__(self, num_channels_hidden, mask, use_condition):
        """Initializes the object.

        Args:
            num_channels_hidden:
            mask:
            use_condition:
        """
        super().__init__()
        self._mask = mask
        self._use_condition = use_condition
        self._hidden_conv = tf.keras.Sequential([
            layer for num in num_channels_hidden for layer in
            [tf.keras.layers.Conv2D(num, kernel_size=[5, 5], padding='same', activation='relu', use_bias=False),
             tf.keras.layers.BatchNormalization()]
        ])
        self._log_scale = tf.keras.layers.Conv2D(1, kernel_size=[5, 5], padding='same', use_bias=False)
        self._shift = tf.keras.layers.Conv2D(1, kernel_size=[5, 5], padding='same', use_bias=False)

    def call(self, image, embedding, reverse=False):
        """Applies the layer to the inputs.

        Args:
            image:
            embedding:
            reverse:

        Returns:

        """
        if self._use_condition:
            embedding = tf.reshape(embedding, shape=[-1, 1, 1, 10])
            embedding = tf.tile(embedding, multiples=[1, 28, 28, 1])
            x = tf.concat([image, embedding], axis=-1)
        else:
            x = image

        mask = self._mask
        masked_inputs = x * mask
        x = self._hidden_conv(masked_inputs)
        log_scale = self._log_scale(x) * (1 - mask)
        shift = self._shift(x) * (1 - mask)

        if reverse:
            scale = tf.exp(-log_scale)
            return (image - shift) * scale, -log_scale
        else:
            scale = tf.exp(log_scale)
            return image * scale + shift, log_scale


class ClassConditionedFlow(tf.keras.layers.Layer):
    """Class conditioned flow model.

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
        self._model = self._get_coupling_layers(num_layers=4, num_channels_hidden=[32, 32])

    def _get_coupling_layers(self, num_layers, num_channels_hidden):
        """Returns a list of convolutional affine coupling layers.

        Args:
            num_layers:
            num_channels_hidden:

        Returns:

        """
        mask = tf.range(784, dtype=tf.float32)
        mask = tf.reshape(mask, shape=[28, 28, 1])
        mask = mask % 2
        layers = []

        for _ in range(num_layers):
            layers.append(ClassConditionedConvAffineCouplingLayer(num_channels_hidden, mask, self._use_condition))
            mask = 1 - mask
        return layers

    def call(self, image, embedding, reverse=False):
        """Applies the model to the inputs.

        Args:
            image:
            embedding:
            reverse:

        Returns:

        """
        toal_logdet = tf.zeros_like(image)

        if not reverse:
            layers = self._model
        else:
            layers = reversed(self._model)

        for layer in layers:
            image, logdet = layer(image, embedding, reverse)
            toal_logdet += logdet
        return image, toal_logdet


class EmbeddingConditionedConvAffineCouplingLayer(tf.keras.layers.Layer):
    """Embedding conditioned convolutional affine coupling layer.

    Attributes:
        _mask
        _use_condition:
        _hidden_conv:
        _log_scale:
        _shift:
        _compress:
    """

    def __init__(self, num_channels_hidden, compression_size, mask, use_condition):
        """Initializes the object.

        Args:
            num_channels_hidden:
            compression_size:
            mask:
            use_condition:
        """
        super().__init__()
        self._mask = mask
        self._use_condition = use_condition
        self._hidden_conv = tf.keras.Sequential([
            layer for num in num_channels_hidden for layer in
            [tf.keras.layers.Conv2D(num, kernel_size=[5, 5], padding='same', activation='relu', use_bias=False),
             tf.keras.layers.BatchNormalization()]
        ])
        self._log_scale = tf.keras.layers.Conv2D(1, kernel_size=[5, 5], padding='same', use_bias=False)
        self._shift = tf.keras.layers.Conv2D(1, kernel_size=[5, 5], padding='same', use_bias=False)
        if use_condition:
            self._compress = tf.keras.layers.Dense(compression_size, use_bias=False)

    def call(self, image, embedding, reverse=False):
        """Applies the layer to the inputs.

        Args:
            image:
            embedding:
            reverse:

        Returns:

        """
        if self._use_condition:
            embedding = self._compress(embedding)
            embedding = tf.expand_dims(embedding, axis=1)
            embedding = tf.expand_dims(embedding, axis=1)
            embedding = tf.tile(embedding, multiples=[1, 64, 64, 1])
            x = tf.concat([image, embedding], axis=-1)
        else:
            x = image

        mask = self._mask
        masked_inputs = x * mask
        x = self._hidden_conv(masked_inputs)
        log_scale = self._log_scale(x) * (1 - mask)
        shift = self._shift(x) * (1 - mask)

        if reverse:
            scale = tf.exp(-log_scale)
            return (image - shift) * scale, -log_scale
        else:
            scale = tf.exp(log_scale)
            return image * scale + shift, log_scale


class EmbeddingConditionedFlow(tf.keras.layers.Layer):
    """Embedding conditioned flow model.

    Attributes:
        _use_condition:
    """

    def __init__(self, use_condition, compression_size):
        """Initializes the object.

        Args:
            use_condition:
            compression_size:
        """
        super().__init__()
        self._use_condition = use_condition
        self._model = self._get_coupling_layers(num_layers=4, num_channels_hidden=[32, 32],
                                                compression_size=compression_size)

    def _get_coupling_layers(self, num_layers, num_channels_hidden, compression_size):
        """Returns a list of convolutional affine coupling layers.

        Args:
            num_layers:
            num_channels_hidden:
            compression_size:

        Returns:

        """
        mask = tf.range(4096, dtype=tf.float32)
        mask = tf.reshape(mask, shape=[64, 64, 1]) % 2
        layers = []

        for _ in range(num_layers):
            layers.append(EmbeddingConditionedConvAffineCouplingLayer(num_channels_hidden, compression_size, mask,
                                                                      self._use_condition))
            mask = 1 - mask
        return layers

    def call(self, image, embedding, reverse=False):
        """Applies the model to the inputs.

        Args:
            image:
            embedding:
            reverse:

        Returns:

        """
        total_logdet = tf.zeros_like(image)

        if not reverse:
            layers = self._model
        else:
            layers = reversed(self._model)

        for layer in layers:
            image, logdet = layer(image, embedding, reverse)
            total_logdet += logdet
        return image, total_logdet


def _get_flow_model(dtst_name, use_condition, compression_size):
    """Returns a flow model.

    Args:
        dtst_name:
        use_condition:
        compression_size:

    Returns:

    """
    if dtst_name in ['mnist', 'fmnist']:
        return ClassConditionedFlow(use_condition)
    return EmbeddingConditionedFlow(use_condition, compression_size)


class Flow(Model):
    """Flow model.

    Attributes:
        _flow:
    """

    def __init__(self, dtst_name, compression_size, noise_size, batch_size, use_condition):
        """Initializes the object.

        Args:
            dtst_name:
            compression_size:
            noise_size:
            batch_size:
            use_condition:
        """
        super().__init__(dtst_name, compression_size, noise_size, batch_size, use_condition)
        self._flow = _get_flow_model(dtst_name, use_condition, compression_size)

    def _get_noise(self, num):
        if self.dtst_name in ['mnist', 'fmnist']:
            return tf.random.normal(shape=[num, 28, 28, 1])
        return tf.random.normal(shape=[num, 64, 64, 3])

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

        with tf.GradientTape() as tape:
            res, logdet = self._flow(image, embedding, reserve=False, training=True)
            log_likelihood, logdet, total_loss, bpd = loss_function(res, logdet)

        grads = tape.gradient(total_loss, self._flow.trainable_varables)
        optimizer.apply_gradients(zip(grads, self._flow.trainable_varables))

        return {FLOW_LOG_PROB: log_likelihood, FLOW_LOG_DET: logdet, FLOW_TOTAL_LOSS: total_loss, FLOW_BPD: bpd}

    def generate(self, embedding, num, num_per_caption):
        """Generate images.

        Args:
            embedding:
            num:
            num_per_caption:

        Returns:

        """
        if not self.use_condition:
            noise = self._get_noise(num)
            fake_img = self._flow(noise, embedding, reverse=True, training=False)
        else:
            num = embedding.shape[0]
            noise = self._get_noise(num_per_caption * num)
            embedding = tf.tile(embedding, multiples=[num_per_caption, 1])
            fake_img = self._flow(noise, embedding, reverse=True, training=False)
        return fake_img

    def __str__(self):
        return 'flow'