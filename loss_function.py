"""Loss functions for different types of model."""

import numpy as np
import tensorflow as tf

from global_string import GAN_DISC_LOSS_FUNC
from global_string import GAN_GEN_LOSS_FUNC


class GANGeneratorLoss:
    """GAN generator loss.

    Attributes:
        _binary_cross_entropy:
    """

    def __init__(self):
        """Initializes the object."""
        super().__init__()
        self._binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')

    def __call__(self, fake_img_real_caption):
        """Calculates the loss value.

        Args:
            fake_img_real_caption:

        Returns:
            Average loss over batch size.
        """
        loss = self._binary_cross_entropy(tf.ones_like(fake_img_real_caption), fake_img_real_caption)
        return tf.reduce_mean(loss)


class GANDiscriminatorLoss:
    """GAN discriminator loss.

    Attributes:
        _binary_cross_entropy:
    """

    def __init__(self):
        """Initializes the object."""
        super().__init__()
        self._binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')

    def __call__(self, real_img_real_caption, real_img_fake_caption, fake_img_real_caption, use_condition):
        """Calculates the loss value.

        Args:
            real_img_real_caption:
            real_img_fake_caption:
            fake_img_real_caption:
            use_condition:

        Returns:
            Average loss over batch size.
        """
        real_loss = self._binary_cross_entropy(tf.ones_like(real_img_real_caption), real_img_real_caption)
        if not use_condition:
            fake_loss = self._binary_cross_entropy(tf.zeros_like(fake_img_real_caption), fake_img_real_caption)
        else:
            fake_loss = (self._binary_cross_entropy(tf.zeros_like(fake_img_real_caption), fake_img_real_caption) +
                         self._binary_cross_entropy(tf.zeros_like(real_img_fake_caption), real_img_fake_caption)) / 2
        return (tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)) / 2


class VAELoss:
    """VAE loss."""

    def __call__(self, x, reconstructed_x, mu, logvar):
        """Calculates the error.

        Args:
            x:
            reconstructed_x:
            mu:
            logvar:

        Returns:

        """
        reconstruction_loss = tf.math.squared_difference(x, reconstructed_x)
        reconstruction_loss = tf.reduce_sum(reconstruction_loss, [1, 2, 3])
        kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1)
        total_loss = reconstruction_loss + kl_loss
        return tf.reduce_mean(reconstruction_loss), tf.reduce_mean(kl_loss), tf.reduce_mean(total_loss)


class FlowLoss:
    """Flow loss."""

    def __call__(self, res, log_det):
        """Calculates the loss value.

        Args:
            res:
            log_det:

        Returns:

        """
        _, h, w, d = res.get_shape()
        log_prob = -0.5 * (tf.square(res)) + tf.math.log(2 * np.pi)
        log_prob = tf.reduce_sum(log_prob, axis=[1, 2, 3])
        log_prob = tf.reduce_mean(log_prob)
        log_det = tf.reduce_sum(log_det, axis=[1, 2, 3])
        log_det = tf.reduce_mean(log_det)
        total_loss = -(log_prob + log_det)
        bpd = total_loss / (h * w * d * tf.math.log(2))
        return log_prob, log_det, total_loss, bpd


def get_loss_func(model):
    """Returns loss function(s) according to the model type.

    Args:
        model:

    Returns:
        Loss function(s).
    """
    if model == 'gan':
        return {GAN_GEN_LOSS_FUNC: GANGeneratorLoss(),
                GAN_DISC_LOSS_FUNC: GANDiscriminatorLoss()}
    elif model == 'vae':
        return VAELoss()
    elif model == 'flow':
        return FlowLoss()
    raise ValueError('Model {} is not supported.'.format(model))
