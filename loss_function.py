"""Loss functions for different types of model."""

import tensorflow as tf

from global_string import GAN_DISC_LOSS_FUNC
from global_string import GAN_GEN_LOSS_FUNC


class GANGeneratorLoss(tf.keras.losses.Loss):
    """GAN generator loss.

    Attributes:
        _binary_cross_entropy:
    """

    def __init__(self):
        """Initializes the object."""
        super().__init__()
        self._binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')

    def call(self, fake_img_real_caption, use_condition):
        """Calculates the loss value.

        Args:
            fake_img_real_caption:
            use_condition:

        Returns:
            Average loss over batch size.
        """
        loss = self._binary_cross_entropy(tf.ones_like(fake_img_real_caption), fake_img_real_caption)
        return tf.reduce_mean(loss)


class GANDiscriminatorLoss(tf.keras.losses.Loss):
    """GAN discriminator loss.

    Attributes:
        _binary_cross_entropy:
    """

    def __init__(self):
        """Initializes the object."""
        super().__init__()
        self._binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')

    def call(self, real_img_real_caption, real_img_fake_caption, fake_img_real_caption, use_condition):
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
            fake_loss = (self._binary_cross_entropy(tf.zeros_like(fake_img_real_caption), fake_img_real_caption) + \
                         self._binary_cross_entropy(tf.zeros_like(real_img_fake_caption), real_img_fake_caption)) / 2
        return tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)


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

    raise ValueError('Model {} is not supported.'.format(model))
