"""Data process functions."""

import tensorflow as tf


def _process_mnist(image, label):
    """Processes MNIST dataset.

    Args:
        image:
        label:

    Returns:

    """
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5
    image = tf.expand_dims(image, axis=-1)
    label = tf.one_hot(label, 10)
    return image, label, 0


def _process_flowers(image, embedding, caption):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [64, 64])
    image = (image - 0.5) / 0.5
    return image, embedding, caption


def get_process_func(dtst_name):
    """Returns a dataset process function.

    Args:
        dtst_name:

    Returns:
        A data process function.
    """
    if dtst_name == 'mnist':
        return _process_mnist
    elif dtst_name == 'fmnist':
        return _process_mnist
    elif dtst_name == 'cub':
        return _process_flowers
    raise ValueError('Dataset {} is not supported'.format(dtst_name))
