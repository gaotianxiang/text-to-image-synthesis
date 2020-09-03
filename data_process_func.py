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


def _process_vae_mnist(image, label):
    """Processes MNIST and FMNIST datasets for VAE.

    Args:
        image:
        label:

    Returns:

    """
    image = tf.cast(image, tf.float32)
    image = image / 255
    image = tf.expand_dims(image, axis=-1)
    label = tf.one_hot(label, 10)
    return image, label, 0


def _process_rgb_images(image, embedding, caption):
    """Processes RGB images.

    Args:
        image:
        embedding:
        caption:

    Returns:

    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [64, 64])
    image = (image - 0.5) / 0.5
    return image, embedding, caption


def _process_vae_rgb_images(image, embedding, caption):
    """Processes RGB images for VAE.

    Args:
        image:
        embedding:
        caption:

    Returns:

    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [64, 64])
    return image, embedding, caption


def _process_rgb_images_make_sure_3_channels(image, embedding, caption):
    """Processes RGB images and makes sure the processed images have 3 channels.

    Args:
        image:
        embedding:
        caption:

    Returns:

    """
    image = tf.tile(image, [1, 1, 3])[:, :, :3]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [64, 64])
    image = (image - 0.5) / 0.5
    return image, embedding, caption


def _process_vae_rgb_images_make_sure_3_channels(image, embedding, caption):
    """Processes RGB images and makes sure the processed images have 3 channels for VAE.

    Args:
        image:
        embedding:
        caption:

    Returns:

    """
    image = tf.tile(image, [1, 1, 3])[:, :, :3]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [64, 64])
    return image, embedding, caption


def get_process_func(preprocess):
    """Returns a dataset process function.

    Args:
        preprocess:

    Returns:
        A data process function.
    """
    if preprocess == 'mnist':
        return _process_mnist
    elif preprocess == 'image':
        return _process_rgb_images
    elif preprocess == 'image_make_sure_3_channels':
        return _process_rgb_images_make_sure_3_channels
    elif preprocess == 'vae_mnist':
        return _process_vae_mnist
    elif preprocess == 'vae_image':
        return _process_vae_rgb_images
    elif preprocess == 'vae_image_3':
        return _process_vae_rgb_images_make_sure_3_channels
    raise ValueError('Preprocess {} is not supported'.format(preprocess))
