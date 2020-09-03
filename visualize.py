"""Visualize tool."""

import numpy as np
import tensorflow as tf


def _post_process(image, model):
    """Post-processes the image.

    Args:
        image:
        model:

    Returns:

    """
    if model == 'gan':
        image = tf.clip_by_value(image, -1, 1)
        image = image * 127.5 + 127.5
        image = tf.cast(image, tf.uint8)
        return image
    elif model == 'vae':
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        return image
    raise ValueError('Model {} is not supported.'.format(model))


def _make_grid(imarray, cols=4, pad=1, padval=255):
    """Lays out a [N, H, W, C] image array as a single image grid."""
    pad = int(pad)
    if pad < 0:
        raise ValueError('pad must be non-negative')
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = N // cols + int(N % cols != 0)
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray = np.pad(imarray, pad_arg, 'constant', constant_values=padval)
    H += pad
    W += pad
    grid = (imarray
            .reshape(rows, cols, H, W, C)
            .transpose(0, 2, 1, 3, 4)
            .reshape(rows * H, cols * W, C))
    if pad:
        grid = grid[:-pad, :-pad]
    return grid


def _fake_only_visualize(fake_img, real_img, caption, num, num_per_caption, num_per_row, model):
    """Visualizes the images in fake-only mode.

    Args:
        fake_img:
        real_img:
        caption:
        num:
        num_per_caption:
        num_per_row:
        model:

    Returns:
        tf.Tensor (tf.uint8).
    """
    fake_img = _post_process(fake_img, model)
    grid = _make_grid(fake_img, cols=num_per_row)
    grid = tf.convert_to_tensor(grid, dtype=tf.uint8)
    return fake_img, grid


def _fake_real_visualize(fake_img, real_img, caption, num, num_per_caption, num_per_row, model):
    """Visualizes the fake and real images.

    Args:
        fake_img:
        real_img:
        caption:
        num:
        num_per_caption:
        num_per_row:
        model:

    Returns:

    """
    fake_img = _post_process(fake_img, model)
    real_img = _post_process(real_img, model)
    fake_grid = _make_grid(fake_img, cols=num_per_row)
    fake_grid = tf.convert_to_tensor(fake_grid, dtype=tf.uint8)
    real_grid = _make_grid(real_img, cols=num_per_row)
    real_grid = tf.convert_to_tensor(real_grid, dtype=tf.uint8)
    grid = tf.concat([real_grid, tf.ones([1] + fake_grid.shape[1:], dtype=tf.uint8) * 255, fake_grid], axis=0)
    return fake_img, grid


def get_visualize_tool(mode):
    if mode == 'fake_only':
        return _fake_only_visualize
    elif mode == 'fake_real':
        return _fake_real_visualize
    raise ValueError('Visualize model {} is not supported.')
