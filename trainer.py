"""Helper functions to train the model."""

from datetime import datetime
import os
import time

import tensorflow as tf

from global_string import GAN_DISC_OPTIM
from global_string import GAN_GEN_OPTIM


def get_optimizer(model, learning_rate):
    """Returns an Adam optimizer.

    Args:
        model:
        learning_rate:

    Returns:
        An Adam optimizer.
    """
    if model == 'gan':
        return {GAN_GEN_OPTIM: tf.keras.optimizers.Adam(learning_rate),
                GAN_DISC_OPTIM: tf.keras.optimizers.Adam(learning_rate)}
    raise ValueError('Model {} is not supported.'.format(model))


def set_up_ckpt(model, data_loader, optimizer, ckpt_dir, num_ckpt_saved):
    """Sets up the ckeckpoint and ckeckpoint manager.

    Args:
        model:
        data_loader:
        optimizer:
        ckpt_dir:
        num_ckpt_saved:

    Returns:
        A checkpoint manager.
    """
    ckpt = tf.train.Checkpoint(model=model, data_loader=data_loader, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=num_ckpt_saved)
    return ckpt, manager


def get_summary_writer(summary_dir):
    """Returns a summary writer.

    Args:
        summary_dir:

    Returns:
        A summary writer.
    """
    return tf.summary.create_file_writer(summary_dir)


def _write_summary(summary_writer, loss, step):
    """Writes summaries.

    Args:
        summary_writer:
        loss:
        step:

    Returns:

    """
    with summary_writer.as_default():
        for name, val in loss.items():
            tf.summary.scalar(name, val, step)


def _get_iterations(optimizer):
    """Returns number of iterations the optimizer has run.

    Args:
        optimizer:

    Returns:

    """
    if isinstance(optimizer, dict):
        v = list(optimizer.values())[0]
        return v.iterations
    return optimizer.iterations


def _save_img(fake_image, grid, output_dir):
    """Saves images to file.

    Args:
        fake_image:
        grid:
        output_dir:

    Returns:

    """
    file_name = '{}.jpg'.format(datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))
    file = tf.image.encode_jpeg(grid)
    tf.io.write_file(os.path.join(output_dir, file_name), file)


def train(model, loss_func, metrics, optimizer, data_loader, ckpt_manager, summary_writer, num_epoch, print_interval):
    """Trains the model.

    Args:
        model:
        loss_func:
        metrics:
        optimizer:
        data_loader:
        ckpt_manager:
        summary_writer:
        num_epoch:
        print_interval:

    Returns:

    """
    for epoch in range(num_epoch):
        start = time.time()
        metrics.reset()

        for step, data in enumerate(data_loader):
            loss = model.train_step(data, loss_func, optimizer)
            metrics.update(loss)

            if step % print_interval == 0:
                metrics.print_info(epoch + 1, step)
                _write_summary(summary_writer, loss, step=_get_iterations(optimizer))
        metrics.print_info(epoch + 1, None)
        print('Epoch {}. Execution time {:.4f} seconds.\n'.format(epoch + 1, time.time() - start))
        ckpt_manager.save()


def generate(model, data_loader, num, num_per_caption, visualize_tool, num_per_row, output_dir):
    """Generates images.

    Args:
        model:
        data_loader:
        num:
        num_per_caption:
        visualize_tool:
        num_per_row:
        output_dir:

    Returns:

    """
    real_img, embedding, caption = next(data_loader)
    fake_img = model.generate(embedding, num, num_per_caption)
    fake_img, grid = visualize_tool(fake_img, real_img, caption, num, num_per_caption, num_per_row, model)
    _save_img(fake_img, grid, output_dir)
