"""Metrics used in the training."""

import tensorflow as tf

from global_string import GAN_DISC_LOSS
from global_string import GAN_GEN_LOSS
from global_string import FLOW_BPD
from global_string import FLOW_LOG_DET
from global_string import FLOW_LOG_PROB
from global_string import FLOW_TOTAL_LOSS
from global_string import VAE_KL_LOSS
from global_string import VAE_RECONSTRUCTION_LOSS
from global_string import VAE_TOTAL_LOSS


class Metrics:
    """Class that stores all the tf metric objects.

    Attributes:
        _metrics:
    """

    def __init__(self, metrics):
        """Initializes the object.

        Args:
            metrics:
        """
        self._metrics = metrics

    def reset(self):
        """Reset the metrics."""
        for _, metric in self._metrics.items():
            metric.reset_states()

    def update(self, loss):
        """Updates metrics.

        Args:
            loss:

        Returns:

        """
        for k, v in loss.items():
            self._metrics[k].update_state(v)

    def print_info(self, epoch, step):
        """Prints out the metric states.

        Args:
            epoch:
            step:

        Returns:

        """
        print('Epoch {} Step {} {}'.format(epoch, step, self.__str__()))

    def __str__(self):
        return ' '.join(['{} {:.4f}'.format(k, v.result()) for k, v in self._metrics.items()])


def get_metrics(model):
    """Return metrics according to the model.

    Args:
        model:

    Returns:
        A metric object.
    """
    if model == 'gan':
        return Metrics({GAN_GEN_LOSS: tf.keras.metrics.Mean(), GAN_DISC_LOSS: tf.keras.metrics.Mean()})
    elif model == 'vae':
        return Metrics({VAE_RECONSTRUCTION_LOSS: tf.keras.metrics.Mean(), VAE_KL_LOSS: tf.keras.metrics.Mean(),
                        VAE_TOTAL_LOSS: tf.keras.metrics.Mean()})
    elif model == 'flow':
        return Metrics({FLOW_LOG_DET: tf.keras.metrics.Mean(), FLOW_LOG_PROB: tf.keras.metrics.Mean(),
                        FLOW_TOTAL_LOSS: tf.keras.metrics.Mean(), FLOW_BPD: tf.keras.metrics.Mean()})
    raise ValueError('Model {} is not supported.'.format(model))
