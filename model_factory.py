"""Returns model according to the configuration."""

from gan import GAN


def get_model(model_name, dtst_name, use_condition, batch_size, noise_size, compression_size):
    """Returns model according to model and dataset name.

    Args:
        model_name: Which type of model to use. Either gan, vae, or flow.
        dtst_name: Dataset name. Model structure could different for different model.
        use_condition: Whether to conditioned generative model.
        batch_size:
        noise_size:
        compression_size:

    Returns:
        A model.
    """
    if model_name == 'gan':
        return GAN(dtst_name, compression_size, noise_size, batch_size, use_condition)
    # elif model_name == 'vae':
    #     return _get_vae_model(dtst_name)
    # elif model_name == 'flow':
    #     return _get_flow_model(dtst_name)
    else:
        raise ValueError('Model name {} is not supported.'.format(model_name))
