"""Returns model according to the configuration."""


class Model:
    """Model interfaces for all generative models.

    Attributes:
        dtst_name:
        compression_size:
        noise_size:
        batch_size
    """

    def __init__(self, dtst_name, compression_size, noise_size, batch_size, use_condition):
        """Initializes the object.

        Args:
            dtst_name: Dataset name.
            compression_size: Size to which the embedding vectors will be projected to.
            noise_size: Size of the noise vectors that are used to generate images.
            batch_size:
            use_condition:
        """
        self.dtst_name = dtst_name
        self.compression_rate = compression_size
        self.noise_size = noise_size
        self.batch_size = batch_size
        self.use_condition = use_condition

    def train_step(self, inputs, loss_function, optimizer):
        raise NotImplementedError

    def generate(self, use_condition, ground_truth_img, embedding, caption, num, num_per_caption, visualize_tool,
                 output_dir):
        raise NotImplementedError


def get_models(model_name, dtst_name):
    """Returns model according to model and dataset name.

    Args:
        model_name: Which type of model to use. Either gan, vae, or flow.
        dtst_name: Dataset name. Model structure could different for different model.

    Returns:
        A model.
    """
    if model_name == 'gan':
        return _get_gan_model(dtst_name)
    # elif model_name == 'vae':
    #     return _get_vae_model(dtst_name)
    # elif model_name == 'flow':
    #     return _get_flow_model(dtst_name)
    else:
        raise ValueError('Model name {} is not supported.'.format(model_name))
