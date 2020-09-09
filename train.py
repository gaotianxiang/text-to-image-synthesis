"""Trains the model."""

from absl import app
from absl import flags

from data_process_func import get_process_func
from data_loader import get_data_loader
from loss_function import get_loss_func
from metrics import get_metrics
from model_factory import get_model
from trainer import get_optimizer
from trainer import get_summary_writer
from trainer import set_up_ckpt
from trainer import train

FLAGS = flags.FLAGS
flags.DEFINE_enum('dtst_name', default=None, enum_values=['mnist', 'fmnist', 'cub', 'flower', 'coco'],
                  help='Name of the dataset that will be used to train the model.')
flags.DEFINE_string('tfrecord_dir', default=None, help='Directory to which the dataset is stored.')
flags.DEFINE_integer('num_caption_per_image', default=5, help='Number of captions per image.')
flags.DEFINE_integer('shuffle', default=0, help='Dataset shuffle buffer size. Zero means no shuffle.')
flags.DEFINE_integer('batch_size', default=64, help='Batch size.')
flags.DEFINE_bool('drop_remainder', default=True, help='Whether to drop remainder in the data loader.')
flags.DEFINE_enum('preprocess', default=None,
                  enum_values=['mnist', 'image', 'image_make_sure_3_channels', 'mnist_flow', 'rgb_flow', 'rgb_flow_3'],
                  help='Number of channels of the dataset images.')

flags.DEFINE_enum('model', default=None, enum_values=['gan', 'vae', 'flow'], help='Which type of model will be used.')
flags.DEFINE_bool('use_condition', default=False, help='Whether to use conditioned generative model.')

flags.DEFINE_float('lr', default=0.001, help='Learning rate.')

flags.DEFINE_string('ckpt_dir', default=None, help='Directory to which the checkponts will be stored.')
flags.DEFINE_integer('num_ckpt_saved', default=2, help='Max number of checkpoints will be saved.')

flags.DEFINE_string('summary_dir', default=None, help='Directory to which the summaries will be stored.')

flags.DEFINE_integer('num_epoch', default=100, help='Number of epochs the model will be trained.')
flags.DEFINE_integer('print_interval', default=100, help='Number of iterations between two log information.')

flags.DEFINE_integer('noise_size', default=100,
                     help='Size of noise vectors. Used in GAN and VAE. Ignored if the model is flow.')
flags.DEFINE_integer('compression_size', default=128, help='Size to which the embedding vectors will be projected to.')


def main(argv):
    del argv

    data_loader = get_data_loader(dataset_name=FLAGS.dtst_name, tfrecord_dir=FLAGS.tfrecord_dir,
                                  num_caption_per_image=FLAGS.num_caption_per_image,
                                  process_func=get_process_func(FLAGS.preprocess), shuffle=FLAGS.shuffle,
                                  batch_size=FLAGS.batch_size, drop_remainder=FLAGS.drop_remainder)
    model = get_model(model_name=FLAGS.model, dtst_name=FLAGS.dtst_name, use_condition=FLAGS.use_condition,
                      batch_size=FLAGS.batch_size, noise_size=FLAGS.noise_size, compression_size=FLAGS.compression_size)
    optimizer = get_optimizer(FLAGS.model, FLAGS.lr)
    _, ckpt_manager = set_up_ckpt(model, data_loader, optimizer, FLAGS.ckpt_dir, FLAGS.num_ckpt_saved)
    loss_func = get_loss_func(FLAGS.model)
    summary_writer = get_summary_writer(FLAGS.summary_dir)
    metrics = get_metrics(FLAGS.model)
    train(model, loss_func, metrics, optimizer, data_loader, ckpt_manager, summary_writer, FLAGS.num_epoch,
          FLAGS.print_interval)


if __name__ == '__main__':
    flags.mark_flags_as_required(['dtst_name', 'tfrecord_dir', 'preprocess', 'model', 'ckpt_dir', 'summary_dir'])
    app.run(main)
