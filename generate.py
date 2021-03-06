"""Generates images."""

from absl import app
from absl import flags

from data_loader import get_data_loader
from data_process_func import get_process_func
from model_factory import get_model
from trainer import generate
from trainer import set_up_ckpt
from visualize import get_visualize_tool

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

flags.DEFINE_string('ckpt_dir', default=None, help='Directory to which the checkponts will be stored.')

flags.DEFINE_integer('noise_size', default=100,
                     help='Size of noise vectors. Used in GAN and VAE. Ignored if the model is flow.')
flags.DEFINE_integer('compression_size', default=128, help='Size to which the embedding vectors will be projected to.')

flags.DEFINE_enum('visualize_tool', default='fake_only', enum_values=['fake_only', 'fake_real', 'real_only'],
                  help='Which visualize model to use.')
flags.DEFINE_string('output_dir', default=None, help='Directory to which the generated images will be stored.')
flags.DEFINE_integer('num', default=64, help='Number of samplings. Ignored if the use_condition is true.')
flags.DEFINE_integer('num_per_caption', default=8,
                     help='Number of samplings per caption. Ignored if the use_condition is false.')
flags.DEFINE_integer('num_per_row', default=10, help='Number of images per row.')


def main(argv):
    del argv

    data_loader = get_data_loader(dataset_name=FLAGS.dtst_name, tfrecord_dir=FLAGS.tfrecord_dir,
                                  num_caption_per_image=FLAGS.num_caption_per_image,
                                  process_func=get_process_func(FLAGS.preprocess), shuffle=FLAGS.shuffle,
                                  batch_size=FLAGS.batch_size, drop_remainder=FLAGS.drop_remainder)
    model = get_model(model_name=FLAGS.model, dtst_name=FLAGS.dtst_name, use_condition=FLAGS.use_condition,
                      batch_size=FLAGS.batch_size, noise_size=FLAGS.noise_size, compression_size=FLAGS.compression_size)
    ckpt, ckpt_manager = set_up_ckpt(model, None, None, FLAGS.ckpt_dir, 1, generate=True)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    visualize_tool = get_visualize_tool(FLAGS.visualize_tool)
    generate(model, data_loader, num=FLAGS.num,
             num_per_caption=FLAGS.num_per_caption, visualize_tool=visualize_tool, num_per_row=FLAGS.num_per_row,
             output_dir=FLAGS.output_dir)


if __name__ == '__main__':
    flags.mark_flags_as_required(['dtst_name', 'tfrecord_dir', 'preprocess', 'model', 'ckpt_dir', 'output_dir'])
    app.run(main)
