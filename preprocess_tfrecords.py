"""Preprocesses and stores datasets in TFRecord format."""

from absl import app
from absl import flags
import glob
import os
import pickle

import ray
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_enum('dataset', None, enum_values=['mnist', 'fmnist', 'cub', 'flowers', 'mscoco'], help='Dataset.')
flags.DEFINE_enum('split', None, enum_values=['train', 'val'],
                  help='Which split to pick. Either \'train\' or \'val\'. Only valid for dataset mnist and fmnist.')
flags.DEFINE_string('output_dir', None, help='Directory to which the records will be stored.')
flags.DEFINE_string('embedding_dir', None, help='Directory to which the embeddings are stored. '
                                                'Required if the dataset is either cub, flowers, or mscoco.')
flags.DEFINE_string('caption_dir', None, help='Directory to which the captions are stored.')
flags.DEFINE_string('img_dir', None, help='Directory to which the images are stored. '
                                          'Required if the dataset is either cub, flowers, or mscoco.')
flags.DEFINE_integer('num_shards', None, help='Number of TFRecord files that will be generated for the dataset.')


def chunkify(original_list, num_shards):
    """Cut the original list into several chunks for parallel processing.

    FLAGS:
        original_list: The original list.
        num_shards: Number of chunks.

    Returns:
        List of list. Each sub-list is an individual chunk.
    """
    size = len(original_list) // num_shards
    start = 0
    results = []
    for _ in range(num_shards - 1):
        results.append(original_list[start:start + size])
        start += size
    results.append(original_list[start:])
    return results


def _bytes_feature(value):
    """Converts values into TF compatible bytes feature.

    FLAGS:
        value: An input value.

    Returns:
        A TF feature.
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy(
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_pickle(path_to_pickle):
    """Reads a pickle file.

    FLAGS:
        path_to_pickle:
    """
    with open(path_to_pickle, 'rb') as f:
        res = pickle.load(f, encoding='bytes')
    return res


def read_txt(path_to_txt):
    """Reads a txt file.

    FLAGS:
        path_to_txt:

    Returns:
        List of strings.
    """
    with open(path_to_txt, 'r') as f:
        content = f.readlines()
    return content


def read_captions(caption_dir):
    """Read captions.

    FLAGS:
        caption_dir:

    Returns:
        Python dictionary. Keys are filenames and values are the corresponding captions.
    """
    res = {}
    for file in glob.iglob('{}/**/*.txt'.format(caption_dir), recursive=True):
        res[os.path.basename(file).split('.')[0] + '.jpg'] = read_txt(file)
    return res


def read_img(path_to_img):
    """Reads a image with tensorflow.

    FLAGS:
        path_to_img:

    Returns:

    """
    return tf.image.decode_image(tf.io.read_file(path_to_img))


class TFRecordDataset:
    """TFRecord dataset interface."""

    def build_tfrecord(self):
        """Builds TFRecords."""
        raise NotImplementedError


class ClassConditionedDataset(TFRecordDataset):
    """Interfaces for class conditioned dataset.

    This interface is implemeted by MNIST and Fashion-MNIST.

    Attributes:
        output_dir:
        split:
        num_shards:
    """

    def __init__(self, output_dir, split='train', num_shards=8):
        """Initializes the object.

        FLAGS:
            output_dir:
            split:
            num_shards:
        """
        self.output_dir = output_dir
        self.split = split
        self.num_shards = num_shards

    def _get_img_label(self):
        """Returns the image and label array according to split.

        Returns:
            image array: uint8 numpy array with shape (num_samples, 28, 28).
            labels array: uint8 numpy array with shape (num_samples,).
        """
        raise NotImplementedError

    def _get_tfexample(self, index, imgs, labels):
        """Gets a TFExample.

        FLAGS:
            index: An index.
            imgs: Image array.
            labels: label array.

        Returns:
            A TFExample.
        """
        img = tf.convert_to_tensor(imgs[index], dtype=tf.uint8)
        label = tf.convert_to_tensor(labels[index], dtype=tf.uint8)
        feature = {
            'image': _bytes_feature(tf.io.serialize_tensor(img)),
            'label': _bytes_feature(tf.io.serialize_tensor(label))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @ray.remote
    def _build_single_tfrecord(self, chunk, imgs, labels, path_to_record):
        """Builds a single TFRecord.

        FLAGS:
            chunk: List of indices that will be processed and written in this TFRecord file.
            imgs; Image array.
            labels: Label array.
            path_to_record: Path to which the record file will be stored.
        """
        print('Start to build tf records for {}.'.format(path_to_record))
        with tf.io.TFRecordWriter(path_to_record) as writer:
            for index in chunk:
                try:
                    tf_example = self._get_tfexample(index, imgs, labels)
                except IOError:
                    print('Path: {}. Index: {}'.format(path_to_record, index))
                else:
                    writer.write(tf_example.SerializeToString())
        print('Finished building tf records for {}.'.format(path_to_record))

    def _get_features(self, indices_chunks, imgs, labels):
        """Creates sub-tasks.

        FLAGS:
            indices_chunks: List of indices chunks.
            imgs: Image array.
            labels: Label array.

        Returns:
            List of ray sub-tasks.
        """
        return [
            self._build_single_tfrecord.remote(
                self,
                chunk,
                imgs,
                labels,
                os.path.join(self.output_dir, self.split,
                             'tfrecord_{}_{}.tfrecords'.format(str(i + 1).zfill(4), str(self.num_shards).zfill(4))))
            for i, chunk in enumerate(indices_chunks)
        ]

    def build_tfrecord(self):
        imgs, labels = self._get_img_label()
        indices_chunks = chunkify([*range(len(labels))], self.num_shards)
        ray.init()
        features = self._get_features(indices_chunks, imgs, labels)
        ray.get(features)


class MNISTTFRecord(ClassConditionedDataset):

    def _get_img_label(self):
        """Returns the image and label array according to split.

        Returns:
            image array: uint8 numpy array with shape (num_samples, 28, 28).
            labels array: uint8 numpy array with shape (num_samples,).
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        if self.split == 'train':
            return x_train, y_train
        return x_test, y_test


class FMNISTTFRecord(ClassConditionedDataset):

    def _get_img_label(self):
        """Returns the image and label array according to split.

        Returns:
            image array: uint8 numpy array with shape (num_samples, 28, 28).
            labels array: uint8 numpy array with shape (num_samples,).
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        if self.split == 'train':
            return x_train, y_train
        else:
            return x_test, y_test


class EmbeddingConditionedDataset(TFRecordDataset):
    """Interfaces for embedding conditioned dataset.

    This interface is implemted by CUB, Flowers, and MSCOCO.

    Attributes:
        embedding_dir:
        img_dir:
        caption_dir:
        output_dir:
        num_shards:
        num_captions_per_img:
    """

    def __init__(self, embedding_dir, img_dir, caption_dir, output_dir, num_shards):
        """Initializes the object.

        FLAGS:
            embedding_dir: Directory to which the embeddings are stored.
            img_dir: Directory to which the images are stored.
            caption_dir: Directory to which the captions are stored.
            output_dir: Directory to which the generated records will be stored.
            num_shards:
        """
        self.embedding_dir = embedding_dir
        self.img_dir = img_dir
        self.caption_dir = caption_dir
        self.output_dir = output_dir
        self.num_shards = num_shards
        self.num_captions_per_img = 10

    def _get_filenames_embeddings_captions(self):
        """Reads file names and embeddings.

        Returns:
            filenames: List of filenames.
            embeddings: Image caption embeddings.
            captions: Image captions.
        """
        raise NotImplementedError

    def _get_tfexample(self, img, embedding, caption):
        """Gets a TFExample.

        FLAGS:
            img: tf.uint8
            embedding:
            caption:

        Returns:
            A TF example.
        """
        embedding = tf.convert_to_tensor(embedding, dtype=tf.float32)
        feature = {
            'image': _bytes_feature(tf.io.serialize_tensor(img)),
            'embedding': _bytes_feature(tf.io.serialize_tensor(embedding)),
            'caption': _bytes_feature(tf.io.serialize_tensor(caption))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _tf_read_img(self, filename):
        """Reads a image.

        FLAGS:
            filename:

        Returns:

        """
        return read_img(os.path.join(self.img_dir, filename))

    def _read_captions(self, filename, captions):
        """Return captions related to the specific filename.

        FLAGS:
            filename:
            captions:
        """
        return captions[os.path.basename(filename)]

    @ray.remote
    def _build_single_tfrecord(self, chunk, filenames, embeddings, captions, path_to_record):
        """Builds a single TFRecord.

        FLAGS:
            chunk:
            filenames:
            embeddings:
            captions:
            path_to_record:
        """
        print('Start to build tf records for {}.'.format(path_to_record))
        with tf.io.TFRecordWriter(path_to_record) as writer:
            for index in chunk:
                img = self._tf_read_img(filenames[index])
                caption = self._read_captions(filenames[index], captions)
                try:
                    tf_example = self._get_tfexample(img, embeddings[index], caption)
                except IOError:
                    print('Path: {}. Index: {}. File name: {}.'.format(path_to_record, index, filenames[index]))
                else:
                    writer.write(tf_example.SerializeToString())
        print('Finished building tf records for {}.'.format(path_to_record))

    def _get_features(self, indices_chunk, filenames, embeddings, captions):
        """Creates sub-tasks

        FLAGS:
            indices_chunk:
            filenames:
            embeddings:
            captions:

        Returns:
            List of sub-tasks.
        """
        return [
            self._build_single_tfrecord.remote(
                self,
                chunk,
                filenames,
                embeddings,
                captions,
                os.path.join(self.output_dir,
                             'tfrecord_{}_{}.tfrecords'.format(str(i + 1).zfill(4), str(self.num_shards).zfill(4)))
            ) for i, chunk in enumerate(indices_chunk)
        ]

    def build_tfrecord(self):
        """Builds TFRecords."""
        filenames, embeddings, captions = self._get_filenames_embeddings_captions()
        indices_chunk = chunkify([*range(len(filenames))], self.num_shards)
        ray.init()
        features = self._get_features(indices_chunk, filenames, embeddings, captions)
        ray.get(features)


class CUBTFRecord(EmbeddingConditionedDataset):

    def _get_filenames_embeddings_captions(self):
        """Reads file names and embeddings.

        Returns:
            filenames: List of filenames.
            embeddings: Image caption embeddings.
        """
        filenames = read_pickle(os.path.join(self.embedding_dir, 'filenames.pickle'))
        filenames = ['{}.jpg'.format(f) for f in filenames]
        embeddings = read_pickle(os.path.join(self.embedding_dir, 'char-CNN-RNN-embeddings.pickle'))
        captions = read_captions(self.caption_dir)
        return filenames, embeddings, captions


class FlowersTFRecord(EmbeddingConditionedDataset):

    def _get_filenames_embeddings_captions(self):
        """Reads file names and embeddings.

        Returns:
            filenames: List of filenames.
            embeddings: Image caption embeddings.
        """
        filenames = read_pickle(os.path.join(self.embedding_dir, 'filenames.pickle'))
        filenames = ['{}.jpg'.format(f.decode('utf8').split('/')[-1]) for f in filenames]
        embeddings = read_pickle(os.path.join(self.embedding_dir, 'char-CNN-RNN-embeddings.pickle'))
        captions = read_captions(self.caption_dir)
        return filenames, embeddings, captions


class MSCOCOTFRecord(EmbeddingConditionedDataset):

    def _get_filenames_embeddings_captions(self):
        """Reads file names and embeddings.

        Returns:
            filenames: List of filenames.
            embeddings: Image caption embeddings.
        """
        filenames = read_pickle(os.path.join(self.embedding_dir, 'filenames.pickle'))
        embeddings = read_pickle(os.path.join(self.embedding_dir, 'char-CNN-RNN-embeddings.pickle'))
        captions = read_pickle(os.path.join(self.embedding_dir, 'captions.pickle'))
        return filenames, embeddings, captions


def get_dtst():
    if FLAGS.dataset == 'mnist':
        dtst = MNISTTFRecord(output_dir=FLAGS.output_dir, split=FLAGS.split, num_shards=FLAGS.num_shards)
    elif FLAGS.dataset == 'fmnist':
        dtst = FMNISTTFRecord(output_dir=FLAGS.output_dir, split=FLAGS.split, num_shards=FLAGS.num_shards)
    elif FLAGS.dataset == 'cub':
        dtst = CUBTFRecord(embedding_dir=FLAGS.embedding_dir, img_dir=FLAGS.img_dir, caption_dir=FLAGS.caption_dir,
                           output_dir=FLAGS.output_dir, num_shards=FLAGS.num_shards)
    elif FLAGS.dataset == 'flowers':
        dtst = FlowersTFRecord(embedding_dir=FLAGS.embedding_dir, img_dir=FLAGS.img_dir, caption_dir=FLAGS.caption_dir,
                               output_dir=FLAGS.output_dir, num_shards=FLAGS.num_shards)
    elif FLAGS.dataset == 'mscoco':
        dtst = MSCOCOTFRecord(embedding_dir=FLAGS.embedding_dir, img_dir=FLAGS.img_dir, caption_dir=FLAGS.caption_dir,
                              output_dir=FLAGS.output_dir, num_shards=FLAGS.num_shards)
    else:
        raise ValueError('Dataset {} is not supported.'.format(FLAGS.dataset))
    return dtst


def main(argv):
    del argv

    dtst = get_dtst()
    dtst.build_tfrecord()


if __name__ == '__main__':
    app.run(main)
