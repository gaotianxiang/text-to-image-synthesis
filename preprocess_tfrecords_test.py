import os
import unittest

import tensorflow as tf

from preprocess_tfrecords import read_captions

MNIST_TRAIN = '/home/tianxiang/datasets/tfrecords/mnist/train/'
MNIST_VAL = '/home/tianxiang/datasets/tfrecords/mnist/val/'

FMNIST_TRAIN = '/home/tianxiang/datasets/tfrecords/fmnist/train/'
FMNIST_VAL = '/home/tianxiang/datasets/tfrecords/fmnist/val/'

CUB_CAPTIONS = '/home/tianxiang/datasets/cub/birds/text_c10/'
CUB_TRAIN_FILENAMES = '/home/tianxiang/datasets/cub/birds/train/filenames.pickle'
CUB_VAL_FILENAMES = '/home/tianxiang/datasets/cub/birds/test/filenames.pickle'

mnist_feature = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string)
}


def _parse_mnist(example):
    feature = tf.io.parse_example(example, features=mnist_feature)
    image = tf.io.parse_tensor(feature['image'], tf.uint8)
    label = tf.io.parse_tensor(feature['label'], tf.uint8)
    return image, label


class PreprocessTFRecordsTest(unittest.TestCase):
    def test_mnist_train(self):
        mnist = tf.data.TFRecordDataset([os.path.join(MNIST_TRAIN, record) for record in os.listdir(MNIST_TRAIN)],
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE)
        mnist = mnist.map(_parse_mnist, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        mnist = mnist.batch(64, drop_remainder=False)

        total_num = 0
        for img, l in mnist:
            self.assertEqual(len(img), len(l))
            total_num += len(img)
        self.assertEqual(total_num, 60000)

    def test_mnist_val(self):
        mnist = tf.data.TFRecordDataset([os.path.join(MNIST_VAL, record) for record in os.listdir(MNIST_VAL)],
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE)
        mnist = mnist.map(_parse_mnist, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        mnist = mnist.batch(64, drop_remainder=False)

        total_num = 0
        for img, l in mnist:
            self.assertEqual(len(img), len(l))
            total_num += len(img)
        self.assertEqual(total_num, 10000)

    def test_fmnist_train(self):
        fmnist = tf.data.TFRecordDataset([os.path.join(FMNIST_TRAIN, record) for record in os.listdir(FMNIST_TRAIN)],
                                         num_parallel_reads=tf.data.experimental.AUTOTUNE)
        fmnist = fmnist.map(_parse_mnist, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        fmnist = fmnist.batch(64, drop_remainder=False)

        total_num = 0
        for img, l in fmnist:
            self.assertEqual(len(img), len(l))
            total_num += len(img)
        self.assertEqual(total_num, 60000)

    def test_fmnist_val(self):
        fmnist = tf.data.TFRecordDataset([os.path.join(FMNIST_VAL, record) for record in os.listdir(FMNIST_VAL)],
                                         num_parallel_reads=tf.data.experimental.AUTOTUNE)
        fmnist = fmnist.map(_parse_mnist, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        fmnist = fmnist.batch(64, drop_remainder=False)

        total_num = 0
        for img, l in fmnist:
            self.assertEqual(len(img), len(l))
            total_num += len(img)
        self.assertEqual(total_num, 10000)

    def test_read_captions_cub(self):



if __name__ == '__main__':
    unittest.main()
