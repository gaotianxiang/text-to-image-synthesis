import os
import unittest

import tensorflow as tf

MNIST_TRAIN = '/home/tianxiang/datasets/tfrecords/mnist/train/'
MNIST_VAL = '/home/tianxiang/datasets/tfrecords/mnist/val/'

FMNIST_TRAIN = '/home/tianxiang/datasets/tfrecords/fmnist/train/'
FMNIST_VAL = '/home/tianxiang/datasets/tfrecords/fmnist/val/'

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

    def test_mnist_val(self):
        mnist = tf.data.TFRecordDataset([os.path.join(MNIST_VAL, record) for record in os.listdir(MNIST_VAL)],
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE)
        mnist = mnist.map(_parse_mnist, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        mnist = mnist.batch(64, drop_remainder=False)

        total_num = 0
        for img, l in mnist:
            self.assertEqual(len(img), len(l))
            total_num += len(img)

    def test_fmnist_train(self):
        fmnist = tf.data.TFRecordDataset([os.path.join(FMNIST_TRAIN, record) for record in os.listdir(FMNIST_TRAIN)],
                                         num_parallel_reads=tf.data.experimental.AUTOTUNE)
        fmnist = fmnist.map(_parse_mnist, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        fmnist = fmnist.batch(64, drop_remainder=False)

        total_num = 0
        for img, l in fmnist:
            self.assertEqual(len(img), len(l))
            total_num += len(img)

    def test_fmnist_val(self):
        fmnist = tf.data.TFRecordDataset([os.path.join(FMNIST_VAL, record) for record in os.listdir(FMNIST_VAL)],
                                         num_parallel_reads=tf.data.experimental.AUTOTUNE)
        fmnist = fmnist.map(_parse_mnist, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        fmnist = fmnist.batch(64, drop_remainder=False)

        total_num = 0
        for img, l in fmnist:
            self.assertEqual(len(img), len(l))
            total_num += len(img)

    def test_dataset_unbatch(self):
        dtst = tf.data.Dataset.range(5)
        dtst = dtst.map(lambda x: ([x] * 3, [x ** 2] * 3))
        dtst = dtst.unbatch().shuffle(100)
        for x, y in dtst:
            print('{} {}'.format(x, y))


if __name__ == '__main__':
    unittest.main()
