import unittest

import tensorflow as tf

from data_process_func import get_process_func
from data_loader import get_data_loader

MNIST_TRAIN = '/home/tianxiang/datasets/tfrecords/mnist/train/'
MNIST_VAL = '/home/tianxiang/datasets/tfrecords/mnist/val/'

FMNIST_TRAIN = '/home/tianxiang/datasets/tfrecords/fmnist/train/'
FMNIST_VAL = '/home/tianxiang/datasets/tfrecords/fmnist/val/'

CUB_TRAIN = '/home/tianxiang/datasets/tfrecords/cub/train/'
CUB_VAL = '/home/tianxiang/datasets/tfrecords/cub/val/'

FLOWERS_TRAIN = '/home/tianxiang/datasets/tfrecords/flowers/train/'
FLOWERS_VAL = '/home/tianxiang/datasets/tfrecords/flowers/val/'

MSCOCO_TRAIN = '/home/tianxiang/datasets/tfrecords/mscoco/train/'
MSCOCO_VAL = '/home/tianxiang/datasets/tfrecords/mscoco/val'


class DataProcessFuncTest(unittest.TestCase):

    def setUp(self):
        self.batch_size = 32
        self.drop_remainder = True

    def test_mnist_train(self):
        data_loader = get_data_loader('mnist', MNIST_TRAIN, 10, process_func=get_process_func('mnist'),
                                      batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        for img, label, _ in data_loader:
            self.assertEqual(img.shape, [self.batch_size, 28, 28, 1])
            self.assertEqual(img.dtype, tf.float32)
            self.assertEqual(label.shape, [self.batch_size, 10])
            self.assertEqual(label.dtype, tf.float32)

    def test_mnist_val(self):
        data_loader = get_data_loader('mnist', MNIST_VAL, 10, process_func=get_process_func('mnist'),
                                      batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        for img, label, _ in data_loader:
            self.assertEqual(img.shape, [self.batch_size, 28, 28, 1])
            self.assertEqual(img.dtype, tf.float32)
            self.assertEqual(label.shape, [self.batch_size, 10])
            self.assertEqual(label.dtype, tf.float32)


if __name__ == '__main__':
    unittest.main()
