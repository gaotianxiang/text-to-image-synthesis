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

FLOWER_TRAIN = '/home/tianxiang/datasets/tfrecords/flowers/train/'
FLOWER_VAL = '/home/tianxiang/datasets/tfrecords/flowers/val/'

MSCOCO_TRAIN = '/home/tianxiang/datasets/tfrecords/mscoco/train/'
MSCOCO_VAL = '/home/tianxiang/datasets/tfrecords/mscoco/val'


class DataProcessFuncTest(unittest.TestCase):

    def setUp(self):
        self.batch_size = 32
        self.drop_remainder = True

    def test_mnist_train(self):
        data_loader = get_data_loader('mnist', MNIST_TRAIN, 10, num_channels=1, process_func=get_process_func('mnist'),
                                      shuffle=True, batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        for img, label, _ in data_loader:
            self.assertEqual(img.shape, [self.batch_size, 28, 28, 1])
            self.assertEqual(img.dtype, tf.float32)
            self.assertEqual(label.shape, [self.batch_size, 10])
            self.assertEqual(label.dtype, tf.float32)

    def test_mnist_val(self):
        data_loader = get_data_loader('mnist', MNIST_VAL, 10, num_channels=1, process_func=get_process_func('mnist'),
                                      shuffle=False, batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        for img, label, _ in data_loader:
            self.assertEqual(img.shape, [self.batch_size, 28, 28, 1])
            self.assertEqual(img.dtype, tf.float32)
            self.assertEqual(label.shape, [self.batch_size, 10])
            self.assertEqual(label.dtype, tf.float32)

    def test_fmnist_train(self):
        data_loader = get_data_loader('fmnist', FMNIST_TRAIN, 10, num_channels=1,
                                      process_func=get_process_func('fmnist'), shuffle=True, batch_size=self.batch_size,
                                      drop_remainder=self.drop_remainder)

        for img, label, _ in data_loader:
            self.assertEqual(img.shape, [self.batch_size, 28, 28, 1])
            self.assertEqual(img.dtype, tf.float32)
            self.assertEqual(label.shape, [self.batch_size, 10])
            self.assertEqual(label.dtype, tf.float32)

    def test_fmnist_val(self):
        data_loader = get_data_loader('mnist', FMNIST_VAL, 10, num_channels=1, process_func=get_process_func('fmnist'),
                                      shuffle=False, batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        for img, label, _ in data_loader:
            self.assertEqual(img.shape, [self.batch_size, 28, 28, 1])
            self.assertEqual(img.dtype, tf.float32)
            self.assertEqual(label.shape, [self.batch_size, 10])
            self.assertEqual(label.dtype, tf.float32)

    def test_cub_train(self):
        data_loader = get_data_loader('cub', CUB_TRAIN, 10, num_channels=3, process_func=get_process_func('cub'),
                                      shuffle=True, batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        for img, embedding, caption in data_loader:
            self.assertEqual(img.shape, [self.batch_size, 64, 64, 3])
            self.assertEqual(img.dtype, tf.float32)
            self.assertEqual(embedding.shape, [self.batch_size, 1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [self.batch_size])
            self.assertEqual(caption.dtype, tf.string)

    def test_cub_val(self):
        data_loader = get_data_loader('cub', CUB_VAL, 10, num_channels=3, process_func=get_process_func('cub'),
                                      shuffle=False, batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        for img, embedding, caption in data_loader:
            self.assertEqual(img.shape, [self.batch_size, 64, 64, 3])
            self.assertEqual(img.dtype, tf.float32)
            self.assertEqual(embedding.shape, [self.batch_size, 1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [self.batch_size])
            self.assertEqual(caption.dtype, tf.string)

    def test_flower_train(self):
        data_loader = get_data_loader('flower', FLOWER_TRAIN, 10, num_channels=3,
                                      process_func=get_process_func('flower'), shuffle=True,
                                      batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        for img, embedding, caption in data_loader:
            self.assertEqual(img.shape, [self.batch_size, 64, 64, 3])
            self.assertEqual(img.dtype, tf.float32)
            self.assertEqual(embedding.shape, [self.batch_size, 1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [self.batch_size])
            self.assertEqual(caption.dtype, tf.string)

    def test_flower_val(self):
        data_loader = get_data_loader('flower', FLOWER_VAL, 10, num_channels=3, process_func=get_process_func('flower'),
                                      shuffle=False, batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        for img, embedding, caption in data_loader:
            self.assertEqual(img.shape, [self.batch_size, 64, 64, 3])
            self.assertEqual(img.dtype, tf.float32)
            self.assertEqual(embedding.shape, [self.batch_size, 1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [self.batch_size])
            self.assertEqual(caption.dtype, tf.string)


if __name__ == '__main__':
    unittest.main()
