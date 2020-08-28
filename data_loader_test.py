import unittest

import tensorflow as tf

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


class DataLoaderTest(unittest.TestCase):
    def test_mnist_train(self):
        dtst = get_data_loader('mnist', MNIST_TRAIN, None, None, 32, True)

        for img, l in dtst:
            self.assertEqual(len(img), len(l))
            self.assertEqual(img.shape, [32, 28, 28])
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(l.shape, [32])
            self.assertEqual(l.dtype, tf.uint8)

    def test_mnist_val(self):
        dtst = get_data_loader('mnist', MNIST_VAL, None, None, 32, True)

        for img, l in dtst:
            self.assertEqual(len(img), len(l))
            self.assertEqual(img.shape, [32, 28, 28])
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(l.shape, [32])
            self.assertEqual(l.dtype, tf.uint8)

    def test_fmnist_train(self):
        dtst = get_data_loader('fmnist', FMNIST_TRAIN, None, None, 32, True)

        for img, l in dtst:
            self.assertEqual(len(img), len(l))
            self.assertEqual(img.shape, [32, 28, 28])
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(l.shape, [32])
            self.assertEqual(l.dtype, tf.uint8)

    def test_fmnist_val(self):
        dtst = get_data_loader('fmnist', FMNIST_VAL, None, None, 32, True)

        for img, l in dtst:
            self.assertEqual(len(img), len(l))
            self.assertEqual(img.shape, [32, 28, 28])
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(l.shape, [32])
            self.assertEqual(l.dtype, tf.uint8)

    def test_cub_train(self):
        dtst = get_data_loader('cub', CUB_TRAIN, 10, None, None, True)

        for img, embedding, caption in dtst:
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(embedding.shape, [1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [])
            self.assertEqual(caption.dtype, tf.string)

    def test_cub_val(self):
        dtst = get_data_loader('cub', CUB_VAL, 10, None, None, True)

        for img, embedding, caption in dtst:
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(embedding.shape, [1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [])
            self.assertEqual(caption.dtype, tf.string)

    def test_flowers_train(self):
        dtst = get_data_loader('flowers', FLOWERS_TRAIN, 10, None, None, True)

        for img, embedding, caption in dtst:
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(embedding.shape, [1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [])
            self.assertEqual(caption.dtype, tf.string)

    def test_flowers_val(self):
        dtst = get_data_loader('flowers', FLOWERS_VAL, 10, None, None, True)

        for img, embedding, caption in dtst:
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(embedding.shape, [1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [])
            self.assertEqual(caption.dtype, tf.string)

    def test_mscoco_train(self):
        dtst = get_data_loader('mscoco', MSCOCO_TRAIN, 5, None, None, True)

        for img, embedding, caption in dtst:
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(embedding.shape, [1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [])
            self.assertEqual(caption.dtype, tf.string)

    def test_mscoco_val(self):
        dtst = get_data_loader('mscoco', MSCOCO_VAL, 5, None, None, True)

        for img, embedding, caption in dtst:
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(embedding.shape, [1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [])
            self.assertEqual(caption.dtype, tf.string)


if __name__ == '__main__':
    unittest.main()
