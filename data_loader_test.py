import unittest

import tensorflow as tf

from data_loader import get_data_loader

MNIST_TRAIN = '/home/tianxiang/datasets/tfrecords/mnist/train/'
MNIST_VAL = '/home/tianxiang/datasets/tfrecords/mnist/val/'

FMNIST_TRAIN = '/home/tianxiang/datasets/tfrecords/fmnist/train/'
FMNIST_VAL = '/home/tianxiang/datasets/tfrecords/fmnist/val/'

CUB_TRAIN = '/home/tianxiang/datasets/tfrecords/cub/train/'
CUB_VAL = '/home/tianxiang/datasets/tfrecords/cub/val/'

FLOWER_TRAIN = '/home/tianxiang/datasets/tfrecords/flower/train/'
FLOWER_VAL = '/home/tianxiang/datasets/tfrecords/flower/val/'

MSCOCO_TRAIN = '/home/tianxiang/datasets/tfrecords/mscoco/train/'
MSCOCO_VAL = '/home/tianxiang/datasets/tfrecords/mscoco/val'


class DataLoaderTest(unittest.TestCase):
    def test_mnist_train(self):
        dtst = get_data_loader('mnist', MNIST_TRAIN, num_caption_per_image=None, process_func=None,
                               shuffle=True, batch_size=32, drop_remainder=True)

        for img, l in dtst:
            self.assertEqual(len(img), len(l))
            self.assertEqual(img.shape, [32, 28, 28])
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(l.shape, [32])
            self.assertEqual(l.dtype, tf.uint8)

    def test_mnist_val(self):
        dtst = get_data_loader('mnist', MNIST_VAL, num_caption_per_image=None, process_func=None,
                               shuffle=True, batch_size=32, drop_remainder=True)

        for img, l in dtst:
            self.assertEqual(len(img), len(l))
            self.assertEqual(img.shape, [32, 28, 28])
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(l.shape, [32])
            self.assertEqual(l.dtype, tf.uint8)

    def test_fmnist_train(self):
        dtst = get_data_loader('fmnist', FMNIST_TRAIN, num_caption_per_image=None, process_func=None,
                               shuffle=True, batch_size=32, drop_remainder=True)

        for img, l in dtst:
            self.assertEqual(len(img), len(l))
            self.assertEqual(img.shape, [32, 28, 28])
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(l.shape, [32])
            self.assertEqual(l.dtype, tf.uint8)

    def test_fmnist_val(self):
        dtst = get_data_loader('fmnist', FMNIST_VAL, num_caption_per_image=None, process_func=None,
                               shuffle=True, batch_size=32, drop_remainder=True)

        for img, l in dtst:
            self.assertEqual(len(img), len(l))
            self.assertEqual(img.shape, [32, 28, 28])
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(l.shape, [32])
            self.assertEqual(l.dtype, tf.uint8)

    def test_cub_train(self):
        dtst = get_data_loader('cub', CUB_TRAIN, num_caption_per_image=10, process_func=None,
                               shuffle=True, batch_size=None, drop_remainder=True)

        for img, embedding, caption in dtst:
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(embedding.shape, [1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [])
            self.assertEqual(caption.dtype, tf.string)

    def test_cub_val(self):
        dtst = get_data_loader('cub', CUB_VAL, num_caption_per_image=10, process_func=None,
                               shuffle=True, batch_size=None, drop_remainder=True)

        for img, embedding, caption in dtst:
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(embedding.shape, [1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [])
            self.assertEqual(caption.dtype, tf.string)

    def test_flower_train(self):
        dtst = get_data_loader('flower', FLOWER_TRAIN, num_caption_per_image=10, process_func=None,
                               shuffle=True, batch_size=None, drop_remainder=True)

        for img, embedding, caption in dtst:
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(embedding.shape, [1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [])
            self.assertEqual(caption.dtype, tf.string)

    def test_flower_val(self):
        dtst = get_data_loader('flower', FLOWER_VAL, num_caption_per_image=10, process_func=None,
                               shuffle=True, batch_size=None, drop_remainder=True)

        for img, embedding, caption in dtst:
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(embedding.shape, [1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [])
            self.assertEqual(caption.dtype, tf.string)

    def test_mscoco_train(self):
        dtst = get_data_loader('coco', MSCOCO_TRAIN, num_caption_per_image=5, process_func=None,
                               shuffle=True, batch_size=None, drop_remainder=True)

        for img, embedding, caption in dtst:
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(embedding.shape, [1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [])
            self.assertEqual(caption.dtype, tf.string)

    def test_mscoco_val(self):
        dtst = get_data_loader('coco', MSCOCO_VAL, num_caption_per_image=5, process_func=None,
                               shuffle=True, batch_size=None, drop_remainder=True)

        for img, embedding, caption in dtst:
            self.assertEqual(img.dtype, tf.uint8)
            self.assertEqual(embedding.shape, [1024])
            self.assertEqual(embedding.dtype, tf.float32)
            self.assertEqual(caption.shape, [])
            self.assertEqual(caption.dtype, tf.string)


if __name__ == '__main__':
    unittest.main()
