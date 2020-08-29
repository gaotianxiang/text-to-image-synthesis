import unittest

import tensorflow as tf

from gan import ClassConditionedDiscriminator
from gan import ClassConditionedGenerator
from gan import EmbeddingConditionedDiscriminator
from gan import EmbeddingConditionedGenerator


class GANTest(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        self.noise_size = 100
        self.compression_size = 128
        self.embedding_size = 1024
        self.mnist_size = 28
        self.img_size = 64

    def test_class_conditioned_generator_non_conditional(self):
        model = ClassConditionedGenerator(use_condition=False)
        noise = tf.random.normal([self.batch_size, self.noise_size])
        embedding = tf.random.normal([self.batch_size, self.embedding_size])

        res = model(noise, embedding)
        self.assertEqual(res.shape, [self.batch_size, 28, 28, 1])

    def test_class_conditioned_generator_conditional(self):
        model = ClassConditionedGenerator(use_condition=True)
        noise = tf.random.normal([self.batch_size, self.noise_size])
        embedding = tf.random.normal([self.batch_size, self.embedding_size])

        res = model(noise, embedding)
        self.assertEqual(res.shape, [self.batch_size, 28, 28, 1])

    def test_embedding_conditioned_generator_non_conditional(self):
        model = EmbeddingConditionedGenerator(use_condition=False, compression_size=self.compression_size)
        noise = tf.random.normal([self.batch_size, self.noise_size])
        embedding = tf.random.normal([self.batch_size, self.embedding_size])

        res = model(noise, embedding)
        self.assertEqual(res.shape, [self.batch_size, 64, 64, 3])

    def test_embedding_conditioned_generator_conditional(self):
        model = EmbeddingConditionedGenerator(use_condition=True, compression_size=self.compression_size)
        noise = tf.random.normal([self.batch_size, self.noise_size])
        embedding = tf.random.normal([self.batch_size, self.embedding_size])

        res = model(noise, embedding)
        self.assertEqual(res.shape, [self.batch_size, 64, 64, 3])

    def test_class_conditioned_discriminator_non_conditional(self):
        model = ClassConditionedDiscriminator(use_condition=False)
        image = tf.random.normal([self.batch_size, self.mnist_size, self.mnist_size, 1])
        embedding = tf.random.normal([self.batch_size, self.embedding_size])

        res = model(image, embedding)
        self.assertEqual(res.shape, [self.batch_size, 1])

    def test_class_conditioned_discriminator_conditional(self):
        model = ClassConditionedDiscriminator(use_condition=True)
        image = tf.random.normal([self.batch_size, self.mnist_size, self.mnist_size, 1])
        embedding = tf.random.normal([self.batch_size, self.embedding_size])

        res = model(image, embedding)
        self.assertEqual(res.shape, [self.batch_size, 1])

    def test_embedding_conditioned_discriminator_non_conditional(self):
        model = EmbeddingConditionedDiscriminator(use_condition=False, compression_size=self.compression_size)
        image = tf.random.normal([self.batch_size, self.img_size, self.img_size, 3])
        embedding = tf.random.normal([self.batch_size, self.embedding_size])

        res = model(image, embedding)
        self.assertEqual(res.shape, [self.batch_size, 1])

    def test_embedding_conditioned_discriminator_conditional(self):
        model = EmbeddingConditionedDiscriminator(use_condition=True, compression_size=self.compression_size)
        image = tf.random.normal([self.batch_size, self.img_size, self.img_size, 3])
        embedding = tf.random.normal([self.batch_size, self.embedding_size])

        res = model(image, embedding)
        self.assertEqual(res.shape, [self.batch_size, 1])


if __name__ == '__main__':
    unittest.main()
