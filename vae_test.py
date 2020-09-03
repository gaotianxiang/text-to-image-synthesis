import unittest

import tensorflow as tf

from vae import ClassConditionedDecoder
from vae import ClassConditionedEncoder
from vae import EmbeddingConditionedDecoder
from vae import EmbeddingConditionedEncoder


class VAETest(unittest.TestCase):

    def setUp(self):
        self.noise_size = 100
        self.batch_size = 32
        self.compression_size = 128

    def test_class_conditioned_encoder_non_conditional(self):
        model = ClassConditionedEncoder(use_condition=False, noise_size=self.noise_size)
        img = tf.random.normal([self.batch_size, 28, 28, 1])
        embedding = tf.random.normal([self.batch_size, 10])

        mu, logsigma = model(img, embedding)
        self.assertEqual(mu.shape, [self.batch_size, self.noise_size])
        self.assertEqual(logsigma.shape, [self.batch_size, self.noise_size])

    def test_class_conditioned_encoder_conditional(self):
        model = ClassConditionedEncoder(use_condition=True, noise_size=self.noise_size)
        img = tf.random.normal([self.batch_size, 28, 28, 1])
        embedding = tf.random.normal([self.batch_size, 10])

        mu, logsigma = model(img, embedding)
        self.assertEqual(mu.shape, [self.batch_size, self.noise_size])
        self.assertEqual(logsigma.shape, [self.batch_size, self.noise_size])

    def test_class_conditioned_decoder_non_conditional(self):
        model = ClassConditionedDecoder(use_condition=False)
        noise = tf.random.normal([self.batch_size, self.noise_size])
        embedding = tf.random.normal([self.batch_size, 10])

        res = model(noise, embedding)
        self.assertEqual(res.shape, [self.batch_size, 28, 28, 1])

    def test_class_conditioned_decoder_conditional(self):
        model = ClassConditionedDecoder(use_condition=True)
        noise = tf.random.normal([self.batch_size, self.noise_size])
        embedding = tf.random.normal([self.batch_size, 10])

        res = model(noise, embedding)
        self.assertEqual(res.shape, [self.batch_size, 28, 28, 1])

    def test_embedding_conditioned_encoder_non_conditional(self):
        model = EmbeddingConditionedEncoder(use_condition=False, noise_size=self.noise_size,
                                            compression_size=self.compression_size)
        img = tf.random.normal([self.batch_size, 64, 64, 3])
        embedding = tf.random.normal([self.batch_size, 1024])

        mu, logsigma = model(img, embedding)
        self.assertEqual(mu.shape, [self.batch_size, self.noise_size])
        self.assertEqual(logsigma.shape, [self.batch_size, self.noise_size])

    def test_embedding_conditioned_encoder_conditional(self):
        model = EmbeddingConditionedEncoder(use_condition=True, noise_size=self.noise_size,
                                            compression_size=self.compression_size)
        img = tf.random.normal([self.batch_size, 64, 64, 3])
        embedding = tf.random.normal([self.batch_size, 1024])

        mu, logsigma = model(img, embedding)
        self.assertEqual(mu.shape, [self.batch_size, self.noise_size])
        self.assertEqual(logsigma.shape, [self.batch_size, self.noise_size])

    def test_embedding_conditioned_decoder_non_conditional(self):
        model = EmbeddingConditionedDecoder(use_condition=False, compression_size=self.compression_size)
        noise = tf.random.normal([self.batch_size, self.noise_size])
        embedding = tf.random.normal([self.batch_size, 1024])

        res = model(noise, embedding)
        self.assertEqual(res.shape, [self.batch_size, 64, 64, 3])

    def test_embedding_conditioned_decoder_conditional(self):
        model = EmbeddingConditionedDecoder(use_condition=True, compression_size=self.compression_size)
        noise = tf.random.normal([self.batch_size, self.noise_size])
        embedding = tf.random.normal([self.batch_size, 1024])

        res = model(noise, embedding)
        self.assertEqual(res.shape, [self.batch_size, 64, 64, 3])


if __name__ == '__main__':
    unittest.main()
