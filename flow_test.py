import tensorflow as tf

from flow import BatchNorm
from flow import ClassConditionedAffineCouplingLayer
from flow import ClassConditionedFlow
from flow import EmbeddingConditionedConvAffineCouplingLayer
from flow import EmbeddingConditionedFlow


class FlowTest(tf.test.TestCase):

    def setUp(self):
        self.batch_size = 32
        self.compression_size = 128

    def test_class_conditioned_affine_coupling_layer_non_conditional(self):
        mask = tf.range(784, dtype=tf.float32)
        mask = tf.reshape(mask, shape=[1, 784]) % 2
        image = tf.random.normal([self.batch_size, 784])
        embedding = tf.random.normal([self.batch_size, 10])

        model = ClassConditionedAffineCouplingLayer(hidden_size=1024, mask=mask, use_condition=False)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

        model = ClassConditionedAffineCouplingLayer(hidden_size=1024, mask=1 - mask, use_condition=False)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

    def test_class_conditioned_affine_coupling_layer_conditional(self):
        mask = tf.range(784, dtype=tf.float32)
        mask = tf.reshape(mask, shape=[1, 784]) % 2
        image = tf.random.normal([self.batch_size, 784])
        embedding = tf.random.normal([self.batch_size, 10])

        model = ClassConditionedAffineCouplingLayer(hidden_size=1024, mask=mask, use_condition=True)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

        model = ClassConditionedAffineCouplingLayer(hidden_size=1024, mask=1 - mask, use_condition=True)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

    def test_embedding_conditioned_conv_affine_coupling_layer_non_conditional(self):
        mask = tf.range(4096, dtype=tf.float32)
        mask = tf.reshape(mask, shape=[64, 64, 1]) % 2
        image = tf.random.normal([self.batch_size, 64, 64, 3])
        embedding = tf.random.normal([self.batch_size, 1024])

        model = EmbeddingConditionedConvAffineCouplingLayer(num_channels_hidden=[32, 32],
                                                            compression_size=self.compression_size, mask=mask,
                                                            use_condition=False)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

        model = EmbeddingConditionedConvAffineCouplingLayer(num_channels_hidden=[32, 32],
                                                            compression_size=self.compression_size, mask=1 - mask,
                                                            use_condition=False)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

    def test_embedding_conditioned_conv_affine_coupling_layer_conditional(self):
        mask = tf.range(4096, dtype=tf.float32)
        mask = tf.reshape(mask, shape=[64, 64, 1]) % 2
        image = tf.random.normal([self.batch_size, 64, 64, 3])
        embedding = tf.random.normal([self.batch_size, 1024])

        model = EmbeddingConditionedConvAffineCouplingLayer(num_channels_hidden=[32, 32],
                                                            compression_size=self.compression_size, mask=mask,
                                                            use_condition=True)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

        model = EmbeddingConditionedConvAffineCouplingLayer(num_channels_hidden=[32, 32],
                                                            compression_size=self.compression_size, mask=1 - mask,
                                                            use_condition=True)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

    def test_class_conditioned_flow_non_conditional(self):
        image = tf.random.normal([self.batch_size, 784])
        embedding = tf.random.normal([self.batch_size, 10])

        model = ClassConditionedFlow(use_condition=False)
        res_forward, logdet_forward = model(image, embedding, reverse=False, training=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True, training=False)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

    def test_class_conditioned_flow_conditional(self):
        image = tf.random.normal([self.batch_size, 784])
        embedding = tf.random.normal([self.batch_size, 10])

        model = ClassConditionedFlow(use_condition=True)
        res_forward, logdet_forward = model(image, embedding, reverse=False, training=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True, training=False)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

    def test_embedding_conditioned_flow_non_conditional(self):
        image = tf.random.normal([self.batch_size, 64, 64, 3])
        embedding = tf.random.normal([self.batch_size, 1024])

        model = EmbeddingConditionedFlow(use_condition=False, compression_size=self.compression_size)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

    def test_embedding_conditioned_flow_conditional(self):
        tf.random.set_seed(111)
        image = tf.random.normal([self.batch_size, 64, 64, 3])
        embedding = tf.random.normal([self.batch_size, 1024])

        model = EmbeddingConditionedFlow(use_condition=True, compression_size=self.compression_size)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllCloseAccordingToType(logdet_forward, -logdet_backward)
        self.assertAllClose(res_backward, image)

    def test_batch_normalization(self):
        image = tf.random.normal([self.batch_size, 64, 64, 3])
        embedding = tf.random.normal([self.batch_size, 1024])

        bn = BatchNorm()
        res_forward, logdet_forward = bn(image, embedding, reverse=False, training=False)
        res_backward, logdet_backward = bn(res_forward, embedding, reverse=True, training=False)
        self.assertAllClose(image, res_backward)
        self.assertAllClose(logdet_forward, -logdet_backward)


if __name__ == '__main__':
    tf.test.main()
