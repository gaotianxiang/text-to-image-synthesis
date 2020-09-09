import tensorflow as tf

from flow import ClassConditionedConvAffineCouplingLayer
from flow import ClassConditionedFlow
from flow import EmbeddingConditionedConvAffineCouplingLayer
from flow import EmbeddingConditionedFlow


class FlowTest(tf.test.TestCase):

    def setUp(self):
        self.batch_size = 32
        self.compression_size = 128

    def test_class_conditioned_conv_affine_coupling_layer_non_conditional(self):
        mask = tf.range(784, dtype=tf.float32)
        mask = tf.reshape(mask, shape=[28, 28, 1]) % 2
        image = tf.random.normal([self.batch_size, 28, 28, 1])
        embedding = tf.random.normal([self.batch_size, 10])

        model = ClassConditionedConvAffineCouplingLayer(num_channels_hidden=[32, 32], mask=mask, use_condition=False)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

        model = ClassConditionedConvAffineCouplingLayer(num_channels_hidden=[32, 32], mask=1 - mask,
                                                        use_condition=False)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

    def test_class_conditioned_conv_affine_coupling_layer_conditional(self):
        mask = tf.range(784, dtype=tf.float32)
        mask = tf.reshape(mask, shape=[28, 28, 1]) % 2
        image = tf.random.normal([self.batch_size, 28, 28, 1])
        embedding = tf.random.normal([self.batch_size, 10])

        model = ClassConditionedConvAffineCouplingLayer(num_channels_hidden=[32, 32], mask=mask, use_condition=True)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

        model = ClassConditionedConvAffineCouplingLayer(num_channels_hidden=[32, 32], mask=1 - mask, use_condition=True)
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
        image = tf.random.normal([self.batch_size, 28, 28, 1])
        embedding = tf.random.normal([self.batch_size, 10])

        model = ClassConditionedFlow(use_condition=False)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
        self.assertAllClose(res_backward, image)
        self.assertAllClose(logdet_forward, -logdet_backward)

    def test_class_conditioned_flow_conditional(self):
        image = tf.random.normal([self.batch_size, 28, 28, 1])
        embedding = tf.random.normal([self.batch_size, 10])

        model = ClassConditionedFlow(use_condition=True)
        res_forward, logdet_forward = model(image, embedding, reverse=False)
        res_backward, logdet_backward = model(res_forward, embedding, reverse=True)
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


if __name__ == '__main__':
    tf.test.main()
