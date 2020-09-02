"""Gets datalaader."""

import os

import tensorflow as tf

class_conditioned_feature = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string)
}

embedding_conditioned_feature = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'embedding': tf.io.FixedLenFeature([], tf.string),
    'caption': tf.io.FixedLenFeature([], tf.string)
}


def _get_tf_record_dataset(tfrecord_dir):
    """Gets a TFRecord dataset.

    Args:
        tfrecord_dir: Directory to which the TFRecords are stored.

    Returns:
        A TFRecord dataset.
    """
    return tf.data.TFRecordDataset([os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir)],
                                   num_parallel_reads=tf.data.experimental.AUTOTUNE)


def _parse_class_conditioned(example):
    """Parses MNIST and FMNIST TFRecords.

    Returns:
        image:
        label:
    """
    features = tf.io.parse_example(example, features=class_conditioned_feature)
    image = tf.io.parse_tensor(features['image'], tf.uint8)
    label = tf.io.parse_tensor(features['label'], tf.uint8)
    return image, label


def _parse_embedding_conditioned(example):
    """Parses CUB, Flowers, or MSCOCO TFRecords.

    Args:
        example:

    Returns:
        image:
        embedding:
        caption:
    """
    features = tf.io.parse_example(example, features=embedding_conditioned_feature)
    image = tf.io.parse_tensor(features['image'], tf.uint8)
    embedding = tf.io.parse_tensor(features['embedding'], tf.float32)
    caption = tf.io.parse_tensor(features['caption'], tf.string)
    return image, embedding, caption


def _parse(dtst_name):
    """Returns a parse function according to the dataset name.

    Args:
        dtst_name:

    Returns:
        A parse function.
    """
    if dtst_name in ['mnist', 'fmnist']:
        return _parse_class_conditioned
    return _parse_embedding_conditioned


def _tile_image(dtst_name, num_captions_per_image):
    """Return a function that tiles the image according to the number of the captions per image.

    Args:
        num_captions_per_image:

    Returns:
        A function that tiles images.
    """
    if dtst_name in ['mnist', 'fmnist']:
        return lambda image, label: (image, label)

    def tile_image(image, embedding, caption):
        image = tf.expand_dims(image, axis=0)
        image = tf.tile(image, multiples=[num_captions_per_image, 1, 1, 1])
        return image, embedding[:num_captions_per_image], caption[:num_captions_per_image]

    return tile_image


def _unbatch(dtst_name, dtst):
    """Unbatches the dataset.

    Args:
        dtst_name:
        dtst:

    Returns:
        A TF dataset.
    """
    if dtst_name in ['mnist', 'fmnist']:
        return dtst
    return dtst.unbatch()


def _process_dataset(dtst, dtst_name, num_captions_per_image, num_channels, process_func, shuffle):
    """Processes the TFRecord dataset.

    Args:
        dtst:
        dtst_name:
        num_captions_per_image:
        num_channels:
        process_func:
        shuffle:

    Returns:
        A processed TF dataset.
    """
    dtst = dtst.map(_parse(dtst_name), num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    dtst = dtst.map(_tile_image(dtst_name, num_captions_per_image), num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    deterministic=False)
    dtst = _unbatch(dtst_name, dtst)
    dtst = dtst.filter(lambda img, embedding, caption: img.shape[-1] == num_channels)
    if process_func is not None:
        dtst = dtst.map(process_func, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
    if shuffle:
        dtst = dtst.shuffle(buffer_size=50000)
    return dtst


def _batchify(dtst, batch_size, drop_remainder):
    """Makes the dataset return batches of data.

    Args:
        dtst:
        batch_size:
        drop_remainder:

    Returns:
        A TF dataset.
    """
    if batch_size is None:
        return dtst
    return dtst.batch(batch_size, drop_remainder).prefetch(tf.data.experimental.AUTOTUNE)


def get_data_loader(dataset_name, tfrecord_dir, num_caption_per_image, num_channels, process_func, shuffle, batch_size,
                    drop_remainder):
    """Gets data loader.

    Args:
        dataset_name:
        tfrecord_dir:
        num_caption_per_image:
        num_channels:
        process_func:
        shuffle:
        batch_size:
        drop_remainder:

    Returns:
        A TF dataset.
    """
    dtst = _get_tf_record_dataset(tfrecord_dir)
    dtst = _process_dataset(dtst, dataset_name, num_caption_per_image, num_channels, process_func, shuffle)
    dtst = _batchify(dtst, batch_size, drop_remainder)
    return dtst
