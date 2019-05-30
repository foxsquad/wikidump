"""Auto encoder used for wikidump process

This is used to overcome slice op when using distributed training
loop. Current collective op does not support index slice.
"""

import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS


flags.DEFINE_integer('train_articles', 10,
                     'Number of article to take to train dataset.',
                     lower_bound=1)

dataset_path = 'first-1000.sqlite3'


def input_fn(mode, input_context=None):
    """A simple dataset builder function.

    This function creates a simple dataset for traning the
    auto encoder model below.

    Note that param `article_to_take` only available for `TRAIN` mode.
    """

    max_charcode = 0x024F

    def to_sequence(article):
        # article is now a tensor
        article = tf.strings.unicode_decode(
            article, 'UTF-8', 'replace', 32)
        article = tf.clip_by_value(article, 0, max_charcode)
        return tf.data.Dataset.from_tensor_slices(article)

    def to_tuple(x):
        x = tf.expand_dims(x, axis=-1)
        return x, x

    base = tf.data.experimental.SqlDataset(
        'sqlite', dataset_path,
        'select content from wiki_text',
        tf.string)
    train_dataset = base.skip(5).take(FLAGS.train_articles)
    val_dataset = base.take(1)
    test_dataset = base.skip(1).take(1)

    if mode == tf.estimator.ModeKeys.TRAIN:
        d = train_dataset
    elif mode == tf.estimator.ModeKeys.EVAL:
        d = val_dataset
    else:
        d = test_dataset

    if input_context:
        d = d.apply(tf.data.experimental.filter_for_shard(
            input_context.num_input_pipelines,
            input_context.input_pipeline_id))
    d = d.flat_map(to_sequence) \
        .apply(tf.data.experimental.map_and_batch(
            to_tuple, FLAGS.batch_size, drop_remainder=True))

    if FLAGS.repeat:
        d = d.apply(tf.data.experimental.shuffle_and_repeat(
            FLAGS.shuffle_buffer, None, FLAGS.shuffle_seed))
    else:
        d = d.shuffle(FLAGS.shuffle_buffer, FLAGS.shuffle_seed)

    return d


def model_fn():
    from tensorflow.python.keras import layers, Sequential

    model = Sequential([
        layers.Dense(20, 'tanh', bias_initializer='glorot_uniform',
                     input_shape=[1]),
        layers.Dense(100, 'tanh', bias_initializer='glorot_uniform'),
        layers.Dense(200, 'tanh', name='main_embedder',
                     bias_initializer='glorot_uniform'),
        layers.Dense(0x250, use_bias=False, name='cat_prob')
    ], name='simple_auto_encoder')

    return model


_loss_fn = tf.losses.SparseCategoricalCrossentropy(True)


def loss_fn(a, b):
    return _loss_fn(a, b)
