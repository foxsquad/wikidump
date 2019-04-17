from absl import flags

FLAGS = flags.FLAGS


dataset_path = 'first-1000.sqlite3'


def input_fn(batch_size, shuffle_buffer, shuffle_seed=None):
    """A simple dataset builder function.

    This function creates a simple dataset for traning the
    auto encoder model below.

    Note that param `article_to_take` only available for `TRAIN` mode.
    """
    import tensorflow as tf

    max_charcode = 0x024F

    def to_sequence(article):
        # article is now a tensor
        article = tf.strings.unicode_decode(
            article, 'UTF-8', 'replace', 32)
        article = tf.clip_by_value(article, 0, max_charcode)
        return tf.data.Dataset.from_tensor_slices(article)

    def wrapper(mode, input_context=None):
        base = tf.data.experimental.SqlDataset(
            'sqlite', dataset_path,
            'select content from wiki_text',
            tf.string)
        train_dataset = base.skip(5).take(FLAGS.article_to_take)
        val_dataset = base.take(1)
        test_dataset = base.skip(1).take(1)

        if mode == tf.estimator.ModeKeys.TRAIN:
            d = train_dataset
        elif mode == tf.estimator.ModeKeys.EVAL:
            d = val_dataset
        else:  # Assum test here
            d = test_dataset

        if input_context:
            d = d.apply(tf.data.experimental.filter_for_shard(
                input_context.num_input_pipelines,
                input_context.input_pipeline_id))
        return d.flat_map(to_sequence) \
                .map(lambda x: tf.expand_dims(x, axis=-1)) \
                .map(lambda x: (x, x)) \
                .shuffle(shuffle_buffer, seed=shuffle_seed) \
                .batch(batch_size)

    return wrapper


def model_fn():
    from tensorflow.keras import layers
    from tensorflow.keras import Sequential

    model = Sequential([
        layers.Dense(20, 'tanh', bias_initializer='glorot_uniform',
                     input_shape=[1]),
        layers.Dense(100, 'tanh', bias_initializer='glorot_uniform'),
        layers.Dense(200, 'tanh', name='main_embedder',
                     bias_initializer='glorot_uniform'),
        layers.Dense(0x250, use_bias=False, name='cat_prob')
    ], name='simple_auto_encoder')

    return model
