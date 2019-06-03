import json
import os

import tensorflow as tf
from absl import flags
from tensorflow.python.eager import context as _context
from tensorflow.python.keras import Model, initializers, losses, regularizers
from tensorflow.python.keras.activations import sigmoid
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.layers import GRU, \
    BatchNormalization, Dense, InputLayer, Layer
from tensorflow.python.ops import math_ops, nn

FLAGS = flags.FLAGS

TF_DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    'data', 'node_cache.tfrecord')


flags.DEFINE_integer('timesteps', 20, 'Train input sequence value')


def fold_to_sequence(old_state, input_element):
    new_state = tf.concat([old_state[1:], [input_element]], axis=0)
    return new_state, new_state


def to_tuple(i):
    # As we don't known exactly which data points is abnormal,
    # we assumed everything is normal, and let the model decide
    # whether it's normal or not.
    return (
        i,  # input
        (
            # decoder output
            i,
            # const value of possitive outcome
            tf.constant(1.0, shape=(FLAGS.timesteps, ))
        )
    )


def _parse_tensor(tensor):
    x = tf.io.parse_tensor(tensor, tf.float32)
    return x


def input_fn(mode, input_context=None):
    base = tf.data.TFRecordDataset(TF_DATA_FILE)
    base = base.map(_parse_tensor)
    init_state = tf.constant(
        0, dtype=tf.float32,
        shape=(FLAGS.timesteps, SEQ_SIZE))
    base = base.apply(tf.data.experimental.scan(
        init_state, fold_to_sequence))

    # The first `train_seq_window` of records is all zeros,
    # due to `init_state`, so we would skip those here.
    base = base.skip(FLAGS.timesteps)
    if mode == tf.estimator.ModeKeys.TRAIN:
        d = base.skip(10000)
    elif mode == tf.estimator.ModeKeys.EVAL:
        d = base.take(5000)
    else:
        d = base.skip(5000).take(5000)

    if input_context:
        d = d.apply(tf.data.experimental.filter_for_shard(
            input_context.num_input_pipelines,
            input_context.input_pipeline_id))
    d = d.apply(tf.data.experimental.map_and_batch(
        to_tuple, FLAGS.batch_size))

    if FLAGS.repeat:
        d = d.apply(tf.data.experimental.shuffle_and_repeat(
            FLAGS.shuffle_buffer, None, FLAGS.shuffle_seed))
    else:
        d = d.shuffle(FLAGS.shuffle_buffer, FLAGS.shuffle_seed)

    return d


class RadiusScoreLayer(Layer):
    """A simpler version of SVM on hyper-space.

    The output of this layer is directly the decision values
    of wheather data point(s) is inside or outside the hyper-
    sphere surface.
    """

    def __init__(self, init_radius=0.5, **kwargs):
        super(RadiusScoreLayer, self).__init__(**kwargs)
        self.init_radius = init_radius

    def build(self, input_shape):
        # Hyper-space dimension size, return from previous layer.
        # We expect the input tensor has 3 dimentions and last one
        # must available.
        assert input_shape.ndims == 3
        assert input_shape[-1] is not None
        self.dim = input_shape[-1]

        self.input_spec = InputSpec(axes={-1: self.dim})

        self.radius = self.add_weight(
            'hyper_sphere_radius',
            shape=(), dtype=tf.float32,
            initializer=initializers.ConstantV2(self.init_radius),
            regularizer=regularizers.l2(0.01),
            trainable=True)

        self.center = self.add_weight(
            'hyper_sphere_center',
            shape=(self.dim, ),
            initializer=initializers.GlorotUniformV2(),
            regularizer=regularizers.l2(0.001),
            trainable=True)

        super(RadiusScoreLayer, self).build(input_shape)

    def call(self, input_tensor):
        x = input_tensor - self.center
        x = tf.norm(x, axis=-1)

        # This is used to simulate soft-sign function,
        # which is a soft, continuous function and
        # differentiable at any point.
        diff = sigmoid(self.radius ** 2 - x ** 2) * 2 - 1
        return diff


class RNNBlock(Layer):
    """Compossed RNN layers."""

    def __init__(self, *args, **kwargs):
        super(RNNBlock, self).__init__(*args, **kwargs)
        self.r1 = GRU(30, 'tanh', 'sigmoid', return_sequences=True)
        self.r2 = GRU(30, 'tanh', 'sigmoid', return_sequences=True)
        self.r3 = GRU(30, 'tanh', 'sigmoid', return_sequences=True)
        self.r4 = GRU(30, 'tanh', 'sigmoid', return_sequences=True)
        self.r5 = GRU(20, 'tanh', 'sigmoid', return_sequences=True)
        self.r6 = GRU(10, 'tanh', 'sigmoid', return_sequences=True)

    def call(self, inputs, training=None):
        e = self.r1(inputs, training=training)
        e = self.r2(e, training=training)
        e = self.r3(e, training=training)
        e = self.r4(e, training=training)
        e = self.r5(e, training=training)
        e = self.r6(e, training=training)
        return e


class DecoderChain(Layer):
    """Embeded decoder.

    This is decoder part of auto-encoder pair.
    """

    def __init__(self, output_dim=None, *args, **kwargs):
        super(DecoderChain, self).__init__(*args, **kwargs)
        assert output_dim is not None
        self.output_dim = output_dim

        inits = dict(
            kernel_initializer=initializers.GlorotNormalV2(),
            bias_initializer=initializers.GlorotNormalV2(),
            kernel_regularizer=regularizers.l2(0.001),
            activity_regularizer=regularizers.l2(0.001)
        )
        self.d1 = Dense(20, 'relu', name='decoder_1', **inits)
        self.d2 = Dense(30, 'relu', name='decoder_2', **inits)
        self.d3 = Dense(output_dim, None, name='decoder_out', **inits)

    def call(self, inputs):
        d = self.d1(inputs)
        d = self.d2(d)
        d = self.d3(d)
        return d

    def get_config(self):
        return {'output_dim': self.output_dim}


class CallSeq(Model):
    """Subclass call sequence analysing model.

    This is a non-linear model and built without Keras Functional API,
    so `keras.utils.plot_model()` might produce false result.
    """

    def __init__(self, ckpt_path=None, *args, **kwargs):
        super(CallSeq, self).__init__(*args, **kwargs)

        if ckpt_path is None:
            # Use inferred global value if not available.
            seq_size = SEQ_SIZE
        else:
            ckpt_reader = tf.train.load_checkpoint(ckpt_path)
            # Find the input config attribute
            input_cfg = None
            for name in ckpt_reader.get_variable_to_shape_map():
                if 'input_layer' in name and 'OBJECT_CONFIG_JSON' in name:
                    input_cfg = json.loads(ckpt_reader.get_tensor(name))
                    break
            assert (
                input_cfg is not None
                and 'config' in input_cfg
                and 'batch_input_shape' in input_cfg['config']
            ), (
                'Checkpoint file in path %s does not contain config option '
                'for input layer.' % ckpt_path
            )
            # Load seq_size from input_cfg
            seq_size = input_cfg['config']['batch_input_shape'][-1]

            # Load seq_size from input_cfg
            batch_input_shape = input_cfg['config']['batch_input_shape']
            seq_size = batch_input_shape[-1]

        self.seq_size = seq_size

        self.input_layer = InputLayer(input_shape=(None, seq_size))
        self.n = BatchNormalization(name='batch_norm')
        self.n2 = BatchNormalization(name='batch_norm_2')

        self.decoder_chain = DecoderChain(seq_size, name='decoder')

        self.rnns = RNNBlock(name='rnns')
        self.radi_check = RadiusScoreLayer(1.0, name='score')

        self._network_nodes = (
            self.input_layer, self.n, self.n2,
            self.decoder_chain, self.rnns, self.radi_check,
        )
        self._feed_input_names = [self.input_layer.name]
        self._feed_output_names = [self.decoder_chain.name,
                                   self.radi_check.name]
        self._feed_loss_fns = [loss_fn_decoded, loss_fn_score]

        self.build((None, None, seq_size))
        if ckpt_path is not None:
            saver = tf.train.Checkpoint(model=self)
            saver.restore(tf.train.latest_checkpoint(ckpt_path))

    def call(self, inputs, training=None):
        e = self.input_layer(inputs)
        e = self.n(e, training=training)

        # Encoding pass
        e = self.rnns(e, training=training)

        e_norm = self.n2(e)

        # Decoding pass
        d = self.decoder_chain(e_norm)

        score = self.radi_check(e_norm)

        return d, score

    def decision(self, inputs):
        """Return decision value based on single or batched input.

        The input tensor must be a sequence of information vector, encoded
        using the same rule as `input_fn` in this module:

        ```
            v_i = [ x_1, x_2, ..., x_n ]
            s_i = [ v_1, v_2, ..., v_n ]
            batched_v = [ s_1, s_2, ..., s_n ]
        ```

        The input tensor must have at least the shape of
        `(seq_length, seq_size)`. `seq_size` can be evaluated automaticaly
        by this module if we have train dataset.

        This model can work with arbitrary length of input sequence.
        """
        shape = tf.shape(inputs)
        if len(shape) == 2:
            inputs = tf.expand_dims(inputs, axis=0)  # Add batch dimension

        _, outputs = self(inputs, training=False)

        if len(shape) == 2:
            # Ouput a single vector, as we received a single sequence.
            outputs = tf.squeeze(outputs, axis=0)
        signs = tf.cast(tf.sign(outputs), tf.int32)
        if _context.executing_eagerly():
            return signs.numpy()
        return signs


def loss_fn_decoded(y_true, y_pred):
    """Somewhat called mean absolute cosine similarity."""
    y_true = nn.l2_normalize(y_true, axis=-1)
    y_pred = nn.l2_normalize(y_pred, axis=-1)

    yy_mul = math_ops.abs(y_true * y_pred)

    return math_ops.reduce_mean(yy_mul)


def loss_fn_score(y_true, y_pred):
    return losses.hinge(y_true, y_pred)


def model_fn():
    return CallSeq(name='a_subclass_model')


loss_fn = [loss_fn_decoded, loss_fn_score]


# Here we find the SEQ_SIZE by evaluate a single entry in dataset,
# assumed that the dataset is uniform by trusting the generate
# function. This happen at the very end of import process for this
# module and required some power.
# The iterable interface on `Dataset` only available when eager
# execution is enabled. It's on by default on TF-2.0, but not
# for earlier versions.
# It's simplier to run this small op with eager execution enabled.
with _context.eager_mode():
    _base = tf.data.TFRecordDataset(TF_DATA_FILE)
    _base = _base.map(_parse_tensor)
    for _i in _base.take(1):
        break
    _shape = tf.shape(_i)[0]
    # This is the same as int and can be passed to TF functions
    SEQ_SIZE = _shape.numpy()


del _i
del _base
del _shape
