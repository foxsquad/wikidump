import os

import tensorflow as tf
from absl import flags
from tensorflow.python.keras import Model, initializers
from tensorflow.python.keras.activations import sigmoid
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.layers import GRU, \
    BatchNormalization, Dense, InputLayer, Layer
from tensorflow.python.keras.losses import MAE
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.ops import math_ops, nn

FLAGS = flags.FLAGS

TF_DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    'data', 'node_cache.tfrecord')

SEQ_SIZE = (
    20   # 20 chars of caller ID
    + 4  # 4 blocks of IPv4 address
    + 3  # latitude, longtitude, accuracy radius
    + 8  # packed timestamp, yy--, --yy, mm, dd, HH, MM, SS, ms
    + 1  # delta time of the same caller
)

flags.DEFINE_integer('timesteps', 20, 'Train input sequence value')

KERAS_F = 'KERAS_F' in os.environ and os.environ.get('KERAS_F', '')


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
            tf.constant(1.0, shape=(FLAGS.timesteps, 1))
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
        d = base.skip(400)
    elif mode == tf.estimator.ModeKeys.EVAL:
        d = base.take(200)
    else:
        d = base.skip(200).take(200)

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
    def __init__(self, init_radius=0.5, **kwargs):
        super(RadiusScoreLayer, self).__init__(**kwargs)
        self.init_radius = init_radius

    def build(self, input_shape):
        # Hyper-space dimension size. This is the last dimension
        # returns from recurent stack.
        self.dim = input_shape[-1]

        self.input_spec = InputSpec(axes={-1: self.dim})

        self.radius = self.add_weight(
            'hyper_sphere_radius',
            shape=(), dtype=tf.float32,
            initializer=initializers.ConstantV2(self.init_radius),
            regularizer=l2(0.01),
            trainable=True)

        self.center = self.add_weight(
            'hyper_sphere_center',
            shape=(self.dim, 1),
            initializer=initializers.GlorotUniformV2(),
            regularizer=l2(0.001),
            trainable=True)

        super(RadiusScoreLayer, self).build(input_shape)

    def call(self, input_tensor):
        i_shape = tf.shape(input_tensor)
        x = tf.reshape(
            input_tensor,
            # This indicate a column vector of shape
            # (batch_size, seq_size, vec_size, 1)
            (i_shape[0], i_shape[1], self.dim, 1)
        )
        x = x - self.center
        x = tf.norm(x, axis=-2)

        # This is used to simulate soft-sign function,
        # which is a soft, continuous function and
        # differentiable at any point.
        diff = sigmoid(self.radius ** 2 - x ** 2) * 2 - 1
        return diff


class RNNBlock(Layer):
    """Compossed RNN layers."""

    def __init__(self, *args, **kwargs):
        super(RNNBlock, self).__init__(*args, **kwargs)
        self.r1 = GRU(30, 'tanh', 'sigmoid',
                      return_sequences=True)
        self.r2 = GRU(30, 'tanh', 'sigmoid',
                      return_sequences=True)
        self.r3 = GRU(30, 'tanh', 'sigmoid',
                      return_sequences=True)
        self.r4 = GRU(30, 'tanh', 'sigmoid',
                      return_sequences=True)
        self.r5 = GRU(20, 'tanh', 'sigmoid',
                      return_sequences=True)
        self.r6 = GRU(10, 'tanh', 'sigmoid',
                      return_sequences=True)

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

        k_reg = l2(0.001)
        a_reg = l2(0.001)

        k_init = initializers.GlorotNormalV2()
        b_init = initializers.GlorotNormalV2()

        self.d1 = Dense(20, 'relu',
                        kernel_initializer=k_init,
                        bias_initializer=b_init,
                        kernel_regularizer=k_reg,
                        activity_regularizer=a_reg,
                        name='decoder_1')
        self.d2 = Dense(30, 'relu',
                        kernel_initializer=k_init,
                        bias_initializer=b_init,
                        kernel_regularizer=k_reg,
                        activity_regularizer=a_reg,
                        name='decoder_2')
        self.d3 = Dense(output_dim,
                        kernel_initializer=k_init,
                        bias_initializer=b_init,
                        kernel_regularizer=k_reg,
                        activity_regularizer=a_reg,
                        name='decoder_out')
        self.output_dim = output_dim

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

    def __init__(self, *args, **kwargs):
        super(CallSeq, self).__init__(*args, **kwargs)
        self.input_layer = InputLayer(input_shape=(None, SEQ_SIZE))
        self.n = BatchNormalization(name='batch_norm')
        self.n2 = BatchNormalization(name='batch_norm_2')

        self.decoder_chain = DecoderChain(SEQ_SIZE, name='decoder')

        self.rnns = RNNBlock(name='rnns')
        self.radi_check = RadiusScoreLayer(1.0, name='score')

        self._network_nodes = {
            self.input_layer, self.n, self.n2,
            self.decoder_chain,
            self.rnns, self.radi_check,
        }
        self._feed_input_names = [self.input_layer.name]
        self._feed_output_names = [self.decoder_chain.name,
                                   self.radi_check.name]
        self._feed_loss_fns = [loss_fn_decoded, loss_fn_score]

    def call(self, inputs, training=None):
        e = self.input_layer(inputs)
        e = self.n(e, training=training)

        # Encoding pass
        e = self.rnns(e, training=training)

        e_norm = self.n2(e)

        # Decoding pass
        d = self.decoder_chain(e_norm)

        score = self.radi_check(e_norm)

        return [d, score]


def loss_fn_decoded(y_true, y_pred):
    """Somewhat called mean absolute cosine similarity."""
    y_true = nn.l2_normalize(y_true, axis=-1)
    y_pred = nn.l2_normalize(y_pred, axis=-1)

    yy_mul = math_ops.abs(y_true * y_pred)

    return math_ops.reduce_mean(yy_mul)


def loss_fn_score(y_true, y_pred):
    return MAE(y_true, y_pred)


def model_fn():
    return CallSeq(name='call_seq_subclass')


loss_fn = [loss_fn_decoded, loss_fn_score]
