import json
import os

import numpy as np
import tensorflow as tf
from absl import flags
from tensorflow.python.eager import context as _context
from tensorflow.python.keras import Model, activations, \
    constraints, initializers, losses, regularizers
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.layers import GRU, \
    BatchNormalization, Dense, InputLayer, Layer
from tensorflow.python.ops import math_ops, nn

FLAGS = flags.FLAGS

TF_DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    'data', 'real_nodes.tfrecord')


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
            i,  # Expected decoder output is the same as input
            tf.constant(1.0, tf.float32, ()),  # Expected positive outcome
            # tf.constant(0.0, tf.float32, ()),  # Expected formal loss
        )
    )


def _parse_tensor(tensor):
    x = tf.io.parse_tensor(tensor, tf.float32)
    return x


def input_fn(mode):
    d = tf.data.TFRecordDataset(TF_DATA_FILE)
    d = d.map(_parse_tensor)
    init_state = tf.constant(
        0, dtype=tf.float32,
        shape=(FLAGS.timesteps, M_ATTR.SEQ_SIZE))
    d = d.apply(tf.data.experimental.scan(
        init_state, fold_to_sequence))

    # The first `train_seq_window` of records is all zeros,
    # due to `init_state`, so we would skip those here.
    d = d.skip(FLAGS.timesteps)
    if mode == tf.estimator.ModeKeys.TRAIN:
        d = d.skip(20000).take(10000)
    elif mode == tf.estimator.ModeKeys.EVAL:
        d = d.take(10000)
    else:
        d = d.skip(10000).take(10000)
    d = d.map(to_tuple)

    return d


class RadiusScoreLayer(Layer):
    """A simpler version of SVM on hyper-space.

    The output of this layer is directly the decision values
    of wheather data point(s) is inside or outside the hyper-
    sphere surface.
    """

    def __init__(self, init_radius=0.5, n=1.0, **kwargs):
        super(RadiusScoreLayer, self).__init__(**kwargs)
        assert 0.0 < n <= 1.0, 'Control value `n` must in range (0, 1]'
        self.init_radius = init_radius
        self.n = n

    def build(self, input_shape):
        # Hyper-space dimension size, return from previous layer.
        # We expect the input tensor has 3 dimensions and last one
        # must available.
        assert input_shape.ndims == 3
        assert input_shape[-1] is not None
        self.dim = input_shape[-1]

        self.input_spec = InputSpec(axes={-1: self.dim})

        self.radius = self.add_weight(
            name='hyper_sphere_radius', shape=(),
            initializer=initializers.constant(self.init_radius),
            regularizer=regularizers.l1(1),  # This will push the radius as small as possible.
            constraint=constraints.NonNeg(),
            trainable=True)

        self.center = self.add_weight(
            name='hyper_sphere_center', shape=(self.dim, ),
            initializer=initializers.glorot_uniform(),
            trainable=True)

        super(RadiusScoreLayer, self).build(input_shape)

    def call(self, input_tensor):
        # This is the formal loss value.
        # formal_loss = (
        #     self.radius**2
        #     + (
        #         tf.math.reduce_sum(self.slack_var)
        #         / self.n
        #         / FLAGS.timesteps
        #     ))

        cx = tf.norm(input_tensor - self.center, axis=-1)
        return activations.tanh(self.radius**2 - cx**2)

    def sign(self, input_tensor):
        """Return predicted sign value for input tensor."""
        cx = tf.norm(input_tensor - self.center, axis=-1)
        return tf.sign(self.radius**2 - cx**2)


class RNNBlock(Layer):
    """Composed RNN layers.

    This layer encapsulates a stack of recurrent layers. Each will process
    input signal into a latent space by its own rules.

    This layer could act as encoder pass in auto-encoder pair for a larger
    model.
    """

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
    """Embedded decoder.

    This is decoder part of auto-encoder pair.
    """

    def __init__(self, output_dim=None, *args, **kwargs):
        super(DecoderChain, self).__init__(*args, **kwargs)
        assert output_dim is not None
        self.output_dim = output_dim

        # Common initializers
        inits = dict(
            kernel_initializer=initializers.glorot_uniform(),
            bias_initializer=initializers.glorot_uniform(),
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
            seq_size = M_ATTR.SEQ_SIZE
        else:
            ckpt_reader = tf.train.load_checkpoint(ckpt_path)
            # Find the input config attribute
            first_layer_shape = None
            for name, value in ckpt_reader.get_variable_to_shape_map().items():
                if 'n/moving_mean' in name or 'batch_norm/moving_mean' in name:
                    first_layer_shape = value
                    break
            assert first_layer_shape is not None, (
                'Checkpoint file in path %s does not contain config option '
                'for input layer.\nDebug string:\n %s' % (
                    ckpt_path, ckpt_reader.debug_string().decode())
            )
            # Load seq_size from input_cfg
            seq_size = first_layer_shape[-1]

        self.seq_size = seq_size

        self.input_layer = InputLayer(input_shape=(None, seq_size))
        self.n = BatchNormalization(name='batch_norm')
        self.n2 = BatchNormalization(name='post_norm')

        self.decoder_chain = DecoderChain(seq_size, name='decoder')

        self.rnns = RNNBlock(name='rnns')
        self.ball_layer = RadiusScoreLayer(1.0, 0.1, name='score')

        self.build((None, None, seq_size))
        if ckpt_path is not None:
            saver = tf.train.Checkpoint(model=self)
            saver.restore(tf.train.latest_checkpoint(ckpt_path))

    def _set_output_attrs(self, outputs):
        super()._set_output_attrs(outputs)
        # Override output names
        self.output_names = ['decoder', 'score']
        # FIXME: Why did we have to do this?
        # The estimator spec (maybe) doesn't override this function,
        # but it could correctly collect the output names somewhere.
        # May be we need to dig deep into tf.estimator module?

    def call(self, inputs, training=None):
        e = self.input_layer(inputs)
        e = self.n(e, training=training)

        # Encoding pass
        e = self.rnns(e, training=training)

        # Decoding pass
        d = self.decoder_chain(e)

        score = self.ball_layer(e)

        return {'decoder': d, 'score': score}

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
        `(seq_length, seq_size)`. `seq_size` can be evaluated automatically
        by this module if we have train dataset.

        This model can work with arbitrary length of input sequence.
        """
        shape = tf.shape(inputs)
        if len(shape) == 2:
            inputs = tf.expand_dims(inputs, axis=0)  # Add batch dimension

        e = self.input_layer(inputs)
        e = self.n(e, training=False)
        e = self.rnns(e, training=False)
        outputs = self.ball_layer.sign(e)

        if len(shape) == 2:
            # Output a single vector, as we received a single sequence.
            outputs = tf.squeeze(outputs, axis=0)
        signs = tf.cast(outputs, tf.int32)
        if _context.executing_eagerly():
            return signs.numpy()
        return signs

    _sig_to_text = {
        np.int32(-1): 'BAD',
        np.int32(0): 'UNKNOWN',
        np.int32(1): 'OK'
    }

    @classmethod
    def decision_to_text(cls, outputs):
        return [cls._sig_to_text[i] for i in outputs]


def loss_fn_decoded(y_true, y_pred):
    """Somewhat called mean absolute cosine similarity."""
    y_true = nn.l2_normalize(y_true, axis=-1)
    y_pred = nn.l2_normalize(y_pred, axis=-1)

    yy_mul = math_ops.abs(y_true * y_pred)

    return math_ops.reduce_mean(yy_mul)


def loss_fn_score(y_true, y_pred):
    return losses.hinge(y_true, y_pred)


def loss_fn_formal(y_true, y_pred):
    return losses.mse(y_true, y_pred)


def model_fn():
    return CallSeq(name='a_subclass_model')


loss_fn = {'decoder': loss_fn_decoded, 'score': loss_fn_score}

metrics = {'score': 'acc'}


class _MAttr(object):
    """Deferred attribute inference."""
    __cached_seq_size__ = None

    @property
    def SEQ_SIZE(self):
        """Inferred sequence size, as read from dataset."""

        if self.__cached_seq_size__ is None:
            # Here we find the SEQ_SIZE by evaluate a single entry in dataset,
            # assumed that the dataset is uniform by trusting the generate
            # function.
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
                self.__cached_seq_size__ = _shape.numpy()

        return self.__cached_seq_size__


M_ATTR = _MAttr()
