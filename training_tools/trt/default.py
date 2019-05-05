"""Default plugin loader for training bootstrap tool."""
from collections import namedtuple

from absl import flags

from . import train_dist, train_local

Plugins = namedtuple('Plugins', ['local', 'distributed'])

PLUGINS = Plugins(train_local, train_dist)


flags.DEFINE_integer('batch_size', 100, 'Batch size of input data.',
                     lower_bound=1, short_name='b')
flags.DEFINE_float('learning_rate', 1e-4,
                   'Initial learning rate for new optimizer, used when '
                   'a new optimizer is created.',
                   lower_bound=1e-10, short_name='lr')

flags.DEFINE_integer('shuffle_buffer', 10000,
                     'Buffer value for dataset shuffle action.')
flags.DEFINE_integer('shuffle_seed', None,
                     'Shuffle seed value. Unspecified means no seed.')
flags.DEFINE_integer('tf_random_seed', None,
                     'TensorFlow random seed. Unspecified means no seed.')
