"""Default plugin loader for training bootstrap tool."""
from absl import flags


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
flags.DEFINE_bool('prefetch', None, 'Enable data prefetch on CPU.')


# Import plugin modules later to avoid flag definitions
# in these modules come after default flags above.
from . import train_dist, train_local  # noqa  # isort:skip


class Plugins(object):

    local = train_local
    distributed = train_dist

    def __iter__(self):
        for i in [self.local, self.distributed]:
            yield i


PLUGINS = Plugins()


# These are for callback configs
flags.DEFINE_bool('decorate', False, 'Enable console decoration.')
flags.DEFINE_integer('update_freq', 1,
                     'Train status logger update frequency, in epoch count.')
