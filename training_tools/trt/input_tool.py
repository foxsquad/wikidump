import functools
import inspect

from absl import flags

FLAGS = flags.FLAGS
A_FINALIZE = 'finalized'
A_MODE = 'mode'
A_INPUT_CTX = 'input_context'


def _wrap_input_fn(input_fn):
    if hasattr(input_fn, A_FINALIZE) and getattr(input_fn, A_FINALIZE):
        return input_fn

    spec = inspect.signature(input_fn)
    argc = len(spec.parameters)

    @functools.wraps(input_fn)
    def wrapped_input_fn(mode, input_context=None):
        """Do post-processing for dataset from `input_fn`."""
        import tensorflow as tf

        if argc == 2:
            # Then we bind both params
            ds = input_fn(mode, input_context)
        elif argc == 1:
            # then we bind `mode` arg to input function
            ds = input_fn(mode)
        elif argc == 0:
            setattr(input_fn, A_MODE, mode)
            setattr(input_fn, A_INPUT_CTX, input_context)
            ds = input_fn()
        else:
            raise SyntaxError(
                'input_fn must have at least one (possitional) argument.')

        if input_context:
            # Currently, we must manually shard the dataset.
            # XXX: Note the change log on TF-v2
            try:
                ds = ds.apply(tf.data.experimental.filter_for_shard(
                    input_context.num_input_pipelines,
                    input_context.input_pipeline_id))
            except AttributeError:
                # FIXME: `filter_for_shard` has been removed at some points.
                pass

        ds = ds.batch(FLAGS.batch_size)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # Only shuffle batch in train mode.
            # Other mode should comsume all the (sub)dataset and
            # shuffle has no meaning at all.
            ds = ds.shuffle(FLAGS.shuffle_buffer, FLAGS.shuffle_seed)

        if FLAGS.repeat:
            ds = ds.repeat(FLAGS.repeat_count)

        return ds
    return wrapped_input_fn


def finalize(f):
    """Mark this input function as finalized.

    This will prevent the output dataset from subsequence
    post porcessing by the model module validator.
    """

    setattr(f, A_FINALIZE, True)
    return f
