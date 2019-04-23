"""Distributed training plugin."""

import json
import os

import yaml
from absl import flags
from absl.app import UsageError

FLAGS = flags.FLAGS

flags.DEFINE_string('configfile', 'distribute-config.yml',
                    'Config file for distribute training loop.',
                    short_name='c')
flags.DEFINE_enum('task', None, ['chief', 'worker', 'ps'],
                  'Task type for this node.',
                  short_name='t')
flags.DEFINE_integer('taskindex', None,
                     'Task index, as defined in --configfile.',
                     lower_bound=0, short_name='i')


def train_loop(model_name, model_fn, input_fn, loss_fn):
    # Early exit, do not import tensorflow as early here.
    if FLAGS.task is None:
        raise UsageError(
            'flag --task must be defined in distributed mode.')
    if FLAGS.taskindex is None:
        raise UsageError(
            'flag --taskindex must be defined in distributed mode.')

    # We are good to go here
    import tensorflow as tf

    with open(FLAGS.configfile) as f:
        configs = yaml.load(
            f.read(), Loader=yaml.SafeLoader)  # type: dict

    batch_size = configs.get('batch_size', 100)
    shuffle_buffer = configs.get('shuffle_buffer', 1000)
    shuffle_seed = configs.get('shuffle_seed', None)
    tf_random_seed = configs.get('tf_random_seed', None)
    step_counter_freq = configs.get('step_counter_freq', None)

    model_dir = configs.get('model_dir', 'model_dir')
    save_checkpoints_steps = configs.get('save_checkpoints_steps', 1000)

    tf_config = {
        'cluster': configs['cluster'],
        'task': {
            'type': FLAGS.task,
            'index': FLAGS.taskindex
        }
    }

    os.environ['TF_CONFIG'] = json.dumps(tf_config)

    hooks = []

    if step_counter_freq and step_counter_freq > 0:
        hooks += [tf.estimator.StepCounterHook(
            every_n_steps=step_counter_freq)]

    input_fn = input_fn(batch_size, shuffle_buffer, shuffle_seed)

    # Prepare distributed strategy
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.RING
    )
    config = tf.estimator.RunConfig(
        tf_random_seed=tf_random_seed,
        train_distribute=strategy,
        eval_distribute=strategy,
        model_dir=model_dir,
        save_checkpoints_steps=save_checkpoints_steps)
    classsifier = tf.estimator.Estimator(
        model_fn=model_fn_wrapper(model_fn, loss_fn),
        config=config)

    train_spec = tf.estimator.TrainSpec(input_fn=input_fn, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn)

    try:
        tf.estimator.train_and_evaluate(
            classsifier,
            train_spec=train_spec,
            eval_spec=eval_spec)
    except KeyboardInterrupt:
        return


def model_fn_wrapper(model_fn, loss_fn):
    # NOTE: As the current API version is v2.0-alpha and is expected
    # changing in the future, review this function when the final
    # v2.0 releases.
    # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf
    import tensorflow as tf

    def wrapper(features, labels, mode):
        model = model_fn()

        if not isinstance(model, tf.keras.Model):
            import inspect

            mro_tree = inspect.getmro(type(model))

            raise ValueError(
                'Output of `model_fn` is not in supported type. '
                'The `model_fn` is expected to return an instance of '
                'Keras Model or its subclass. The inheritance tree of '
                'output is: %s' % ' > '.join(mro_tree))

        logits = model(features, training=False)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'logits': logits}
            return tf.estimator.EstimatorSpec(
                labels=labels,
                predictions=predictions)

        loss = loss_fn(labels, logits)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)

        global_step = tf.compat.v1.train.get_or_create_global_step()

        optimizer = tf.compat.v1.train.AdamOptimizer(1e-2)
        optimizer_op = optimizer.minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=optimizer_op)
    return wrapper
