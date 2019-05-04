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

# Args for repeatable distributed training
flags.DEFINE_integer('shuffle_buffer', 10000,
                     'Buffer value for dataset shuffle action.')
flags.DEFINE_integer('shuffle_seed', None,
                     'Shuffle seed value. Unspecified means no seed.')
flags.DEFINE_integer('tf_random_seed', None,
                     'TensorFlow random seed. Unspecified means no seed.')
flags.DEFINE_integer('step_counter_freq', None,
                     'Step counter frequency. Must be a positive integer '
                     'if specified.', lower_bound=1)
flags.DEFINE_integer('save_checkpoints_freq', 1000,
                     'Save checkpoint frequency, counted as global steps. '
                     'Note that saving too frequently might caught slow '
                     'in training progress',
                     lower_bound=1)
flags.DEFINE_string('model_dir', 'model_dir',
                    'Model directory, to save checkpoint and TensorBoard '
                    'event file.')

flags.DEFINE_boolean('repeat', False, '\
Repeat dataset on distributed training. Note that this flag \
does not have effect on local training process. This flag is \
intended for indefinite training on distributed system.')


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

    tf_config = {
        'cluster': configs['cluster'],
        'task': {
            'type': FLAGS.task,
            'index': FLAGS.taskindex
        }
    }

    os.environ['TF_CONFIG'] = json.dumps(tf_config)

    hooks = []

    if FLAGS.step_counter_freq:
        hooks.append(tf.estimator.StepCounterHook(
            every_n_steps=FLAGS.step_counter_freq))

    input_fn = input_fn(FLAGS.batch_size, FLAGS.shuffle_buffer, FLAGS.shuffle_seed)

    # Prepare distributed strategy
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    config = tf.estimator.RunConfig(
        tf_random_seed=FLAGS.tf_random_seed,
        train_distribute=strategy,
        eval_distribute=strategy,
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_freq)
    # classifier = tf.estimator.Estimator(
    #     model_fn=model_fn_wrapper(model_fn, loss_fn),
    #     config=config)
    model = model_fn()
    model.compile(
        loss=loss_fn,
        optimizer=tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate))
    classifier = tf.keras.estimator.model_to_estimator(
        keras_model=model, config=config)

    train_spec = tf.estimator.TrainSpec(input_fn=input_fn, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn)

    try:
        tf.estimator.train_and_evaluate(
            classifier,
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
