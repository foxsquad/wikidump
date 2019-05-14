"""Local training plugins."""

import os

from absl import flags, logging

FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.',
                     lower_bound=1, short_name='e')

flags.DEFINE_integer('patience', 3,
                     'EarlyStopping callback patience value. '
                     'Use 0 to disable early stopping feature.',
                     lower_bound=0)

flags.DEFINE_string('checkpoint_dir', None,
                    'Directory to save checkpoint. Default value for '
                    'checkpoint directory is "{model-name}_ckpt".')
flags.DEFINE_bool('save_state', True,
                  'Whether to save training state. The optimizer state '
                  'will be loaded from this file.',
                  short_name='s')
flags.DEFINE_string('state_file', None,
                    'Path to save state file. Default value for state file is '
                    '"{checkpoint_dir}/state.h5".')
flags.DEFINE_bool('load_state', True,
                  'Resume training process with best '
                  'effort. Last train state will be loaded from saved state, '
                  'otherwise a new optimizer will be compiled with learning '
                  'rate defined in "--learning-rate".')
flags.DEFINE_bool('load_weights', True,
                  'Load weights from latest available '
                  'checkpoint, otherwise weights will be initialized with '
                  'default value in `model_fn`.')

flags.DEFINE_bool('tensorboard', False, 'Enable TensorBoard logging.')
flags.DEFINE_integer('log_freq', 10,
                     'Number of batch to write Tensor Board event log.',
                     lower_bound=1)
flags.DEFINE_bool('cleanup', True,
                  'Try to remove old tfevents files from last run.')

flags.DEFINE_bool('summary', False, 'Print out model summary after creation.')


def train_loop(model_name, model_fn, input_fn, loss_fn):
    import tensorflow as tf
    from tensorflow.python.keras.models import Model

    from .callbacks import ModelCheckpoint, SaveStateCallback, SimpleLogger

    # Process additional FLAGS
    if FLAGS.checkpoint_dir is None:
        FLAGS.checkpoint_dir = f'{model_name}_ckpt'
    if FLAGS.state_file is None:
        FLAGS.state_file = os.path.join(FLAGS.checkpoint_dir, 'state.hdf5')
    if FLAGS.prefetch is None:
        # Enable prefetch automaticaly on GPU-enabled machine and prefetch
        # argument was not specified.
        if tf.test.is_gpu_available():
            FLAGS.prefetch = True

    # Ensure that checkpoint dir exist before we move on
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir, mode=0o766)

    logging.info('Calling `input_fn` to generate dataset')

    train_dataset = input_fn(tf.estimator.ModeKeys.TRAIN)
    val_dataset = input_fn(tf.estimator.ModeKeys.EVAL)
    test_dataset = input_fn(tf.estimator.ModeKeys.PREDICT)

    if FLAGS.prefetch:
        train_dataset = train_dataset.prefetch(FLAGS.batch_size * 2)
        val_dataset = val_dataset.prefetch(FLAGS.batch_size * 2)

    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)

    logging.info('Calling `model_fn` to create model')
    model = None
    if FLAGS.load_state:
        logging.info('Loading model from last state file "%s"',
                     FLAGS.state_file)
        try:
            model = tf.keras.models.load_model(
                FLAGS.state_file,
                custom_objects={'loss_fn': loss_fn})
        except ValueError as e:
            logging.error('ValueError: %s', e)
            logging.error('Failed to load model from last state, '
                          'will construct a blank model instead.')
        except OSError as e:
            logging.error('OSError: %s', e)
            logging.error('Last state file not found, will construct '
                          'a blank model.')
    model = model or model_fn()  # type: Model

    if FLAGS.summary and model.built:
        model.summary()

    if not model.optimizer:
        logging.info('Compiling model optimizer and loss function')
        model.compile(optimizer, loss_fn)

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        ModelCheckpoint(
            FLAGS.checkpoint_dir, val_dataset=val_dataset,
            save_best_only=True, max_to_keep=5,
            load_weights_on_model_set=FLAGS.load_weights),
        SimpleLogger()]

    if FLAGS.save_state:
        callbacks.append(SaveStateCallback(FLAGS.state_file))

    if FLAGS.patience:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=FLAGS.patience,
            restore_best_weights=True))

    if FLAGS.tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=FLAGS.checkpoint_dir, histogram_freq=1,
            write_graph=True, write_images=False,
            update_freq=FLAGS.batch_size * FLAGS.log_freq,
            profile_batch=0))

    if FLAGS.cleanup:
        for dirpath, dirnames, filenames in os.walk(FLAGS.checkpoint_dir):
            _prune(dirpath, filenames)

    logging.info('Begin training process')
    try:
        model.fit(
            train_dataset, validation_data=val_dataset,
            epochs=FLAGS.epochs, verbose=FLAGS.v,
            callbacks=callbacks)
    except KeyboardInterrupt:
        pass
    except ValueError as e:
        logging.error('ValueError: %s', e)
    finally:
        logging.info('Train process done.')

    ev_test = model.evaluate(test_dataset, verbose=0)
    if isinstance(ev_test, list):
        ev_test = ev_test[-1]
    logging.info('[on test dataset] test_loss = %.4f', ev_test)


def _prune(dirpath, filenames):
    for filename in filenames:
        if '.tfevents.' in filename:
            filepath = os.path.join(dirpath, filename)
            logging.warning('Going to remove event file %s', filepath)
            os.remove(filepath)
