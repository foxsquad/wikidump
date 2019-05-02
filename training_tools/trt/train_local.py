"""Local training plugins."""

import os

from absl import flags, logging

FLAGS = flags.FLAGS


flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.',
                     lower_bound=1, short_name='e')
flags.DEFINE_integer('batch_size', 100, 'Batch size of input data.',
                     lower_bound=1, short_name='b')
flags.DEFINE_float('learning_rate', 1e-4, '\
Initial learning rate for new optimizer, used when \
a new optimizer is created.', lower_bound=1e-10, short_name='lr')
flags.DEFINE_integer('patience', 3, 'EarlyStopping callback patience value. '
                     'Use 0 to disable early stopping feature.',
                     lower_bound=0)

flags.DEFINE_integer('buffer', 1000, 'Shuffle buffer value for train dataset.',
                     lower_bound=1)
flags.DEFINE_integer('seed', None, 'Shuffle random seed.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint. '
                    'Default value for checkpoint directory is '
                    '"{model-name}_ckpt".')
flags.DEFINE_bool('save_state', True, 'Whether to save training state. '
                  'The optimizer state will be loaded from this file.',
                  short_name='s')
flags.DEFINE_string('state_file', None, 'Path to save state file. '
                    'Default value for state file is '
                    '"{checkpoint_dir}/_state.h5".')
flags.DEFINE_bool('load_state', True, 'Resume training process with best '
                  'effort. Last train state will be loaded from saved state, '
                  'otherwise a new optimizer will be compiled with learning '
                  'rate defined in "--learning-rate".')
flags.DEFINE_bool('load_weights', True, 'Load weights from latest available '
                  'checkpoint, otherwise weights will be initialized with '
                  'default value in `model_fn`.')

flags.DEFINE_bool('tensorboard', False, 'Enable TensorBoard logging.')
flags.DEFINE_integer('log_freq', 10, 'Number of batch to write log.',
                     lower_bound=1)

flags.DEFINE_bool('decorate', False, 'Enable console decoration.')
flags.DEFINE_integer('update_freq', 1, 'Update frequency.')
flags.DEFINE_bool('summary', False, 'Print out model summary after creation.')


def train_loop(model_name, model_fn, input_fn, loss_fn):
    import tensorflow as tf
    from tensorflow.python.keras import callbacks as C

    from trt.callbacks import ModelCheckpoint, SaveStateCallback, SimpleLogger

    # Process additional FLAGS
    if FLAGS.checkpoint_dir is None:
        FLAGS.checkpoint_dir = f'{model_name}_ckpt'
    if FLAGS.state_file is None:
        FLAGS.state_file = os.path.join(FLAGS.checkpoint_dir, '_state.hdf5')

    logging.info('Calling `input_fn` to generate dataset')
    data_fn = input_fn(FLAGS.batch_size, FLAGS.buffer, FLAGS.seed)

    train_dataset = data_fn(tf.estimator.ModeKeys.TRAIN)
    val_dataset = data_fn(tf.estimator.ModeKeys.EVAL)
    test_dataset = data_fn(tf.estimator.ModeKeys.PREDICT)

    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)

    model = None
    if FLAGS.load_state:
        logging.info('Loading model from last state file "%s"',
                     FLAGS.state_file)
        try:
            model = tf.keras.models.load_model(
                FLAGS.state_file,
                custom_objects={'loss_fn': loss_fn})
        except ValueError as e:
            logging.error(e)
            logging.error('Failed to load model from last state, '
                          'will construct a blank model instead.')
        except OSError as e:
            logging.error(e)
            logging.error('Last state file not found,'
                          ' will construct a blank model.')
    model = model or model_fn()

    if FLAGS.summary:
        model.summary()

    if not model.optimizer:
        logging.info('Compiling model optimizer and loss function')
        model.compile(optimizer, loss_fn)

    ckpt_callback = ModelCheckpoint(
        FLAGS.checkpoint_dir, val_dataset=val_dataset,
        save_best_only=True, max_to_keep=5,
        load_weights_on_model_set=FLAGS.load_weights)

    if FLAGS.save_state:
        save_state = [SaveStateCallback(FLAGS.state_file)]
    else:
        save_state = []

    if FLAGS.patience:
        early_stopping = [C.EarlyStopping(
            monitor='val_loss',
            patience=FLAGS.patience,
            restore_best_weights=True)]
    else:
        early_stopping = []

    if FLAGS.tensorboard:
        tfb = [C.TensorBoard(
            log_dir=FLAGS.checkpoint_dir,
            histogram_freq=0, write_graph=True,
            write_images=False,
            update_freq=FLAGS.batch_size * FLAGS.log_freq)]
    else:
        tfb = []

    logging.info('Begin training process')
    try:
        model.fit(
            train_dataset, validation_data=val_dataset,
            epochs=FLAGS.epochs, verbose=FLAGS.v,
            callbacks=[
                C.TerminateOnNaN(),
                ckpt_callback,
                SimpleLogger()
            ] + early_stopping + save_state + tfb)
    except KeyboardInterrupt:
        pass
    finally:
        logging.info('Train process done.')
        if FLAGS.save_state:
            logging.info('Saving current training state')
            model.save(FLAGS.state_file)

    ev_test = model.evaluate(test_dataset, verbose=0)
    logging.info('[on test dataset] test_loss = %.4f', ev_test)
