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
flags.DEFINE_integer('patience', 3, 'EarlyStopping callback patience value.',
                     lower_bound=1)

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

flags.DEFINE_bool('decorate', False, 'Enable console decoration.')
flags.DEFINE_integer('update_freq', 1, 'Update frequency.')
flags.DEFINE_bool('summary', False, 'Print out model summary after creation.')


class Spinner(object):
    __base__ = u'↖↗↘↙'

    def __generator_fn__(self):
        counter = 0
        base = str(self.__base__)
        base_length = len(base)

        while True:
            yield base[counter]
            counter += 1

            # Avoid counter overflow for long running process
            if counter % base_length == 0:
                counter = 0

    def __init__(self):
        super().__init__()
        self.__generator__ = self.__generator_fn__()

    def __str__(self):
        return next(self.__generator__)
    __repr__ = __str__


def train_loop(model_name, model_fn, input_fn, loss_fn):
    import tensorflow as tf
    from tensorflow.python.keras import callbacks as C
    from tensorflow.python.estimator.estimator import ModeKeys

    # Process additional FLAGS
    if FLAGS.checkpoint_dir is None:
        FLAGS.checkpoint_dir = '{model_name}_ckpt'.format(
            model_name=model_name)
    if FLAGS.state_file is None:
        FLAGS.state_file = os.path.join(
            FLAGS.checkpoint_dir, '_state.h5')

    nan = float('nan')
    spinner = Spinner() if FLAGS.decorate else ''

    class SimpleLogger(C.Callback):
        """A simple end-of-epoch logger."""

        def on_batch_end(self, batch, logs=None):
            if FLAGS.update_freq and batch % FLAGS.update_freq != 0:
                return
            logs = logs or {}
            loss = logs.get('loss')
            print(f' {spinner} loss {loss:.4f}\r', end='')

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            loss = logs.get('loss', nan)
            val_loss = logs.get('val_loss', nan)
            print(f'Epoch {epoch + 1} - '
                  f'loss: {loss:.4f}  '
                  f'val loss: {val_loss:.4f}\r')

    class SaveStateCallback(C.Callback):
        def __init__(self, state_file):
            super().__init__()
            self.state_file = state_file

        def on_epoch_end(self, epoch, logs=None):
            self.model.save(self.state_file)

    logging.info('Calling `input_fn` to generate dataset')
    data_fn = input_fn(FLAGS.batch_size, FLAGS.buffer, FLAGS.seed)

    train_dataset = data_fn(ModeKeys.TRAIN)
    vali_dataset = data_fn(ModeKeys.EVAL)
    test_dataset = data_fn(ModeKeys.PREDICT)

    ckpt_path = os.path.join(
        FLAGS.checkpoint_dir,
        'epoch_{epoch}_{val_loss:.4f}.ckpt')
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)

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
            model = model_fn()
        except OSError as e:
            logging.error(e)
            logging.error('Last state file not found,'
                          ' will construct a blank model.')
            model = model_fn()
    else:
        model = model_fn()

    if FLAGS.summary:
        model.summary()

    if FLAGS.load_weights:
        if latest_checkpoint is not None:
            logging.info('Restoring model weights from latest checkpoint "%s"',
                         latest_checkpoint)
            try:
                model.load_weights(latest_checkpoint)
            except ValueError:
                logging.error('Latest checkpoint at "%s" not compatible with '
                              'current model structure.', latest_checkpoint)
        else:
            logging.warning('Latest checkpoint not found at dir "%s", '
                            'will initialize model weights as defined in '
                            '`model_fn`.', FLAGS.checkpoint_dir)

    if not model.optimizer:
        logging.info('Compilling model optimizer and loss function')
        model.compile(optimizer, loss_fn)

    ckpt_callback = C.ModelCheckpoint(
        ckpt_path,
        save_best_only=True,
        save_weights_only=True)

    logging.info('Calculating `val_loss` with current model state')
    ckpt_callback.best = model.evaluate(vali_dataset, verbose=0)
    logging.info('Current val_loss: %.3f', ckpt_callback.best)

    if FLAGS.save_state:
        save_state = [SaveStateCallback(FLAGS.state_file)]
    else:
        save_state = []

    logging.info('Begin training process')
    try:
        model.fit(
            train_dataset, validation_data=vali_dataset,
            epochs=FLAGS.epochs, verbose=FLAGS.v,
            callbacks=[
                C.EarlyStopping(
                    monitor='val_loss',
                    patience=FLAGS.patience,
                    restore_best_weights=True),
                C.TerminateOnNaN(),
                ckpt_callback,
                SimpleLogger()
            ] + save_state)
    except KeyboardInterrupt:
        pass
    finally:
        logging.info('Train process done.')
        if FLAGS.save_state:
            logging.info('Saving current training state')
            model.save(FLAGS.state_file)

    ev_test = model.evaluate(test_dataset, verbose=0)
    logging.info('[on test dataset] test_loss = %.4f', ev_test)
