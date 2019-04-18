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
                    '"{checkpoint-dir}/_state.hdf5".')
flags.DEFINE_bool('load_state', True, 'Resume training process with best '
                  'effort. Last train state will be loaded from saved state, '
                  'otherwise a new optimizer will be compiled with learning '
                  'rate defined in "--learning-rate".')
flags.DEFINE_bool('load_weights', True, 'Load weights from latest available '
                  'checkpoint, otherwise weights will be initialized with '
                  'default value in `model_fn`.')

flags.DEFINE_bool('decorate', False, 'Enable console decoration.')


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


def train_loop(model_name, model_fn, input_fn):
    import tensorflow as tf
    import tensorflow.keras.callbacks as C

    # Process additional FLAGS
    if FLAGS.checkpoint_dir is None:
        FLAGS.checkpoint_dir = '{model_name}_ckpt'.format(
            model_name=model_name)
    if FLAGS.state_file is None:
        FLAGS.state_file = os.path.join('{checkpoint_dir}', '_state.hdf5')\
            .format(checkpoint_dir=FLAGS.checkpoint_dir)

    nan = float('nan')
    spinner = Spinner() if FLAGS.decorate else ''

    class SimpleLogger(C.Callback):
        """A simple end-of-epoch logger."""

        def on_batch_end(self, batch, logs=None):
            if batch % 20 == 0:
                logs = logs or {}
                loss = logs.get('loss')
                print('\r {} loss {:.4f}'.format(spinner, loss), end='')

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            loss = logs.get('loss', nan)
            val_loss = logs.get('val_loss', nan)
            print(
                '\rEpoch {e} - loss: {loss:.4f}  val loss: {val_loss:.4f}'
                .format(e=epoch + 1, loss=loss, val_loss=val_loss)
            )

    class SaveStateCallback(C.Callback):
        def __init__(self, state_file):
            super().__init__()
            self.state_file = state_file

        def on_epoch_end(self, epoch, logs=None):
            self.model.save(self.state_file)

    logging.info('Calling `input_fn` to generate dataset')
    data_fn = input_fn(FLAGS.batch_size, FLAGS.buffer, FLAGS.seed)

    train_dataset = data_fn(tf.estimator.ModeKeys.TRAIN)
    vali_dataset = data_fn(tf.estimator.ModeKeys.EVAL)
    test_dataset = data_fn(tf.estimator.ModeKeys.PREDICT)

    ckpt_path = os.path.join(
        FLAGS.checkpoint_dir,
        'epoch_{epoch}_{val_loss:.4f}.ckpt')
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)

    def loss(a, b):
        return tf.keras.losses.sparse_categorical_crossentropy(
            a, b, True
        )

    if FLAGS.load_state:
        logging.info('Loading model from last state file "%s"',
                     FLAGS.state_file)
        try:
            model = tf.keras.models.load_model(
                FLAGS.state_file,
                custom_objects={'loss': loss})
        except ValueError:
            logging.error('Failed to load model from last state, '
                          'will construct a blank model instead.')
            model = model_fn()
        except OSError:
            logging.error('Last state file not found,'
                          ' will construct a blank model.')
            model = model_fn()
    else:
        model = model_fn()

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
        model.compile(optimizer, loss)

    ckpt_callback = C.ModelCheckpoint(
        ckpt_path,
        save_best_only=True,
        save_weights_only=True)
    logging.info('Restoring best `val_loss`')
    ckpt_callback.best = model.evaluate(vali_dataset, verbose=0)

    if FLAGS.save_state:
        save_state = [SaveStateCallback(FLAGS.state_file)]
    else:
        save_state = []

    try:
        model.fit(
            train_dataset, validation_data=vali_dataset,
            epochs=FLAGS.epochs, verbose=FLAGS.v,
            callbacks=[
                C.EarlyStopping(
                    monitor='val_loss', patience=1,
                    restore_best_weights=True),
                C.TerminateOnNaN(),
                ckpt_callback,
                SimpleLogger()
            ] + save_state)
    except KeyboardInterrupt:
        pass
    finally:
        print('\nTrain done.')

    ev_test = model.evaluate(test_dataset)
    print('\rEvaluation on test dataset:       {:.4f}'.format(ev_test))
