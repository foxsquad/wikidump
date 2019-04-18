#!/usr/bin/env python3

import os.path
import sys

from absl import app, flags, logging
from absl.flags import argparse_flags


FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.',
                     lower_bound=1, short_name='e')
flags.DEFINE_integer('batch_size', 100, 'Batch size for input data.',
                     lower_bound=1, short_name='b')
flags.DEFINE_float('learning_rate', 1e-4, 'Optimizer initial learning rate, '
                   'used when a new optimizer is needed to be created.',
                   lower_bound=1e-10, short_name='lr')
flags.DEFINE_integer('patience', 3, 'EarlyStopping callback patience value.',
                     lower_bound=1)

flags.DEFINE_integer('buffer', 1000, 'Buffer for shuffle action.')
flags.DEFINE_integer('seed', None, 'Buffer random seed.')
flags.DEFINE_integer('article_to_take', 10,
                     'Number of article to take to train dataset.',
                     lower_bound=1)

flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint. '
                    'Default value for checkpoint directory is '
                    '"{model-name}_ckpt".')
flags.DEFINE_bool('save_state', False, 'Whether to save training state. '
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

flags.DEFINE_bool('decorate', False, 'Enable console decoration.',
                  short_name='d')


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
    if 'win32' in sys.platform:
        patch_mkl()
    import tensorflow as tf
    import tensorflow.keras.callbacks as C

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
        'epoch_{epoch}_{val_loss:.3f}.ckpt')
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)

    def loss(a, b):
        return tf.keras.losses.sparse_categorical_crossentropy(
            a, b, True
        )

    if FLAGS.load_state:
        logging.info('Loading model from last state')
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


def patch_mkl():
    import ctypes
    from win32 import win32api
    try:
        import _thread as thread
    except ImportError:
        import thread

    try:
        # Load the DLL manually to ensure its handler gets
        # set before our handler.
        ctypes.CDLL('libmmd.dll')
        ctypes.CDLL('libifcoremd.dll')
    except OSError:
        # If the libifcoremd.dll does not exist, skip the rest steps.
        return

    print('Patching "libifcoremd.dll"... ', end='')
    try:
        # Now set our handler for CTRL_C_EVENT. Other control event
        # types will chain to the next handler.
        def handler(dwCtrlType, hook_sigint=thread.interrupt_main):
            if dwCtrlType == 0:  # CTRL_C_EVENT
                hook_sigint()
                return 1  # don't chain to the next handler
            return 0  # chain to the next handler

        win32api.SetConsoleCtrlHandler(handler, 1)

        print('Patch done.')
    except Exception:
        print('Patch failed.')


def validate_model_module(name):
    """Validate model module availability."""
    import importlib

    model_module = importlib.import_module(name)
    model_fn = model_module.model_fn
    input_fn = model_module.input_fn

    _name = model_module.__name__
    model_name = _name.split('.')[-1] if '.' in _name else _name

    return model_name, model_fn, input_fn


model_module_name = ''


def main(argv):
    argv = argv[1:]

    model_name, model_fn, input_fn = validate_model_module(model_module_name)

    # Process additional FLAGS
    if FLAGS.checkpoint_dir is None:
        FLAGS.checkpoint_dir = '{model_name}_ckpt'.format(
            model_name=model_name)
    if FLAGS.state_file is None:
        FLAGS.state_file = os.path.join('{checkpoint_dir}', '_state.hdf5')\
            .format(checkpoint_dir=FLAGS.checkpoint_dir)

    train_loop(model_name, model_fn, input_fn)


def local_config(argv=('',), **kwargs):
    parser = argparse_flags.ArgumentParser(
        prog='trainingtool',
        description='TensorFlow training bootstrap tool.')

    parser.add_argument('model_module_name', help='Model module name')

    arg0 = argv[0] if argv else ''
    ns = parser.parse_args(argv[1:])  # Strip binary name from argv

    # Populate model_module_name to global FLAGS
    global model_module_name
    model_module_name = ns.model_module_name

    return [arg0]


if __name__ == "__main__":
    app.run(main, flags_parser=local_config)
