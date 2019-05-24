#!/usr/bin/env python3

import os.path
import sys

import yaml
from absl import app, flags, logging
from absl.flags import argparse_flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean('distributed', False, '\
Enable distributed training. Currently, this will use the \
multi worker distributed strategy with estimator training \
loop.')
flags.DEFINE_string('config', None, 'Config file to read from.')
flags.DEFINE_bool('summary-only', False, 'Load model and print summary only.')

# Import these pre-defined module later to keep main module flags
# appears first.
from . import default  # noqa isort:skip
from .default import PLUGINS  # noqa isort:skip


flags.adopt_module_key_flags(default)
for plugin in PLUGINS:
    flags.adopt_module_key_flags(plugin)


def patch_mkl():
    import ctypes
    try:
        from win32 import win32api
    except ImportError:
        logging.warning('Module win32api not found, could not patch DLL.')
        return
    import _thread as thread

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


class ModelModule(object):
    """Wrapper for model module.

    Avoid excessively access to other members in imported module.
    """

    def __init__(self, module):
        name = module.__name__
        assert hasattr(module, 'model_fn'), 'missing model_fn in %s' % name
        assert hasattr(module, 'input_fn'), 'missing input_fn in %s' % name
        assert hasattr(module, 'loss_fn'), 'missing loss_fn in %s' % name
        self._module = module
        self._name = name.split('.')[-1] if '.' in name else name

    @property
    def model_fn(self):
        return self._module.model_fn

    @property
    def input_fn(self):
        return self._module.input_fn

    @property
    def loss_fn(self):
        return self._module.loss_fn

    @property
    def name(self):
        return self._name


def validate_model_module(name):
    """Validate model module availability."""
    import importlib

    _model_module = importlib.import_module(name)
    model_module = ModelModule(_model_module)
    return model_module


def flags_parser(argv=('',), **_):
    parser = argparse_flags.ArgumentParser(
        prog='trt',
        description='A TensorFlow training bootstrap tool.')

    parser.add_argument('model_module_name', metavar='MODEL_MODULE',
                        help='''\
        Model module name. The module must expose these functions with
        compatible signature: model_fn(), input_fn(mode, input_context),
        loss_fn(y_true, y_pred). The signatures should compatible with
        functions pass to `EstimatorSpec`.''')
    parser.add_argument('remained', metavar='...', nargs='...',
                        help='Additional flags that might be passed to '
                             'model module.')

    arg0 = argv[0] if argv else ''
    ns = parser.parse_args(argv[1:])  # Strip binary name from argv

    # Load and validate `model_module`
    mm = validate_model_module(ns.model_module_name)
    # Update FLAGS after we load the model module, as model module
    # might defined it own flag(s)
    FLAGS([arg0] + ns.remained)

    return arg0, mm.name, mm.model_fn, mm.input_fn, mm.loss_fn


def read_config_file():
    """Read config as defined in `config` file.

    Note that any value specified here will be preceded by
    manually CLI arguments."""

    def update_flags_default(config_data):
        for k, v in config_data.items():
            if hasattr(FLAGS, k):
                FLAGS.set_default(k, v)

    # Automatic load default expected file first
    expected_file = 'trt.yml'
    if os.path.exists(expected_file):
        logging.info('Settings default value from %s', expected_file)
        with open(expected_file) as f:
            config = yaml.safe_load(f)
        # Update default values with the values in the config file
        update_flags_default(config)

    # Locate the config file in the arguments
    if FLAGS.config and os.path.exists(FLAGS.config):
        with open(FLAGS.config) as f:
            config = yaml.safe_load(f)

        update_flags_default(config)


def print_summary(model_fn, input_fn):
    """Try to print keras model summary."""
    import tensorflow as tf

    model = model_fn()
    if model.built:
        model.summary()
    else:
        logging.info('Model not built automatically.')
        logging.info('Will feed model with some provided input data.')
        dataset = input_fn(tf.estimator.ModeKeys.EVAL)
        # Feed model with some input data, which we assumed that
        # they are valid, as provided by model module.
        for train_data, _ in dataset.take(1):
            break
        model(train_data, training=False)
        model.summary()
    try:
        tf.keras.utils.plot_model(
            model, f'{model.name}_io.png',
            show_shapes=True)
    except ImportError as e:
        logging.error('Could not plot model graph.')
        logging.error(e)


def main(argv):
    *argv, model_name, model_fn, input_fn, loss_fn = argv

    if 'win32' in sys.platform:
        patch_mkl()

    if FLAGS.get_flag_value('summary-only', False):
        print_summary(model_fn, input_fn)
        return 0

    # Select train loop function
    if FLAGS.distributed:
        logging.info('Using distributed train loop.')
        m = PLUGINS.distributed
    else:
        logging.info('Using local train loop')
        m = PLUGINS.local

    logging.info('Starting train loop.')
    train_loop = m.train_loop
    train_loop(model_name, model_fn, input_fn, loss_fn)


def cli():
    # Add current working directory to system search paths
    # before attempt any import statement
    sys.path.insert(0, os.path.abspath(os.path.curdir))

    app.call_after_init(read_config_file)
    app.run(main, flags_parser=flags_parser)


if __name__ == "__main__":
    cli()
