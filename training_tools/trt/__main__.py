#!/usr/bin/env python3

import os.path
import sys

import yaml
from absl import app, flags, logging
from absl.flags import argparse_flags

from trt.default import PLUGINS
FLAGS = flags.FLAGS


flags.DEFINE_boolean('distributed', False, '\
Enable distributed training. Currently, this will use the \
multiworker distributed strategy with estimator training \
loop.')
flags.DEFINE_string('config', None, 'Config file to read from.')

for plugin in PLUGINS:
    flags.adopt_module_key_flags(plugin)


def patch_mkl():
    import ctypes
    try:
        from win32 import win32api
    except ImportError:
        logging.warning('Module win32 not found, couldd not patch DLL.')
        return
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
    loss_fn = model_module.loss_fn

    _name = model_module.__name__
    model_name = _name.split('.')[-1] if '.' in _name else _name

    return model_name, model_fn, input_fn, loss_fn


def main(argv):
    *argv, model_name, model_fn, input_fn, loss_fn = argv

    if 'win32' in sys.platform:
        patch_mkl()

    # Select train loop function
    if FLAGS.distributed:
        m = PLUGINS.distributed
    else:
        m = PLUGINS.local

    train_loop = m.train_loop
    train_loop(model_name, model_fn, input_fn, loss_fn)


def local_config(argv=('',), **kwargs):
    parser = argparse_flags.ArgumentParser(
        prog='trainingtool',
        description='A TensorFlow training bootstrap tool.')

    parser.add_argument('model_module_name', metavar='MODEL_MODULE',
                        help='''\
        Model module name. The module must expose these functions with
        compatible signature: model_fn(),
        input_fn(batch_size, shuffle_buffer, shuffle_seed),
        loss_fn(y_true, y_pred)''')
    parser.add_argument('remained', metavar='...', nargs='...',
                        help='Additional flags that might be passed to '
                        'model module.')

    arg0 = argv[0] if argv else ''
    ns = parser.parse_args(argv[1:])  # Strip binary name from argv

    # Load the model_module
    model_name, model_fn, input_fn, loss_fn =\
        validate_model_module(ns.model_module_name)
    # Update FLAGS after we load the model module, as model module
    # might defined it own flag(s)
    FLAGS([arg0] + ns.remained)

    return [arg0, model_name, model_fn, input_fn, loss_fn]


def read_config_file():
    """Read config as defined in `config` file.

    Note that any value specified here will be preceded by
    manually CLI arguments."""
    # Locate the config file in the arguments
    if FLAGS.config and os.path.exists(FLAGS.config):
        with open(FLAGS.config) as f:
            config = yaml.safe_load(f)

        # Update default values with the values in the config file
        for k, v in config.items():
            if hasattr(FLAGS, k):
                FLAGS.set_default(k, v)


if __name__ == "__main__":
    # Add current working directory to system search paths
    # before atempt any import statement
    sys.path.insert(0, os.path.abspath(os.path.curdir))

    app.call_after_init(read_config_file)
    app.run(main, flags_parser=local_config)
