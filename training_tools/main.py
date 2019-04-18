#!/usr/bin/env python3

import os.path
import sys

from absl import app, flags
from absl.flags import argparse_flags

from default import PLUGINS
FLAGS = flags.FLAGS


flags.DEFINE_bool('distributed', False, '\
Enable distributed training. Currently, this will use the \
multiworker distributed strategy with estimator training \
loop.')


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

    if 'win32' in sys.platform:
        patch_mkl()

    # Select train loop function
    if FLAGS.distributed:
        m = PLUGINS.distributed
    else:
        m = PLUGINS.local

    train_loop = m.train_loop
    train_loop(model_name, model_fn, input_fn)


def local_config(argv=('',), **kwargs):
    parser = argparse_flags.ArgumentParser(
        prog='trainingtool',
        description='TensorFlow training bootstrap tool.')

    parser.add_argument('model_module_name', metavar='MODEL',
                        help='Model module name')

    arg0 = argv[0] if argv else ''
    ns = parser.parse_args(argv[1:])  # Strip binary name from argv

    # Populate model_module_name
    global model_module_name
    model_module_name = ns.model_module_name

    return [arg0]


if __name__ == "__main__":
    # Add current working directory to system search paths
    # before atempt any import statement
    sys.path.insert(0, os.path.abspath(os.path.curdir))

    app.run(main, flags_parser=local_config)
