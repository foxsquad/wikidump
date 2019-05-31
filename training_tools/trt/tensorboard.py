import sys

from tensorboard import main


def patch_mkl():
    import ctypes
    try:
        from win32 import win32api
    except ImportError:
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


def cli():
    if 'win32' in sys.platform:
        patch_mkl()
    main.run_main()


if __name__ == "__main__":
    cli()
