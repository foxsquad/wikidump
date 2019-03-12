"""Wikidump reader and processor module

"""

import os

with open(os.path.join(
        os.path.dirname(__file__),
        'DUMP_VERSION')) as f:
    DUMP_VERSION = f.readline().strip()

with open(os.path.join(
        os.path.dirname(__file__),
        'TORRENT_HASH')) as f:
    HASH = f.readline().strip()


__all__ = ['DUMP_VERSION', 'HASH']
