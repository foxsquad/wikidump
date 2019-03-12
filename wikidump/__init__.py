"""Wikidump reader and processor module.

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

BZ_FILE = 'enwiki-%s-pages-articles-multistream.xml.bz2' % DUMP_VERSION
BZ_PATH = os.path.join('data', BZ_FILE)

__all__ = ['DUMP_VERSION', 'HASH', 'BZ_FILE', 'BZ_PATH']
