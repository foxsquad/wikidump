"""Wikidump reader and processor module.

"""

import os

with open(os.path.join(
        os.path.dirname(__file__),
        'scripts',
        'DUMP_VERSION')) as f:
    DUMP_VERSION = f.readline().strip()

with open(os.path.join(
        os.path.dirname(__file__),
        'scripts',
        'TORRENT_HASH')) as f:
    HASH = f.readline().strip()

BZ_FILE = 'enwiki-%s-pages-articles-multistream.xml.bz2' % DUMP_VERSION
BZ_PATH = os.path.join('data', BZ_FILE)

DEFAULT_NAMESPACE = 'http://www.mediawiki.org/xml/export-0.10/'

# Known namespaces used by Database Exporter
NSMAP = {
    None: DEFAULT_NAMESPACE,
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
}

__all__ = ['DUMP_VERSION', 'HASH', 'BZ_FILE',
           'BZ_PATH', 'DEFAULT_NAMESPACE', 'NSMAP']
