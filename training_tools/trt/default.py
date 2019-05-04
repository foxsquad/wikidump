"""Default plugin loader for training bootstrap tool."""
from collections import namedtuple

from . import train_dist, train_local

Plugins = namedtuple('Plugins', ['local', 'distributed'])

PLUGINS = Plugins(train_local, train_dist)
