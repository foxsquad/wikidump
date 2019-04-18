"""Default plugin loader for training bootstrap tool."""
from collections import namedtuple

import train_local
import train_dist

Plugins = namedtuple('Plugins', ['local', 'distributed'])

PLUGINS = Plugins(train_local, train_dist)
