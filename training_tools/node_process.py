"""Cache nodes.csv into TFRecords file.

Collection of configurable commands for process raw data.
Data being read by pandas DataFrame and process based on
config file `node_process.yml`.

Samples::
    # node_process.yml
    fields:
      field_1: command_name_1
      field_2:
        name: command_name_2
        cmd_1_arg_name: cmd_1_arg_value
"""

from __future__ import division, unicode_literals

import os
import sys
import abc
from collections import UserList
from datetime import datetime

import yaml
from absl import app, flags
from pip._internal.commands import InstallCommand

SRC_DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    'data', 'train_sample.csv')
GEO_DB_FILE = os.path.join(
    os.path.dirname(__file__),
    'data', 'GeoLite2-City.mmdb')
CFG_FILE = os.path.join(
    os.path.dirname(__file__),
    'node_process.yml')

flags.DEFINE_string('src_file', SRC_DATA_FILE,
                    'Source data file', short_name='s')
flags.DEFINE_string('geo_db_file', GEO_DB_FILE,
                    'GeoDB database.', short_name='db')
flags.DEFINE_string('cfg_file', CFG_FILE,
                    'Path to YAML config file. This file must have '
                    'at least a `fields` field.', short_name='c')
flags.DEFINE_string('output_file', None,
                    'Path to output TFRecords file.', short_name='O')
FLAGS = flags.FLAGS


def _install_missing(pkg_spec):
    install_command = InstallCommand()
    install_command.main([pkg_spec])


try:
    from dateutil import tz as dtz
except ImportError:
    _install_missing('python-dateutil')
    from dateutil import tz as dtz

try:
    import maxminddb
except ImportError:
    _install_missing('maxminddb')
    import maxminddb

try:
    import pandas as pd
except ImportError:
    _install_missing('pandas')
    import pandas as pd


def load_config(cfg_file):
    """Load config and return command set."""
    with open(cfg_file, 'r') as f:
        d = yaml.full_load(f)
        d = d['fields']

    # Build command list
    cmds = CommandSet()
    for field, value in d.items():
        if isinstance(value, str):
            command_name = value
            build_args = {}
        elif isinstance(value, dict):
            command_name = value.pop('name')
            build_args = value

        cmd_cls = Command.get(command_name)

        cmd = cmd_cls(field)  # type: Command
        cmd.load(**build_args)

        cmds.append(cmd)

    return cmds


class CommandSet(UserList):
    """A set of process commands."""

    def __init__(self, initlist=None):
        initlist = initlist or []
        return super().__init__(initlist)

    @property
    def fields(self):
        """Registered fields read provided from configuration file"""
        return [cmd.field_name for cmd in self]

    def process_line(self, line):
        r = []
        for item, command in zip(line, self):
            r += command.call(item)
        return r


class Command(metaclass=abc.ABCMeta):
    """Base command class."""

    __registered_commands__ = dict()

    def __init__(self, field_name):
        self.field_name = field_name

    @classmethod
    def load(self, **cfg):
        raise NotImplementedError('Command.load()')

    @classmethod
    def call(self, data):
        raise NotImplementedError('Command.call()')

    @staticmethod
    def get(name):
        """Return registered command by name."""
        return Command.__registered_commands__[name]  # type: Command

    @staticmethod
    def register(name):
        """Register command class with name."""
        def decorator(cls):
            Command.__registered_commands__[name] = cls
            return cls
        return decorator

# -------------------------------------
# Transform function
# -------------------------------------
@Command.register(name='raw')
class RawCommand(Command):
    """Return data piece as-is."""

    def load(self):
        return

    def call(self, data):
        return [data]


@Command.register(name='caller_id')
class CallerIdCommand(Command):
    """Process caller ID."""

    def load(self, length=20, direction='>', null_str='\x00'):
        assert direction in '<>', 'Invalid direction value.'

        self.length = length
        self.direction = direction
        self.null_str = '\x00'

    def call(self, data):
        # Filter num char, for safe
        data = str(data)
        data = [
            d for d in data
            if str.isdigit(d)
        ]
        data = ''.join(data)

        return [
            ord(c)
            for c in f'{data:{self.null_str}{self.direction}{self.length}}'
        ]


@Command.register(name='timestamp')
class TimestampCommand(Command):
    def load(self, tz='utc'):
        self.tz = dtz.gettz(tz)

    def call(self, data):
        t = datetime.fromtimestamp(data, self.tz)
        return [
            t.year // 100, t.year % 100,
            t.month, t.day,
            t.hour, t.minute, t.second,
            t.microsecond
        ]


@Command.register(name='geoip')
class GeoIpParserCommand(Command):
    def __init__(self, field_name):
        super().__init__(field_name)
        with open(GEO_DB_FILE, 'rb') as f:
            self.reader = maxminddb.open_database(f, maxminddb.MODE_FD)

    def load(self):
        return

    def call(self, data):
        src_ip = data.split('@')[1]
        geoip_data = self.reader.get(src_ip)

        latitude = geoip_data['location']['latitude']
        longitude = geoip_data['location']['longitude']
        accuracy_radius = geoip_data['location']['accuracy_radius']

        return [latitude, longitude, accuracy_radius]


@Command.register(name='scale')
class ScaleCommand(Command):
    def load(self, value=1000):
        self.value = value

    def call(self, data):
        return [data / self.value]


def sub_data_generator(nrows=None):
    command_set = load_config(FLAGS.cfg_file)
    d = pd.read_csv(FLAGS.src_file, nrows=nrows)

    sub_data = d[command_set.fields]  # type: pd.DataFrame

    for idx, fields in sub_data.iterrows():
        yield command_set.process_line(fields)


def sample_data_length():
    i = sub_data_generator(3)
    sample = next(i)

    return len(sample)


def sample_table_length():
    d = pd.read_csv(FLAGS.src_file)  # type: pd.DataFrame
    return len(d)


def cache_to_tf_records(argv):
    import tensorflow as tf
    from tensorflow.python.eager import context
    from tensorflow.python.util import deprecation
    from call_seq import TF_DATA_FILE

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, total=None, *args, **kwargs):
            lno = 0
            for item in iterable:
                yield item
                lno += 1
                sys.stdout.write(f'\r wrote {lno}/{total} lines')

    context.executing_eagerly()

    with deprecation.silence():
        base = tf.data.Dataset.from_generator(
            sub_data_generator,
            tf.float32,
            (sample_data_length(), )
        )
    base = base.map(tf.io.serialize_tensor)

    with tf.io.TFRecordWriter(FLAGS.output_file or TF_DATA_FILE) as writer:
        for line in tqdm(
                base, total=sample_table_length(),
                ncols=80, mininterval=1.0):
            writer.write(line.numpy())


if __name__ == "__main__":
    app.run(cache_to_tf_records)
