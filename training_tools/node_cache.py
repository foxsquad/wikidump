"""Cache nodes.csv into TFRecords file."""

import ipaddress
from datetime import datetime, timezone

import tensorflow as tf
from pip._internal.commands import InstallCommand

from call_seq import SEQ_SIZE, TF_DATA_FILE


def _install_missing(pkg_spec):
    install_command = InstallCommand()
    install_command.main([pkg_spec])


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


def sub_data_generator():
    d = pd.read_csv('nodes.csv')
    with open('GeoLite2-City.mmdb', 'rb') as f:
        reader = maxminddb.open_database(f, maxminddb.MODE_FD)
    null_str = '\x00'

    sub_data = d[['Call_ID', 'Caller_ID', 'Timestamp', 'Delta']]

    for idx, (call_id, caller_id, timestamp, delta_t) in sub_data.iterrows():
        src_ip = call_id.split('@')[1]

        geoip_data = reader.get(src_ip)
        latitude = geoip_data['location']['latitude']
        longitude = geoip_data['location']['longitude']
        accuracy_radius = geoip_data['location']['accuracy_radius']

        t = datetime.fromtimestamp(timestamp, timezone.utc)
        time_packed = [
            t.year // 100,
            t.year % 100,
            t.month,
            t.day,
            t.hour,
            t.minute,
            t.second,
            t.microsecond
        ]

        yield \
            [ord(i) for i in f'{caller_id:{null_str}>20}'] \
            + list(ipaddress.IPv4Address(src_ip).packed) \
            + [latitude, longitude, accuracy_radius] \
            + time_packed + [delta_t / 1000]
        # 20 + 4 + 3 + 8 + 1


def cache_to_tf_records():
    generator = sub_data_generator
    base = tf.data.Dataset.from_generator(
        generator,
        tf.float32,
        (SEQ_SIZE, )
    )
    base = base.map(tf.io.serialize_tensor)
    line_no = 0

    with tf.io.TFRecordWriter(TF_DATA_FILE) as writer:
        for line in base:
            line_no += 1
            writer.write(line.numpy())
            print(f'\r writing line {line_no}', end='')


if __name__ == "__main__":
    cache_to_tf_records()
