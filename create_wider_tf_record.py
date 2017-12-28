#!/usr/bin/env python

r"""Convert raw WIDER dataset to TFRecord for object_detection.
Example usage:
    python create_wider_tf_record.py \
        --data_dir=/home/user/wider \
        --output_dir=/home/user/wider/tf_records
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import logging
import tensorflow as tf

from wider.widerface import Wider
from wider.widerface.tensorflow_exporter import TensorflowExporter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('dataset', '', 'widerface or fddb.')
flags.DEFINE_string('data_dir', '', 'Root directory to raw WIDER dataset.')
flags.DEFINE_string('output_dir', '', 'Path to output TFRecords')
FLAGS = flags.FLAGS


def main(_):

    if FLAGS.dataset == 'widerface':
        from wider.widerface import Wider
        from wider.widerface.darknet_exporter import DarknetExporter
        ds = Wider(FLAGS.data_dir)
        exporter = DarknetExporter(ds)
        exporter.export(FLAGS.output_dir)
    elif FLAGS.dataset == 'fddb':
        from wider.fddb import FDDB
        from wider.fddb.darknet_exporter import DarknetExporter
        ds = FDDB(FLAGS.data_dir)
        exporter = DarknetExporter(ds)
        exporter.export(FLAGS.output_dir)
    else:
        logging.error('Invalid dataset name %s', FLAGS.dataset)


if __name__ == '__main__':
    tf.app.run()
