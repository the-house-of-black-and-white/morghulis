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

from wider import Wider

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw WIDER dataset.')
flags.DEFINE_string('output_dir', '', 'Path to output TFRecords')
FLAGS = flags.FLAGS


def main(_):
    wider = Wider(FLAGS.data_dir)
    wider.generate_tf_records(FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
