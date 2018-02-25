#!/usr/bin/env python

r"""Downloads the given dataset to destination directory.
Example usage:
    python download_dataset.py \
        --dataset=widerface \
        --output_dir=/home/user/widerface/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import logging
import re
import sys
from collections import defaultdict

import os

import tensorflow as tf

# from . import Wider
# from morghulis.os_utils import ensure_dir

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

FILENAME_RE = re.compile(r'.*/\d+--\w+/(.*)')


def load_tfrecord(file_name):
    features = {'x': tf.FixedLenFeature([2], tf.int64)}
    data = []
    for s_example in tf.python_io.tf_record_iterator(file_name):
        example = tf.parse_single_example(s_example, features=features)
        data.append(tf.expand_dims(example['x'], 0))
    return tf.concat(0, data)


def items(tf_record):
    example = tf.train.Example()
    for record in tf.python_io.tf_record_iterator(tf_record):
        example.ParseFromString(record)
        f = example.features.feature
        filename = f['image/filename'].bytes_list.value[0]
        scores = f['image/detection/score'].float_list
        xmin_list = f['image/detection/bbox/xmin'].float_list
        xmax_list = f['image/detection/bbox/xmax'].float_list
        ymin_list = f['image/detection/bbox/ymin'].float_list
        ymax_list = f['image/detection/bbox/ymax'].float_list
        detections = []
        for score, xmin, xmax, ymin, ymax in zip(scores.value, xmin_list.value, xmax_list.value, ymin_list.value, ymax_list.value):
            detections.append((score, xmin, xmax, ymin, ymax))
        yield filename, detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', action='store', required=True,
                        help='input tf record containing the predictions')
    parser.add_argument('--data_dir', dest='data_dir', action='store', required=True, help='')
    parser.add_argument('--output_dir', dest='output_dir', action='store', required=True, help='')
    args = parser.parse_args()

    tfrecord = args.input
    for i in items(tfrecord):
        filename, bboxes = i
        print(filename)





    # input_csv = args.input
    # data_dir = args.data_dir
    # output_dir = args.output_dir
    # ensure_dir(output_dir)
    #
    # predictions = defaultdict(list)
    # with open(input_csv, 'r') as csvfile:
    #     spamreader = csv.reader(csvfile, delimiter=' ')
    #     for row in spamreader:
    #         match = next(re.finditer(FILENAME_RE, row[0]))
    #         _id = match.group(1)
    #         score = float(row[1])
    #         xmin = float(row[2])
    #         ymin = float(row[3])
    #         xmax = float(row[4])
    #         ymax = float(row[5])
    #         # left_x top_y width height detection_score
    #         predictions[_id].append((xmin, ymin, xmax - xmin, ymax - ymin, score))
    #
    # ds = Wider(data_dir)
    #
    # for sample in ds.val_set():
    #     img_filename = '{}/{}.txt'.format(sample.category_dir(), sample.filename)
    #     target_file = os.path.join(output_dir, img_filename)
    #     ensure_dir(target_file)
    #     with open(target_file, 'w') as dest:
    #         dest.write('{}\n'.format(img_filename))
    #         if img_filename in predictions:
    #             pred = predictions[img_filename]
    #             dest.write('{}\n'.format(len(pred)))
    #             for p in pred:
    #                 dest.write('{} {} {} {} {}\n'.format(*p))
    #         else:
    #             dest.write('{}\n'.format(0))


if __name__ == '__main__':
    main()
