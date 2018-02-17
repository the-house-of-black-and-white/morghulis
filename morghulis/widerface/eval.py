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

from . import Wider
from morghulis.os_utils import ensure_dir

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

FILENAME_RE = re.compile(r'.*/\d+--\w+/(.*)')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', action='store', required=True,
                        help='input csv containing the predictions')
    parser.add_argument('--data_dir', dest='data_dir', action='store', required=True, help='')
    parser.add_argument('--output_dir', dest='output_dir', action='store', required=True, help='')
    args = parser.parse_args()
    input_csv = args.input
    data_dir = args.data_dir
    output_dir = args.output_dir
    ensure_dir(output_dir)

    predictions = defaultdict(list)
    with open(input_csv, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        for row in spamreader:
            match = next(re.finditer(FILENAME_RE, row[0]))
            _id = match.group(1)
            score = float(row[1])
            xmin = float(row[2])
            ymin = float(row[3])
            xmax = float(row[4])
            ymax = float(row[5])
            # left_x top_y width height detection_score
            predictions[_id].append((xmin, ymin, xmax - xmin, ymax - ymin, score))

    ds = Wider(data_dir)

    for sample in ds.val_set():
        img_filename = '{}/{}.txt'.format(sample.category_dir(), sample.filename)
        target_file = os.path.join(output_dir, img_filename)
        ensure_dir(target_file)
        with open(target_file, 'w') as dest:
            dest.write('{}\n'.format(img_filename))
            if img_filename in predictions:
                pred = predictions[img_filename]
                dest.write('{}\n'.format(len(pred)))
                for p in pred:
                    dest.write('{} {} {} {} {}\n'.format(*p))
            else:
                dest.write('{}\n'.format(0))


if __name__ == '__main__':
    main()
