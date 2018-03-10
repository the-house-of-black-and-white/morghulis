#!/usr/bin/env python

r"""
Example usage:
    python download_dataset.py \
        --dataset=widerface \
        --output_dir=/home/user/widerface/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import re
import sys
from collections import defaultdict

from morghulis.fddb import FDDB
from morghulis.os_utils import ensure_dir
from morghulis.tf_utils import read_detections_from

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

FILENAME_RE = re.compile(r'.*(\d{4}\/\d{2}\/\d{2}\/big\/img_\d+).*')


def extract_predictions_from_tf_record(tf_record):
    predictions = defaultdict(list)
    for filename, detections in read_detections_from(tf_record):
        match = next(re.finditer(FILENAME_RE, filename))
        _id = match.group(1)
        for d in detections:
            predictions[_id].append(d)
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', action='store', required=True,
                        help='input tfrecord containing the predictions')
    parser.add_argument('--data_dir', dest='data_dir', action='store', required=True, help='')
    parser.add_argument('--output_dir', dest='output_dir', action='store', required=True, help='')
    args = parser.parse_args()
    input_file = args.input
    data_dir = args.data_dir
    output_dir = args.output_dir
    ensure_dir(output_dir)

    predictions = extract_predictions_from_tf_record(input_file)
    ds = FDDB(data_dir)
    for fold_id, fold_file in ds.folds():
        target_file = os.path.join(output_dir, 'fold-{}-out.txt'.format(fold_id))

        with open(fold_file, 'r') as src, open(target_file, 'w') as dest:
            for img_filename in src:
                img_filename = img_filename.strip()
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
