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
import logging
import os
import re
import sys

from morghulis.os_utils import ensure_dir
from morghulis.tf_utils import read_detections_from
from morghulis.widerface import Wider

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

FILENAME_RE = re.compile(r'.*/\d+--\w+/(.*)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', action='store', required=True,
                        help='input tf record containing the predictions')
    parser.add_argument('--data_dir', dest='data_dir', action='store', required=True, help='')
    parser.add_argument('--output_dir', dest='output_dir', action='store', required=True, help='')
    args = parser.parse_args()

    tfrecord = args.input
    data_dir = args.data_dir
    output_dir = args.output_dir
    ds = Wider(data_dir)
    events = ds.events()

    for i in read_detections_from(tfrecord):
        event_id = filename.split('_')[0]
        event = events[event_id]
        item_name = os.path.splitext(os.path.basename(filename))[0]
        result_filename = '{}/{}.txt'.format(event, item_name)
        target_file = os.path.join(output_dir, result_filename)
        ensure_dir(target_file)
        with open(target_file, 'w') as dest:
            dest.write('{}\n'.format(item_name))
            total = len(bboxes)
            dest.write('{}\n'.format(total))
            for bbox in bboxes:
                dest.write('{} {} {} {} {}\n'.format(*bbox))

if __name__ == '__main__':
    main()
