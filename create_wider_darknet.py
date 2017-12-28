#!/usr/bin/env python

r"""Convert raw WIDER dataset to Darknet format for object_detection.
Example usage:
    python create_wider_darknet.py \
        --data_dir=/home/user/wider \
        --output_dir=/home/user/wider/tf_records
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import logging
import argparse

from wider.widerface import Wider
from wider.widerface.darknet_exporter import DarknetExporter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', action='store', required=True, help='')
    parser.add_argument('--output_dir', dest='output_dir', action='store', required=True, help='')
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    wider = Wider(data_dir)
    exporter = DarknetExporter(wider)
    exporter.export(output_dir)


if __name__ == '__main__':
    main()
