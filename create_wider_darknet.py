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

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', action='store', required=True, help='widerface or fddb')
    parser.add_argument('--data_dir', dest='data_dir', action='store', required=True, help='')
    parser.add_argument('--output_dir', dest='output_dir', action='store', required=True, help='')
    args = parser.parse_args()
    dataset = args.dataset
    data_dir = args.data_dir
    output_dir = args.output_dir

    if dataset == 'widerface':
        from wider.widerface import Wider
        from wider.widerface.darknet_exporter import DarknetExporter
        ds = Wider(data_dir)
        exporter = DarknetExporter(ds)
        exporter.export(output_dir)
    elif dataset == 'fddb':
        from wider.fddb import FDDB
        from wider.fddb.darknet_exporter import DarknetExporter
        ds = FDDB(data_dir)
        exporter = DarknetExporter(ds)
        exporter.export(output_dir)
    else:
        logging.error('Invalid dataset name %s', dataset)


if __name__ == '__main__':
    main()
