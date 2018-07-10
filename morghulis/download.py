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

import sys
import logging
import argparse

from morghulis import create_dataset

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', action='store', required=True, help='widerface, fddb or afw')
    parser.add_argument('--output_dir', dest='output_dir', action='store', required=True, help='')
    args = parser.parse_args()
    dataset = args.dataset
    output_dir = args.output_dir
    ds = create_dataset(dataset, output_dir)
    ds.download()


if __name__ == '__main__':
    main()
