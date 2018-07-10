#!/usr/bin/env python

r"""Browse the given dataset
Example usage:
    python browse.py \
        --dataset=widerface \
        --data_dir=/home/user/widerface/
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
    parser.add_argument('--dataset', dest='dataset', action='store', required=True, help='widerface, fddb, afw, mafa')
    parser.add_argument('--data_dir', dest='data_dir', action='store', required=True, help='')
    args = parser.parse_args()
    dataset = args.dataset
    data_dir = args.data_dir
    ds = create_dataset(dataset, data_dir)
    ds.browse()


if __name__ == '__main__':
    main()
