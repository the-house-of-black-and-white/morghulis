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

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', action='store', required=True, help='widerface or fddb')
    parser.add_argument('--output_dir', dest='output_dir', action='store', required=True, help='')
    args = parser.parse_args()
    dataset = args.dataset
    output_dir = args.output_dir

    if dataset == 'widerface':
        from wider.widerface import Wider
        ds = Wider(output_dir)
    elif dataset == 'fddb':
        from wider.fddb import FDDB
        ds = FDDB(output_dir)
    else:
        logging.error('Invalid dataset name %s', dataset)
        raise ValueError('Invalid dataset name %s' % dataset)

    ds.download()


if __name__ == '__main__':
    main()
