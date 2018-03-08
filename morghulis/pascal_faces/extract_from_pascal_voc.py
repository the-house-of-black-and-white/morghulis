#!/usr/bin/env python

r"""Extract Pascal faces from the full Pascal VOC
Example usage:
    python -m morghulis.pascal_faces.extract_from_pascal_voc \
        --data_dir /datasets/pascal_voc/ \
        --output_dir /datasets/pascal_faces/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import logging
import argparse
import shutil

import os

from morghulis.os_utils import ensure_dirs

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

log = logging.getLogger(__name__)


def _get_dirs(root_dir):
    root_dir = os.path.join(root_dir, 'VOCdevkit', 'VOC2012/')
    images_dir = os.path.join(root_dir, 'JPEGImages/')
    annotations_dir = os.path.join(root_dir, 'Annotations/')
    layout_dir = os.path.join(root_dir, 'ImageSets', 'Layout/')
    train_gt = os.path.join(layout_dir, 'train.txt')
    val_gt = os.path.join(layout_dir, 'val.txt')
    trainval_gt = os.path.join(layout_dir, 'trainval.txt')
    return root_dir, images_dir, annotations_dir, layout_dir, train_gt, val_gt, trainval_gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', action='store', required=True, help='')
    parser.add_argument('--output_dir', dest='output_dir', action='store', required=True, help='')
    args = parser.parse_args()

    output_dir = args.output_dir
    input_dir = args.data_dir

    root_dir, images_dir, annotations_dir, layout_dir, train_gt, val_gt, trainval_gt = _get_dirs(input_dir)
    root_dir_out, images_dir_out, annotations_dir_out, layout_dir_out, train_gt_out, val_gt_out, trainval_gt_out = _get_dirs(output_dir)

    log.info('Creating target directory structure')
    ensure_dirs([images_dir_out, annotations_dir_out,layout_dir_out])

    log.info('Copying person layout txt files')
    shutil.copy2(train_gt, train_gt_out)
    shutil.copy2(val_gt, val_gt_out)
    shutil.copy2(trainval_gt, trainval_gt_out)

    log.info('Copying annotations and images')
    with(open(trainval_gt, 'r')) as trainval:
        for line in trainval:
            image_id = line.strip().split(' ')[0]
            shutil.copy2(os.path.join(annotations_dir, image_id + '.xml'), annotations_dir_out)
            shutil.copy2(os.path.join(images_dir, image_id + '.jpg'), images_dir_out)

    log.info('Packaging')
    shutil.make_archive(os.path.join(output_dir, 'PASCAL_faces_trainval_2012'), format='gztar', root_dir=output_dir, base_dir='VOCdevkit', logger=log)


if __name__ == '__main__':
    main()
