# -*- coding: utf-8 -*-
"""
face.data
classes= 1
train  = /home/user/widerface/trainval.txt
valid  = /home/user/widerface/test.txt
names = face.names
backup = backup/

face.names


images/
annotations/

"""

import os
import logging

from . import ensure_dir

log = logging.getLogger(__name__)


classes = ["face"]


class DarknetExporter:

    def __init__(self, wf):
        self.widerface = wf

    @staticmethod
    def _convert(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        cx, cy = box.center
        x = cx * dw
        w = box.w * dw
        y = cy * dh
        h = box.h * dh
        return x, y, w, h

    def _export(self, target_dir, dataset_name='train'):
        log.info('Converting %s data', dataset_name)
        images_root = os.path.join(target_dir, 'images/')
        annotations_root = images_root
        ensure_dir(annotations_root)
        with open(os.path.join(target_dir, '{}.txt'.format(dataset_name)), 'w') as f:
            for i in getattr(self.widerface, '{}_set'.format(dataset_name))():
                path = i.copy_to(images_root)
                f.write('{}\n'.format(path))
                head, _ = os.path.splitext(path)
                head, tail = os.path.split(head)
                annotation_file = os.path.join(annotations_root, tail+'.txt')
                ensure_dir(annotation_file)
                with open(annotation_file, 'w') as anno:
                    for face in i.faces:
                        bbox = self._convert(i.size, face)
                        anno.write('0 ' + ' '.join([str(a) for a in bbox]) + '\n')

    def _prepare(self, target_dir):
        log.info('Preparing target dir: %s', target_dir)
        ensure_dir(os.path.join(target_dir, 'images/'))

        log.info('Creating obj.names')
        with open(os.path.join(target_dir, 'obj.names'), 'w') as obj_names:
            obj_names.write('face\n')

        log.info('Creating obj.data')
        with open(os.path.join(target_dir, 'obj.data'), 'w') as obj_data:
            obj_data.write('classes = 1\n')
            obj_data.write('train = data/train.txt\n')
            obj_data.write('valid = data/val.txt\n')
            obj_data.write('names = data/obj.names\n')
            obj_data.write('backup = backup/\n')

    def export(self, target_dir):
        ensure_dir(target_dir)
        self._prepare(target_dir)
        self._export(target_dir, 'train')
        self._export(target_dir, 'val')
