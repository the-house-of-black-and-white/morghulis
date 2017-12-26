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
        x = box.x1 * dw
        w = box.w * dw
        y = box.y1 * dh
        h = box.h * dh
        return x, y, w, h

    def _export(self, target_dir, dataset_name='train'):
        annotations_root = os.path.join(getattr(self.widerface, '{}_dir'.format(dataset_name)), 'annotations')
        ensure_dir(annotations_root)
        with open(os.path.join(target_dir, '{}.txt'.format(dataset_name)), 'w') as f:
            for i in getattr(self.widerface, '{}_set'.format(dataset_name))():
                f.write('{}\n'.format(i.path))
                head, _ = os.path.splitext(i.path)
                head, tail = os.path.split(head)
                annotation_file = os.path.join(annotations_root, os.path.basename(head), tail+'.txt')
                ensure_dir(annotation_file)
                with open(annotation_file, 'w') as anno:
                    for face in i.faces:
                        bbox = self._convert(i.size, face)
                        anno.write('0 ' + ' '.join([str(a) for a in bbox]) + '\n')

    def export(self, target_dir):
        self._export(target_dir, 'train')
        self._export(target_dir, 'val')
