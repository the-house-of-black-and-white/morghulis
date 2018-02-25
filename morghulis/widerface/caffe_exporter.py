# -*- coding: utf-8 -*-
import hashlib
import logging
import os

import caffe
import lmdb
import numpy as np

from morghulis.os_utils import ensure_dir

log = logging.getLogger(__name__)


class CaffeExporter:

    def __init__(self, wf):
        self.widerface = wf

    def _convert(self, image):

        img_data = image.image
        pix = np.array(img_data)
        key = hashlib.sha256(pix.tobytes()).hexdigest()
        width = img_data.width
        height = img_data.height

        annotated_datum = caffe.proto.caffe_pb2.AnnotatedDatum()
        datum = annotated_datum.datum
        datum.channels = 3
        datum.height = height
        datum.width = width
        datum.data = pix.tobytes()
        datum.label = 0

        annotation_group = annotated_datum.annotation_group.add()
        annotation_group.group_label = 0
        for face in image.faces:
            annotation = annotation_group.annotation.add()
            bbox = annotation.bbox
            bbox.xmin = face.x1 / width
            bbox.ymin = face.y1 / height
            bbox.xmax = face.x2 / width
            bbox.ymax = face.y2 / height
            bbox.label = 0
            bbox.difficult = False
            # float score = 7;
            # float size = 8;
        return key, annotated_datum

    def _export(self, target_dir, dataset_name='train', map_size=1024 * 1024 * 1024 * 1024):
        log.info('Converting %s data', dataset_name)
        env = lmdb.open(os.path.join(target_dir, dataset_name), map_size=map_size)
        count = 0
        txn = lmdb.Transaction(env, write=True)
        for image in getattr(self.widerface, '{}_set'.format(dataset_name))():
            key, datum = self._convert(image)
            txn.put(key, datum.SerializeToString())
            if count % 1000 == 0:
                #Commit db
                txn.commit()
                txn = lmdb.Transaction(env, write=True)
                log.info('Processed %s files.', count)
            count += 1

        #write the last batch
        if count % 1000 != 0:
            txn.commit()
            log.info('Processed %s files.', count)

    @staticmethod
    def _prepare(target_dir):
        log.info('Preparing target dir: %s', target_dir)

    def export(self, target_dir):
        ensure_dir(target_dir)
        self._prepare(target_dir)
        self._export(target_dir, 'train')
        self._export(target_dir, 'val')
