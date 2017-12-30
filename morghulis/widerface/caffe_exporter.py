# -*- coding: utf-8 -*-
import logging
import lmdb
import caffe
from morghulis import ensure_dir

log = logging.getLogger(__name__)


class CaffeExporter:

    def __init__(self, wf):
        self.widerface = wf

    def _convert(self):
        annotated_datum = caffe.proto.caffe_pb2.AnnotatedDatum()

        # datum.channels = X.shape[1]
        # datum.height = X.shape[2]
        # datum.width = X.shape[3]
        # datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        # datum.label = int(y[i])


    def _export(self, target_dir, dataset_name='train'):
        pass

    @staticmethod
    def _prepare(target_dir):
        log.info('Preparing target dir: %s', target_dir)

    def export(self, target_dir):
        ensure_dir(target_dir)
        self._prepare(target_dir)
        self._export(target_dir, 'train')
        self._export(target_dir, 'val')
