# -*- coding: utf-8 -*-
import logging

from wider import ensure_dir

log = logging.getLogger(__name__)


class CaffeExporter:

    def __init__(self, wf):
        self.widerface = wf

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
