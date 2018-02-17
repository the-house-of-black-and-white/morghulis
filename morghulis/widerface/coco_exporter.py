# coding=utf-8

import logging
import os

from morghulis.exporters.coco import BaseCocoExporter
from morghulis.os_utils import ensure_dir

log = logging.getLogger(__name__)


class CocoExporter(BaseCocoExporter):
    def __init__(self, dataset):
        BaseCocoExporter.__init__(self, dataset)

    def export(self, target_dir):
        ensure_dir(target_dir)
        target_file = '{}_{}.json'
        output_filename = os.path.join(target_dir, target_file.format(self.dataset.name, 'train'))
        self._export(output_filename, self.dataset.train_set())
        output_filename = os.path.join(target_dir, target_file.format(self.dataset.name, 'val'))
        self._export(output_filename, self.dataset.val_set())
