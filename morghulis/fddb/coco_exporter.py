# coding=utf-8

import logging

from morghulis.exporters.coco import BaseCocoExporter

log = logging.getLogger(__name__)


class CocoExporter(BaseCocoExporter):
    def __init__(self, dataset):
        BaseCocoExporter.__init__(self, dataset)
