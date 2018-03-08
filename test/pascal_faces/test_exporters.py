import logging
import os
import sys
import unittest
from shutil import rmtree

from morghulis.pascal_faces import PascalFaces
from morghulis.widerface import Wider
from morghulis.widerface.coco_exporter import CocoExporter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

DS_DIR = os.path.dirname(__file__) + '/PASCAL_sample/'
TMP_DIR = '/opt/project/.tmp/pascal_faces/'
TMP_DIRS = [
    os.path.join(TMP_DIR, 'coco/'),
    os.path.join(TMP_DIR, 'tensorflow/'),
]


class CocoExporterTests(unittest.TestCase):

    def setUp(self):
        rmtree(TMP_DIRS[0], ignore_errors=True)
        self.ds = PascalFaces(DS_DIR)

    def test_coco(self):
        self.ds.export(TMP_DIRS[0], target_format='coco')


class TfExporterTests(unittest.TestCase):

    def setUp(self):
        rmtree(TMP_DIRS[1], ignore_errors=True)
        self.ds = PascalFaces(DS_DIR)

    def test_tf(self):
        self.ds.export(TMP_DIRS[1], target_format='tensorflow')