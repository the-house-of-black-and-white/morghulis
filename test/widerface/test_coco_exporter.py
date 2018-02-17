import logging
import os
import sys
import unittest
from shutil import rmtree

from morghulis.widerface import Wider
from morghulis.widerface.coco_exporter import CocoExporter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

WIDER_DIR = os.path.dirname(__file__) + '/WIDERFACE_sample/'
TMP_DIR = '/opt/project/.tmp/widerface/coco/'


class CocoExporterTests(unittest.TestCase):

    def setUp(self):
        rmtree(TMP_DIR, ignore_errors=True)
        self.wider = Wider(WIDER_DIR)
        self.exporter = CocoExporter(self.wider)

    def test_sanity(self):
        self.exporter.export(TMP_DIR)

    # def test_faces(self):
    #     soldier_drilling = [image for image in self.wider.train_set() if 'Soldier_Drilling' in image.filename]
    #     image = soldier_drilling[0]
    #     self.assertEqual(4, len(image.faces))
