import logging
import os
import sys
import unittest
from shutil import rmtree

from morghulis.afw import AFW

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

AFW_DIR = os.path.dirname(__file__) + '/AFW_sample/'
TMP_DIR = '/opt/project/.tmp/afw/coco/'


class CocoExporterTests(unittest.TestCase):

    def setUp(self):
        rmtree(TMP_DIR, ignore_errors=True)
        self.afw = AFW(AFW_DIR)

    def test_sanity(self):
        self.afw.export(TMP_DIR, 'coco')

