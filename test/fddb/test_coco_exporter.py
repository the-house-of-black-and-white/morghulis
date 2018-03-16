import logging
import os
import sys
import unittest
from shutil import rmtree

from morghulis.fddb import FDDB

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

FDDB_DIR = os.path.dirname(__file__) + '/FDDB_sample/'
TMP_DIR = '/opt/project/.tmp/fddb/coco/'


class CocoExporterTests(unittest.TestCase):

    def setUp(self):
        rmtree(TMP_DIR, ignore_errors=True)
        self.ds = FDDB(FDDB_DIR)

    def test_sanity(self):
        self.ds.export(TMP_DIR, 'coco')
