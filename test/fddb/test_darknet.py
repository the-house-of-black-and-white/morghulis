import logging
import sys
import unittest
from shutil import rmtree

import os

from morghulis.fddb import FDDB
from morghulis.fddb.darknet_exporter import DarknetExporter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

FDDB_DIR = os.path.dirname(__file__) + '/FDDB_sample/'
TMP_DIR = '/opt/project/.tmp/fddb/darknet'


class FDDBDarknetTests(unittest.TestCase):

    def setUp(self):
        rmtree(TMP_DIR, ignore_errors=True)
        self.fddb = FDDB(FDDB_DIR)
        self.darknetExporter = DarknetExporter(self.fddb)

    def test_sanity(self):
        self.darknetExporter.export(TMP_DIR)
