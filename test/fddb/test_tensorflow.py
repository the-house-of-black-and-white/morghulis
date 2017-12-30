import logging
import sys
import unittest
from shutil import rmtree

import os

from morghulis.fddb import FDDB
from morghulis.fddb.tensorflow_exporter import TensorflowExporter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)


FDDB_DIR = os.path.dirname(__file__) + '/FDDB_sample/'
TMP_DIR = '/opt/project/.tmp/fddb/tensorflow/'


class FDDBTensorflowTests(unittest.TestCase):

    def setUp(self):
        rmtree(TMP_DIR, ignore_errors=True)
        self.dataset = FDDB(FDDB_DIR)
        self.tfExporter = TensorflowExporter(self.dataset)

    def test_sanity(self):
        self.tfExporter.export(TMP_DIR)
