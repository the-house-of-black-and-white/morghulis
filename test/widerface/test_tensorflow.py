import logging
import sys
import unittest
from shutil import rmtree

from wider.widerface import Wider
from wider.widerface.tensorflow_exporter import TensorflowExporter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

WIDER_DIR = 'sample/'
TMP_DIR = '/opt/project/.tmp/tensorflow/'


class TensorflowExporterTests(unittest.TestCase):

    def setUp(self):
        rmtree(TMP_DIR, ignore_errors=True)
        self.wider = Wider(WIDER_DIR)
        self.tfExporter = TensorflowExporter(self.wider)

    def test_sanity(self):
        self.tfExporter.export(TMP_DIR)
