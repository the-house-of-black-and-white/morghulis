import logging
import sys
import unittest
from shutil import  rmtree

from wider import Wider
from wider.darknet_exporter import DarknetExporter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

WIDER_DIR = 'sample/'
TMP_DIR = '/opt/project/.tmp/darknet/'


class DarknetTests(unittest.TestCase):

    def setUp(self):
        rmtree(TMP_DIR)
        self.wider = Wider(WIDER_DIR)
        self.darknetExporter = DarknetExporter(self.wider)

    def test_sanity(self):
        self.darknetExporter.export(TMP_DIR)
