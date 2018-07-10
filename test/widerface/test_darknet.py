import logging
import os
import sys
import unittest
from shutil import rmtree

from morghulis.widerface import Wider

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

WIDER_DIR = os.path.dirname(__file__) + '/WIDERFACE_sample/'
TMP_DIR = '/opt/project/.tmp/widerface/darknet/'


class DarknetTests(unittest.TestCase):

    def setUp(self):
        rmtree(TMP_DIR, ignore_errors=True)
        self.wider = Wider(WIDER_DIR)

    def test_sanity(self):
        self.wider.export(TMP_DIR, 'darknet')
