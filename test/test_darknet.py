import logging
import sys
import unittest

from wider import Wider
from wider.darknet_exporter import DarknetExporter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

WIDER_DIR = 'sample/'


class DarknetTests(unittest.TestCase):

    def setUp(self):
        self.wider = Wider(WIDER_DIR)
        self.darknetExporter = DarknetExporter(self.wider)

    def test_sanity(self):
        self.darknetExporter.export('/opt/project/.tmp/darknet/')


# if __name__ == '__main__':
#     unittest.main()