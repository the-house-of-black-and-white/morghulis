import logging
import sys
import unittest

import os

from morghulis.os_utils import ensure_dir
from morghulis.pascal_faces import PascalFaces

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

DS_DIR = os.path.dirname(__file__) + '/PASCAL_sample/'
TMP_DIR = '/opt/project/.tmp/'
ensure_dir(TMP_DIR)


class WiderTests(unittest.TestCase):

    def setUp(self):
        self.ds = PascalFaces(DS_DIR)

    def test_images(self):
        train_set = [image for image in self.ds.images()]
        self.assertEqual(7, len(train_set))

