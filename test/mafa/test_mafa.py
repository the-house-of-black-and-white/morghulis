import logging
import sys
import unittest

import os

from morghulis.mafa import Mafa
from morghulis.os_utils import ensure_dir

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

TMP_DIR = '/opt/project/.tmp/'
DS_DIR = os.path.join(TMP_DIR, 'mafa_download/')
ensure_dir(TMP_DIR)


class MafaTests(unittest.TestCase):

    def setUp(self):
        self.ds = Mafa(DS_DIR)

    def test_test_set(self):
        train_set = self.ds.test_set()
        self.assertEqual(4935, len(train_set))

    def test_train_set(self):
        train_set = self.ds.train_set()
        self.assertEqual(25876, len(train_set))

    # def _get_image(self, img='_0001'):
    #     return [image for image in self.ds.images() if img in image.filename][0]
    #
    # def test_faces(self):
    #     image = self._get_image()
    #     self.assertEqual(1, len(image.faces))
    #
    # def test_face_details(self):
    #     image = self._get_image()
    #     self.assertEqual(1, len(image.faces))
    #     face = image.faces[0]
    #     self.assertEqual(435.06887379739675, face.x1)
    #     self.assertEqual(41.308602150537695, face.y1)

    @unittest.skip("skipping because it takes too long")
    def test_download(self):
        self.ds = Mafa(os.path.join(TMP_DIR, 'mafa_download/'))
        self.ds.download()
