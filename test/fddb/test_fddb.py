import logging
import sys
import unittest

import os

from morghulis import FDDB, ensure_dir

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

FDDB_DIR = os.path.dirname(__file__) + '/FDDB_sample/'
TMP_DIR = '/opt/project/.tmp/'
ensure_dir(TMP_DIR)


class FDDBTests(unittest.TestCase):

    def setUp(self):
        self.fddb = FDDB(FDDB_DIR)

    def _get_image(self, img='2002/08/01/big/img_1468'):
        return [image for image in self.fddb.images() if img in image.filename][0]

    def test_train_set(self):
        train_set = [image for image in self.fddb.images()]
        self.assertEqual(7, len(train_set))

    def test_faces(self):
        image = self._get_image()
        self.assertEqual(5, len(image.faces))

    def test_face_details(self):
        image = self._get_image()
        self.assertEqual(5, len(image.faces))
        face = image.faces[0]
        self.assertEqual((7.294545, 96.341818), face.center)
        self.assertEqual(16.087936, face.w)
        self.assertEqual(19.110234, face.h)
        self.assertEqual(-1.469174, face.angle)
        self.assertEqual(9.555117, face.major_axis_radius)
        self.assertEqual(8.043968, face.minor_axis_radius)
        self.assertEqual(0, face.invalid)

    def test_image(self):
        image = self._get_image()
        self.assertEqual(334, image.width)
        self.assertEqual(450, image.height)
        self.assertEqual('JPEG', image.format)

    @unittest.skip("skipping because it takes too long")
    def test_download(self):
        self.fddb = FDDB(os.path.join(TMP_DIR, 'fddb_download/'))
        self.fddb.download()
