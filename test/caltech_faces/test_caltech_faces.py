import logging
import sys
import unittest

import os

from morghulis.caltech_faces import CaltechFaces
from morghulis.os_utils import ensure_dir

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

TMP_DIR = '/opt/project/.tmp/'
DS_DIR = os.path.join(TMP_DIR, 'caltech_faces_download/')
ensure_dir(TMP_DIR)


class CaltechFacesTests(unittest.TestCase):

    def setUp(self):
        self.ds = CaltechFaces(DS_DIR)

    def test_images(self):
        train_set = [image for image in self.ds.images()]
        self.assertEqual(450, len(train_set))

    def _get_image(self, img='_0001'):
        return [image for image in self.ds.images() if img in image.filename][0]

    def test_faces(self):
        image = self._get_image()
        self.assertEqual(1, len(image.faces))

    def test_face_details(self):
        image = self._get_image()
        self.assertEqual(1, len(image.faces))
        face = image.faces[0]
        self.assertEqual(435.06887379739675, face.x1)
        self.assertEqual(41.308602150537695, face.y1)

    @unittest.skip("skipping because it takes too long")
    def test_download(self):
        self.ds = CaltechFaces(os.path.join(TMP_DIR, 'caltech_faces_download/'))
        self.ds.download()
