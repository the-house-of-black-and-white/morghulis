import logging
import sys
import unittest

from wider import Wider, ensure_dir

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

WIDER_DIR = 'sample/'
TMP_DIR = '/opt/project/.tmp/'
ensure_dir(TMP_DIR)


class WiderTests(unittest.TestCase):

    def setUp(self):
        self.wider = Wider(WIDER_DIR)

    def test_train_set(self):
        train_set = [image for image in self.wider.train_set()]
        self.assertEqual(6, len(train_set))

    def test_val_set(self):
        val_set = [image for image in self.wider.val_set()]
        self.assertEqual(4, len(val_set))

    def test_faces(self):
        soldier_drilling = [image for image in self.wider.train_set() if 'Soldier_Drilling' in image.filename]
        image = soldier_drilling[0]
        self.assertEqual(4, len(image.faces))

    def test_faces(self):
        press_conference = [image for image in self.wider.train_set() if 'Press_Conference' in image.filename]
        image = press_conference[0]
        self.assertEqual(1, len(image.faces))
        face = image.faces[0]
        self.assertEqual(400, face.x1)
        self.assertEqual(150, face.y1)
        self.assertEqual(208, face.w)
        self.assertEqual(290, face.h)
        self.assertEqual(0, face.blur)
        self.assertEqual(1, face.expression)
        self.assertEqual(2, face.illumination)
        self.assertEqual(3, face.invalid)
        self.assertEqual(4, face.occlusion)
        self.assertEqual(5, face.pose)

    def test_image(self):
        soldier_drilling = [image for image in self.wider.train_set() if 'Soldier_Drilling' in image.filename]
        image = soldier_drilling[0]
        self.assertEqual(1024, image.width)
        self.assertEqual(682, image.height)
        self.assertEqual('JPEG', image.format)

