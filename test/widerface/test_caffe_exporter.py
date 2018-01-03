import logging
import sys
import unittest
from shutil import rmtree

import os
import numpy as np
import lmdb
import caffe

from morghulis.widerface import Wider
from morghulis.widerface.caffe_exporter import CaffeExporter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

WIDER_DIR = os.path.dirname(__file__) + '/WIDERFACE_sample/'
TMP_DIR = '/opt/project/.tmp/widerface/caffe/'


class CaffeExporterTests(unittest.TestCase):

    def setUp(self):
        rmtree(TMP_DIR, ignore_errors=True)
        self.wider = Wider(WIDER_DIR)
        self.caffeExporter = CaffeExporter(self.wider)

    def test_sanity(self):
        self.caffeExporter.export(TMP_DIR)
        env = lmdb.open(os.path.join(TMP_DIR, 'train'), readonly=True)
        with env.begin() as txn:
            cursor = txn.cursor()
            count = 0
            for key, value in cursor:
                datum = caffe.proto.caffe_pb2.AnnotatedDatum()
                datum.ParseFromString(value)
                count = count + 1
                print(datum.type)

            self.assertEqual(6, count)

    # def test_faces(self):
    #     soldier_drilling = [image for image in self.wider.train_set() if 'Soldier_Drilling' in image.filename]
    #     image = soldier_drilling[0]
    #     self.assertEqual(4, len(image.faces))
