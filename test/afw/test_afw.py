import logging
import sys
import unittest

import os

from morghulis import AFW, ensure_dir

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger(__name__)

AFW_DIR = os.path.dirname(__file__) + '/AFW_sample/'
TMP_DIR = '/opt/project/.tmp/'
ensure_dir(TMP_DIR)


class AFWTests(unittest.TestCase):

    def setUp(self):
        self.afw = AFW(AFW_DIR)

    def test_train_set(self):
        for i in self.afw.images():
            log.debug(i)
            for f in i.faces:
                log.debug(f)

    @unittest.skip("skipping because it takes too long")
    def test_download(self):
        self.afw = AFW(os.path.join(TMP_DIR, 'afw_download/'))
        self.afw.download()

