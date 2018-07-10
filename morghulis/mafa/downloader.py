import logging
import os

from morghulis.downloader import BaseDownloader
from morghulis.os_utils import ensure_dir

log = logging.getLogger(__name__)

# from BAIDU
TRAIN_DATA = 'train-images.zip', '15NsUpSIFeHfoGKHAjQBISQ'
TRAIN_ANNO = 'MAFA-Label-Train.zip', '1t-Tx5lpjgrljErjmZWMOWw'

# from Google Drive
TEST_DATA = 'test-images.zip', '1jJHdmmscqxvNQ2dxKUrLaHqW3w1Yo_9S'
TEST_ANNO = 'MAFA-Label-Test.zip', '1uN0a4P0wAFwJLid_r7VHFs0KUcizIRGN'


class MafaDownloader(BaseDownloader):

    def __init__(self, target_dir):
        super(MafaDownloader, self).__init__(target_dir)

    def download(self):
        ensure_dir(self.target_dir)

        log.info('Downloading the train images from baidu...')
        log.warning('You might want to download using a better client')
        test_zip = os.path.join(self.target_dir, TRAIN_DATA[0])
        self.download_file_from_baidu(TRAIN_DATA[1], test_zip)
        self.extract_zip_file(test_zip, self.target_dir)

        log.info('downloading the train annotations from baidu...')
        test_zip = os.path.join(self.target_dir, TRAIN_ANNO[0])
        self.download_file_from_baidu(TRAIN_ANNO[1], test_zip)
        self.extract_zip_file(test_zip, self.target_dir)

        log.info('downloading the test images from google drive...')
        test_zip = os.path.join(self.target_dir, TEST_DATA[0])
        self.download_file_from_google_drive(TEST_DATA[1], test_zip)
        self.extract_zip_file(test_zip, self.target_dir)

        log.info('downloading the test annotations from google drive...')
        test_zip = os.path.join(self.target_dir, TEST_ANNO[0])
        self.download_file_from_google_drive(TEST_ANNO[1], test_zip)
        self.extract_zip_file(test_zip, self.target_dir)

        log.info('done')
