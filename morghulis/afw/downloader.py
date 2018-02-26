import logging
import os

from morghulis.downloader import BaseDownloader
from morghulis.os_utils import ensure_dir

log = logging.getLogger(__name__)

IMAGES_URL = 'https://www.ics.uci.edu/~xzhu/face/AFW.zip'


class AFWDownloader(BaseDownloader):

    def __init__(self, target_dir):
        super(AFWDownloader, self).__init__(target_dir)

    def download(self):
        ensure_dir(self.target_dir)
        log.info('Downloading dataset...')
        zip_file = self.download_file_from_web_server(IMAGES_URL, self.target_dir)
        log.info('Finished download. Unzipping...')
        self.extract_zip_file(os.path.join(self.target_dir, zip_file), self.target_dir)
        log.info('done')
