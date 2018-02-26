import logging
import os

from morghulis.downloader import BaseDownloader
from morghulis.os_utils import ensure_dir

log = logging.getLogger(__name__)

IMAGES_URL = 'http://www.vision.caltech.edu/Image_Datasets/faces/faces.tar'


class CaltechFacesDownloader(BaseDownloader):

    def __init__(self, target_dir):
        super(CaltechFacesDownloader, self).__init__(target_dir)

    def download(self):
        ensure_dir(self.target_dir)
        tar_file = self.download_file_from_web_server(IMAGES_URL, self.target_dir)
        self.extract_tar_file(os.path.join(self.target_dir, tar_file), self.target_dir)
        log.info('done')
