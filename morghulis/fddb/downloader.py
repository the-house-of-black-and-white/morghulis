import logging
import os

from morghulis.downloader import BaseDownloader
from morghulis.os_utils import ensure_dir

log = logging.getLogger(__name__)

IMAGES_URL = 'http://tamaraberg.com/faceDataset/originalPics.tar.gz'
ANNOTATIONS_URL = 'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'


class FddbDownloader(BaseDownloader):

    def __init__(self, target_dir):
        super(FddbDownloader, self).__init__(target_dir)

    def download(self):
        ensure_dir(self.target_dir)

        log.info('Downloading images')
        annotation_zip_file = self.download_file_from_web_server(IMAGES_URL, self.target_dir)
        log.info('Finished download. Extracting.')
        self.extract_tar_file(os.path.join(self.target_dir, annotation_zip_file), os.path.join(self.target_dir, 'originalPics/'))

        log.info('Downloading annotations')
        annotation_zip_file = self.download_file_from_web_server(ANNOTATIONS_URL, self.target_dir)
        log.info('Finished download. Extracting.')
        self.extract_tar_file(os.path.join(self.target_dir, annotation_zip_file), self.target_dir)

        log.info('done')

