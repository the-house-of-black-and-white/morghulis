import logging
import os

from morghulis.downloader import BaseDownloader
from morghulis.os_utils import ensure_dir

log = logging.getLogger(__name__)


TRAINVAL_DATA = 'PASCAL_faces_trainval_2012.tar.gz', '1l181EnYeab-fwJ2JbolFyl7uIkgJzfIt'


class PascalFacesDownloader(BaseDownloader):

    def __init__(self, target_dir):
        super(PascalFacesDownloader, self).__init__(target_dir)

    def download(self):
        ensure_dir(self.target_dir)

        log.info('downloading the pascal faces from google drive...')
        train_zip = os.path.join(self.target_dir, TRAINVAL_DATA[0])
        self.download_file_from_google_drive(TRAINVAL_DATA[1], train_zip)
        self.extract_tar_file(train_zip, self.target_dir)

        log.info('done')
