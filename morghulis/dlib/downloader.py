import logging

from morghulis.downloader import BaseDownloader
from morghulis.os_utils import ensure_dir

log = logging.getLogger(__name__)


TRAINVAL_DATA = 'http://dlib.net/files/data/dlib_face_detection_dataset-2016-09-30.tar.gz'


class PascalFacesDownloader(BaseDownloader):

    def __init__(self, target_dir):
        super(PascalFacesDownloader, self).__init__(target_dir)

    def download(self):
        ensure_dir(self.target_dir)
        log.info('downloading the dlib face detection dataset ...')
        train_zip = self.download_file_from_web_server(TRAINVAL_DATA, self.target_dir)
        self.extract_zip_file(train_zip, self.target_dir)
        log.info('done')
