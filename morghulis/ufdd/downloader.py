import logging
import os

from morghulis.downloader import BaseDownloader
from morghulis.os_utils import ensure_dir

log = logging.getLogger(__name__)


VAL_DATA = 'UFDD_val.zip', '1o-lsXB7XLc4F39zQyZgwrabWyN1M5NBY'
ANNOTATIONS_DATA = 'UFDD-annotationfile.zip', '1aGR7FryrRuS86S9LBAqFksy-QDqsgBRV'


class UFDDDownloader(BaseDownloader):

    def __init__(self, target_dir):
        super(UFDDDownloader, self).__init__(target_dir)

    def download(self):
        ensure_dir(self.target_dir)

        log.info('downloading the validation images from google drive...')
        val_zip = os.path.join(self.target_dir, VAL_DATA[0])
        self.download_file_from_google_drive(VAL_DATA[1], val_zip)
        self.extract_zip_file(val_zip, self.target_dir)

        log.info('downloading the bounding boxes annotations...')
        anno_zip = os.path.join(self.target_dir, ANNOTATIONS_DATA[0])
        self.download_file_from_google_drive(ANNOTATIONS_DATA[1], anno_zip)
        self.extract_zip_file(anno_zip, self.target_dir)

        log.info('done')
