import logging
import os

from morghulis.downloader import BaseDownloader
from morghulis.os_utils import ensure_dir

log = logging.getLogger(__name__)

TRAIN_DATA = 'WIDER_train.zip', '0B6eKvaijfFUDQUUwd21EckhUbWs'
VAL_DATA = 'WIDER_val.zip', '0B6eKvaijfFUDd3dIRmpvSk8tLUk'
ANNOTATIONS_URL = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip'
EVAL_TOOLS_URL = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip'


class WiderFaceDownloader(BaseDownloader):

    def __init__(self, target_dir):
        super(WiderFaceDownloader, self).__init__(target_dir)

    def download(self):
        ensure_dir(self.target_dir)

        log.info('downloading the training images from google drive...')
        train_zip = os.path.join(self.target_dir, TRAIN_DATA[0])
        self.download_file_from_google_drive(TRAIN_DATA[1], train_zip)
        self.extract_zip_file(train_zip, self.target_dir)

        log.info('downloading the validation images from google drive...')
        val_zip = os.path.join(self.target_dir, VAL_DATA[0])
        self.download_file_from_google_drive(VAL_DATA[1], val_zip)
        self.extract_zip_file(val_zip, self.target_dir)

        log.info('downloading the bounding boxes annotations...')
        annotation_zip_file = self.download_file_from_web_server(ANNOTATIONS_URL, self.target_dir)
        self.extract_zip_file(os.path.join(self.target_dir, annotation_zip_file), self.target_dir)

        log.info('downloading eval tools...')
        tools_zip_file = self.download_file_from_web_server(EVAL_TOOLS_URL, self.target_dir)
        self.extract_zip_file(os.path.join(self.target_dir, tools_zip_file), self.target_dir)

        log.info('done')
