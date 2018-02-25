import logging

import requests
import os
import zipfile

from morghulis.os_utils import ensure_dir


log = logging.getLogger(__name__)

TRAIN_DATA = 'WIDER_train.zip', '0B6eKvaijfFUDQUUwd21EckhUbWs'
VAL_DATA = 'WIDER_val.zip', '0B6eKvaijfFUDd3dIRmpvSk8tLUk'
ANNOTATIONS_URL = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip'
EVAL_TOOLS_URL = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip'


class WiderFaceDownloader:

    def __init__(self, target_dir):
        self.target_dir = target_dir

    def download(self):
        ensure_dir(self.target_dir)

        log.info('downloading the training images from google drive...')
        train_zip = os.path.join(self.target_dir, TRAIN_DATA[0])
        self.download_file_from_google_drive(TRAIN_DATA[1], train_zip)
        log.info('Finished download. Unziping...')
        self.extract_zip_file(train_zip, self.target_dir)

        log.info('downloading the validation images from google drive...')
        val_zip = os.path.join(self.target_dir, VAL_DATA[0])
        self.download_file_from_google_drive(VAL_DATA[1], val_zip)
        log.info('Finished download. Unziping...')
        self.extract_zip_file(val_zip, self.target_dir)

        log.info('downloading the bounding boxes annotations...')
        annotation_zip_file = self.download_file_from_web_server(ANNOTATIONS_URL, self.target_dir)
        log.info('Finished download. Unziping...')
        self.extract_zip_file(os.path.join(self.target_dir, annotation_zip_file), self.target_dir)

        log.info('downloading eval tools...')
        tools_zip_file = self.download_file_from_web_server(EVAL_TOOLS_URL, self.target_dir)
        log.info('Finished download. Unziping...')
        self.extract_zip_file(os.path.join(self.target_dir, tools_zip_file), self.target_dir)

        log.info('done')

    def download_file_from_google_drive(self, id, destination):
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': id}, stream=True)
        token = self.get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        self.save_response_content(response, destination)

    def download_file_from_web_server(self, url, destination):
        local_filename = url.split('/')[-1]
        # NOTE the stream=True parameter
        response = requests.get(url, stream=True)
        self.save_response_content(response, os.path.join(destination, local_filename))
        return local_filename

    #  TODO Add progress bar
    def get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(self, response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    def extract_zip_file(self, zip_file_name, destination):
        zip_ref = zipfile.ZipFile(zip_file_name, 'r')
        zip_ref.extractall(destination)
        zip_ref.close()
