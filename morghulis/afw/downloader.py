import logging
import os
import zipfile

import requests

from morghulis.os_utils import ensure_dir

log = logging.getLogger(__name__)

IMAGES_URL = 'https://www.ics.uci.edu/~xzhu/face/AFW.zip'


class AFWDownloader:

    def __init__(self, target_dir):
        self.target_dir = target_dir

    def download(self):
        ensure_dir(self.target_dir)
        log.info('Downloading dataset...')
        zip_file = self.download_file_from_web_server(IMAGES_URL, self.target_dir)
        log.info('Finished download. Unzipping...')
        self.extract_zip_file(os.path.join(self.target_dir, zip_file), self.target_dir)
        log.info('done')

    def download_file_from_web_server(self, url, destination):
        local_filename = url.split('/')[-1]
        target_file = os.path.join(destination, local_filename)
        if not os.path.exists(target_file):
            response = requests.get(url, stream=True)
            self.save_response_content(response, target_file)
        return local_filename

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
