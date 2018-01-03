import logging

import requests
import os
import tarfile

from morghulis.os_utils import ensure_dir


log = logging.getLogger(__name__)

IMAGES_URL = 'http://tamaraberg.com/faceDataset/originalPics.tar.gz'
ANNOTATIONS_URL = 'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'


class FddbDownloader:

    def __init__(self, target_dir):
        self.target_dir = target_dir

    def download(self):
        ensure_dir(self.target_dir)

        log.info('Downloading images')
        annotation_zip_file = self.download_file_from_web_server(IMAGES_URL, self.target_dir)
        log.info('Finished download. Extracting.')
        self.extract_zip_file(os.path.join(self.target_dir, annotation_zip_file), self.target_dir)

        log.info('Downloading annotations')
        annotation_zip_file = self.download_file_from_web_server(ANNOTATIONS_URL, self.target_dir)
        log.info('Finished download. Extracting.')
        self.extract_zip_file(os.path.join(self.target_dir, annotation_zip_file), self.target_dir)

        log.info('done')

    def download_file_from_web_server(self, url, destination):
        local_filename = url.split('/')[-1]
        response = requests.get(url, stream=True)
        self.save_response_content(response, os.path.join(destination, local_filename))
        return local_filename

    def save_response_content(self, response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    def extract_zip_file(self, zip_file_name, destination):
        zip_ref = tarfile.TarFile(zip_file_name, 'r')
        zip_ref.extractall(destination)
        zip_ref.close()
