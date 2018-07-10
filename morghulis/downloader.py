import logging
import re
import json
import os
import tarfile
import zipfile
from abc import ABCMeta, abstractmethod

import requests

log = logging.getLogger(__name__)

CHUNK_SIZE = 32768


class BaseDownloader:
    __metaclass__ = ABCMeta

    def __init__(self, target_dir):
        self.target_dir = target_dir

    @abstractmethod
    def download(self):
        pass

    def download_file_from_web_server(self, url, destination):
        log.info('Downloading {} into {}'.format(url, destination))
        local_filename = url.split('/')[-1]
        target_file = os.path.join(destination, local_filename)
        if not os.path.exists(target_file):
            response = requests.get(url, stream=True)
            self.save_response_content(response, target_file)
        log.info('Finished download')
        return local_filename

    def download_file_from_google_drive(self, id, destination):
        log.info('Downloading from Google Drive id={} into {}'.format(id, destination))
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': id}, stream=True)
        token = self.get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        self.save_response_content(response, destination)

        log.info('Finished download')

    def download_file_from_baidu(self, id, destination):
        log.info('Downloading from Baidu id={} into {}'.format(id, destination))
        session = requests.Session()
        HEADERS = {
            "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1700.77 Safari/537.36"
        }
        response = session.get('https://pan.baidu.com/s/{}'.format(id), headers=HEADERS)
        response.raise_for_status()
        regex = r".*yunData\.setData\((?P<info>\{.*\})\)"
        m = next(re.finditer(regex, response.text, re.MULTILINE))
        info = json.loads(m.group('info'))
        log.debug(json.dumps(info, indent=4))
        qs = dict(
            uk=info['uk'],
            shareid=info['shareid'],
            timestamp=info['timestamp'],
            sign=info['sign']
        )

        download_link = 'http://pan.baidu.com/share/download?channel=chunlei&clienttype=0&web=1&uk={uk}&shareid={shareid}&timestamp={timestamp}&sign={sign}'
        fs_id = info['file_list']['list'][0]['fs_id']
        data = dict(
            fid_list='["{}"]'.format(fs_id)
        )
        log.debug(download_link.format(**qs))
        response = session.post(download_link.format(**qs), data=data, headers=HEADERS)
        response.raise_for_status()
        download_info = response.json()
        log.debug(download_info)
        if 'dlink' in download_info:
            dlink_ = download_info['dlink']
            log.debug('Download link: {}'.format(dlink_))
            response = session.get(dlink_, stream=True)
            self.save_response_content(response, destination)
            log.info('Finished download')
        else:
            log.warning('Could not download. Try again later.')

    #  TODO Add progress bar
    @staticmethod
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def save_response_content(response, destination):
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    @staticmethod
    def extract_zip_file(zip_file_name, destination):
        log.info('Extracting {} into {}'.format(zip_file_name, destination))
        zip_ref = zipfile.ZipFile(zip_file_name, 'r')
        zip_ref.extractall(destination)
        zip_ref.close()

    @staticmethod
    def extract_tar_file(zip_file_name, destination):
        log.info('Extracting {} into {}'.format(zip_file_name, destination))
        zip_ref = tarfile.TarFile.open(zip_file_name, 'r')
        zip_ref.extractall(destination)
        zip_ref.close()
