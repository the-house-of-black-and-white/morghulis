# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

from morghulis.model import Image as BaseImage, BaseFace, BaseDataset
from morghulis.ufdd.downloader import UFDDDownloader

log = logging.getLogger(__name__)


class Image(BaseImage):
    def __init__(self, filename, raw_filename=None):
        BaseImage.__init__(self, filename, raw_filename)

    def __str__(self):
        return 'Image(filename={})'.format(self.filename)


class Face(BaseFace):
    def __init__(self, anno):
        """
        x1, y1, w, h, blur
        :param annotations:
        """
        self._x1 = float(anno[0])
        self._y1 = float(anno[1])
        self._w = float(anno[2])
        self._h = float(anno[3])

    @property
    def x1(self):
        return self._x1

    @property
    def y1(self):
        return self._y1

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    def is_valid(self):
        if self.invalid or self.w <= 0.0 or self.h <= 0.0:
            return False
        return True

    def __str__(self):
        return 'Face(x1={}, y1={}, w={}, h={})'.format(self.x1, self.y1, self.w, self.h)


class UFDD(BaseDataset):

    @property
    def name(self):
        return 'UFDD'

    @property
    def description(self):
        return 'Unconstrained Face Detection Dataset (UFDD)'

    @property
    def url(self):
        return 'https://ufdd.info/'

    def images(self):
        for i in self.trainval_set():
            yield i

    def __init__(self, root_dir):
        super(UFDD, self).__init__(root_dir)
        self._val_gt = os.path.join(self.root_dir, 'UFDD-annotationfile', 'UFDD_split', 'UFDD_val_bbx_gt-woDistractor.txt')
        self._val_images_dir = os.path.join(self.root_dir, 'UFDD_val', 'images')

    @staticmethod
    def _image_set(gt_txt, images_dir):
        """
        The format of txt ground truth.
        File name
        Number of bounding box
        x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
        :return:
        """
        with open(gt_txt) as f:
            filename = f.readline().rstrip()
            total = 1
            while filename:
                log.debug(filename)
                image = Image(os.path.join(images_dir, filename), filename)
                face_num = int(f.readline().rstrip())
                log.debug(face_num)
                for _ in range(face_num):
                    anno = f.readline().rstrip().split()
                    log.debug(anno)
                    face = Face(anno)
                    if face.is_valid():
                        image.add_face(face)
                    else:
                        log.debug('Skipping INVALID %s from %s', face, image)
                filename = f.readline().rstrip()
                total += 1
                yield image

    def train_set(self):
        return []

    def val_set(self):
        for i in self._image_set(self._val_gt, self._val_images_dir):
            yield i

    def trainval_set(self):
        for i in self.train_set():
            yield i
        for i in self.val_set():
            yield i

    def test_set(self):
        raise NotImplementedError()

    @property
    def train_dir(self):
        return None

    @property
    def val_dir(self):
        return os.path.join(self.root_dir, 'UFDD_val')

    def download(self):
        from morghulis.ufdd.downloader import UFDDDownloader
        downloader = UFDDDownloader(self.root_dir)
        downloader.download()
