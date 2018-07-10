# coding=utf-8
import logging
import os

import scipy.io

from morghulis.model import Image as BaseImage, BaseFace, BaseDataset
from .downloader import CaltechFacesDownloader

log = logging.getLogger(__name__)


class Image(BaseImage):
    def __init__(self, filename, raw_filename=None):
        BaseImage.__init__(self, filename, raw_filename)


class Face(BaseFace):

    def __init__(self, column):
        """
        Each column of this matrix hold the coordinates of
        the bike within the image, in the form:

        [x_bot_left y_bot_left x_top_left y_top_left ...
         x_top_right y_top_right x_bot_right y_bot_right]
        """
        # x_bot_left, y_bot_left = column[0], column[1]
        x_top_left, y_top_left = column[2], column[3]
        # x_top_right, y_top_right = column[4], column[5]
        x_bot_right, y_bot_right = column[6], column[7]
        self._x1 = float(x_top_left)
        self._y1 = float(y_top_left)
        self._w = float(x_bot_right - x_top_left)
        self._h = float(y_bot_right - y_top_left)

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


class CaltechFaces(BaseDataset):

    def __init__(self, root_dir):
        super(CaltechFaces, self).__init__(root_dir)
        self.images_dir = self.root_dir
        self.annotations_file = os.path.join(self.images_dir, 'ImageData.mat')

    @property
    def name(self):
        return 'Caltech Faces'

    @property
    def description(self):
        return """Frontal face dataset. 
        Collected by Markus Weber at California Institute of Technology.
        450 face images. 896 x 592 pixels. Jpeg format. 
        27 or so unique people under with different lighting/expressions/backgrounds.
        """

    @property
    def url(self):
        return 'http://www.vision.caltech.edu/html-files/archive.html'

    def images(self):
        f = scipy.io.loadmat(self.annotations_file)
        data = f['SubDir_Data']
        image_count = 1
        for column in data.T:
            image_filename = os.path.join(self.images_dir, 'image_{:04d}.jpg'.format(image_count))
            image = Image(image_filename, image_filename)
            image.add_face(Face(column))
            image_count += 1
            yield image

    def download(self):
        CaltechFacesDownloader(self.root_dir).download()


