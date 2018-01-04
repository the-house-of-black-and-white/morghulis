# coding=utf-8
import logging
import os

import h5py

from downloader import AFWDownloader
from morghulis.model import Image as BaseImage, BaseFace, BaseDataset

log = logging.getLogger(__name__)


class Image(BaseImage):
    def __init__(self, filename, raw_filename=None):
        BaseImage.__init__(self, filename, raw_filename)


class Face(BaseFace):
    def __init__(self, rect):
        """
        <major_axis_radius minor_axis_radius angle center_x center_y 1>
        :param annotations:
        """
        x1, y1, w, h = rect
        self._x1 = x1
        self._y1 = y1
        self._w = w
        self._h = h

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

    @property
    def center(self):
        return self.x1 + (self.w / 2.), self.y1 + (self.h / 2.)

    def __str__(self):
        return 'Face(x1={}, y1={}, w={}, h={})'.format(self.x1, self.y1, self.w, self.h)


class AFW(BaseDataset):
    def __init__(self, root_dir):
        super(AFW, self).__init__(root_dir)
        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, 'testimages')
        self.annotations_file = os.path.join(self.images_dir, 'anno.mat')

    def images(self):
        """
        Each row of "anno" corresponds to an image: 
            1st column is image name; 
            2nd column contains bounding boxes of faces [x1, y1; x2, y2]; 
            3rd column is pose [yaw, pitch, roll]. We only evaluate on yaw; 
            4th column contains 6 landmarks.
        :param annotation_file:
        :return:
        """
        f = h5py.File(self.annotations_file)
        f_base = f[u'anno']
        for indx in xrange(len(f_base[0])):
            # getting the name of the image
            obj = f[f_base[0][indx]]
            name = ''.join(chr(i) for i in obj[:])
            image = Image(os.path.join(self.images_dir, name), raw_filename=name)
            for face_indx in xrange(len(f[f[u'anno'][1][indx]])):
                # getting the bounding boxes of the face [x1, y1, x2, y2]. (upper left corner and lower right corner)
                obj = f[f[f_base[1][indx]][face_indx][0]]
                x1 = float(obj[0, 0])
                y1 = float(obj[1, 0])
                x2 = float(obj[0, 1])
                y2 = float(obj[1, 1])
                width = x2 - x1
                height = y2 - y1
                rect_init = [x1, y1, width, height]
                face = Face(rect_init)
                image.add_face(face)
                yield image
                # getting pose [yaw, pitch, roll]
                # obj = f[f[f_base[2][indx]][face_indx][0]]
                # yaw = float(obj[0])
                # pitch = float(obj[1])
                # roll = float(obj[2])

                # getting 6 landmarks. (left eye, right eye, nose, left mouth, mouth center, mouth right)
                # obj = f[f[f_base[3][indx]][face_indx][0]]
                # l_eye_x = float(obj[0, 0])
                # r_eye_x = float(obj[0, 1])
                # nose_x = float(obj[0, 2])
                # l_mouth_x = float(obj[0, 3])
                # c_mouth_x = float(obj[0, 4])
                # r_mouth_x = float(obj[0, 5])
                #
                # l_eye_y = float(obj[1, 0])
                # r_eye_y = float(obj[1, 1])
                # nose_y = float(obj[1, 2])
                # l_mouth_y = float(obj[1, 3])
                # c_mouth_y = float(obj[1, 4])
                # r_mouth_y = float(obj[1, 5])
                #
                # kpts = [l_eye_x, l_eye_y, r_eye_x, r_eye_y, nose_x, nose_y, l_mouth_x, l_mouth_y, c_mouth_x, c_mouth_y,
                #         r_mouth_x, r_mouth_y]

    def download(self):
        AFWDownloader(self.root_dir).download()

    def get_tensorflow_exporter(self):
        from morghulis.afw.tensorflow_exporter import TensorflowExporter
        return TensorflowExporter

    def get_darknet_exporter(self):
        from morghulis.afw.darknet_exporter import DarknetExporter
        return DarknetExporter
