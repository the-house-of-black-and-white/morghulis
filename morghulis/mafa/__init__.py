# coding=utf-8
import logging
import os

import scipy.io

from morghulis.model import Image as BaseImage, BaseFace, BaseDataset
from .downloader import MafaDownloader

log = logging.getLogger(__name__)


class Image(BaseImage):
    def __init__(self, filename, raw_filename=None):
        BaseImage.__init__(self, filename, raw_filename)


class Face(BaseFace):

    def __init__(self, data):
        """
        Data is 18d array (x,y,w,h,face_type,x1,y1,w1,h1, occ_type, occ_degree, gender, race, orientation, x2,y2,w2,h2), where
            (a) (x,y,w,h) is the bounding box of a face,
            (b) face_type stands for the face type and has: 1 for masked face, 2 for unmasked face and 3 for invalid face.
            (c) (x1,y1,w1,h1) is the bounding box of the occluder. Note that (x1,y1) is related to the face bounding box position (x,y)
            (d) occ_type stands for the occluder type and has: 1 for simple, 2 for complex and 3 for human body.
            (e) occ_degree stands for the number of occluded face parts
            (f) gender and race stand for the gender and race of one face
            (g) orientation stands for the face orientation/pose, and has: 1-left, 2-left frontal, 3-frontal, 4-right frontal, 5-right
            (h) (x2,y2,w2,h2) is the bounding box of the glasses and is set to (-1,-1,-1,-1) when no glasses.  Note that (x2,y2) is related to the face bounding box position (x,y)
        """

        self._x1, self._y1, self._w, self._h, face_type, x1, y1, w1, h1, occ_type, occ_degree, gender, race, orientation, x2, y2, w2, h2 = data

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
        pass


class Mafa(BaseDataset):

    def __init__(self, root_dir):
        super(Mafa, self).__init__(root_dir)
        self.images_dir = os.path.join(self.root_dir, 'images/')
        self.test_annotations_file = os.path.join(self.root_dir, 'LabelTestAll.mat')
        self.annotations_file = os.path.join(self.root_dir, 'LabelTrainAll.mat')

    @property
    def name(self):
        return 'mafa'

    @property
    def description(self):
        return """MAFA (MAsked FAces) 
        is a masked face detection benchmark dataset, of which images are collected from Internet images.
        MAFA contains 30,811 images and 35,806 masked faces. 
        Faces in the dataset have various various orientations and occlusion degrees, 
        while at least one part of each face is occluded by mask.
        In the annotation process, each image contains at least one face occluded 
        by various types of masks, while the six main attributes of each masked face,
        including locations of faces, eyes and masks, face orientation, occlusiondegree and mask type.
        """

    @property
    def url(self):
        return 'http://www.escience.cn/people/geshiming/mafa.html'

    def images(self):
        test_annotations = scipy.io.loadmat(self.test_annotations_file)
        data = test_annotations['LabelTest']
        for r in data[0]:
            raw_filename = r[0][0]
            data = r[1][0]
            image_filename = os.path.join(self.images_dir, raw_filename)
            image = Image(image_filename, raw_filename)
            image.add_face(Face(data))
            yield image


    # def get_tensorflow_exporter(self):
    #     pass
    #
    # def get_caffe_exporter(self):
    #     pass
    #
    # def get_darknet_exporter(self):
    #     pass
    #
    # def get_coco_exporter(self):
    #     pass

    def download(self):
        MafaDownloader(self.root_dir).download()
