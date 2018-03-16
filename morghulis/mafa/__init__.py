# coding=utf-8
import logging
import os
import cv2
import scipy.io

from morghulis.model import Image as BaseImage, BaseFace, BaseDataset
from .downloader import MafaDownloader

log = logging.getLogger(__name__)


class Image(BaseImage):
    def __init__(self, filename, raw_filename=None):
        BaseImage.__init__(self, filename, raw_filename)

    def draw_faces(self, image=None, color=(0, 0, 255), thickness=3):
        image = BaseImage.draw_faces(self, image, color, thickness)
        if image is None:
            image = self.image_as_nparray()
        for f in self.faces:
            x = int(f.x1)
            y = int(f.y1)
            cv2.rectangle(image, (x, y), (int(f.x2), int(f.y2)), color, thickness)

            x1, y1, w1, h1 = f.occluder_bbox
            cv2.rectangle(image, (int(x + x1), int(y + y1)), (int(x + x1 + w1), int(y + y1 + h1)), (0, 255, 255), thickness)

            x2, y2, w2, h2 = f.glasses_bbox
            if x2 != -1:
                cv2.rectangle(image, (int(x + x2), int(y + y2)), (int(x + x2 + w2), int(y + y2 + h2)), (255, 0, 0), thickness)

        return image


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
        self._x1, self._y1, self._w, self._h, self._type, x1, y1, w1, h1, self._occlusion_type, self._occ_degree, self._gender, self._race, self._orientation, x2, y2, w2, h2 = data
        self._occluder_bbox = x1, y1, w1, h1
        self._glasses_bbox = x2, y2, w2, h2

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

    @property
    def type(self):
        """
        face_type stands for the face type and has:
        :return: 1 for masked face, 2 for unmasked face and 3 for invalid face.
        """
        return self._type

    @property
    def occlusion_type(self):
        """
        occ_type stands for the occluder type and has:
        :return: 1 for simple, 2 for complex and 3 for human body.
        """
        return self._occlusion_type

    @property
    def occlusion_degree(self):
        """
        the number of occluded face parts
        :return: the number of occluded face parts
        """
        return self._occ_degree

    @property
    def gender(self):
        return self._gender

    @property
    def race(self):
        return self._race

    @property
    def orientation(self):
        """
        Stands for the face orientation/pose, and has:
        :return: 1-left, 2-left frontal, 3-frontal, 4-right frontal, 5-right
        """
        return self._orientation

    @property
    def occluder_bbox(self):
        """
        bounding box of the occluder. Note that (x1,y1) is related to the face bounding box position (x,y)
        :return:
        """
        return self._occluder_bbox

    @property
    def glasses_bbox(self):
        """"
        bounding box of the glasses and is set to (-1,-1,-1,-1) when no glasses.  Note that (x2,y2) is related to the face bounding box position (x,y)
        """
        return self._glasses_bbox


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

    def _load_images_from(self, annotation_file, key):
        test_annotations = scipy.io.loadmat(annotation_file)
        data = test_annotations[key]
        for r in data[0]:
            raw_filename = r[0][0]
            image_filename = os.path.join(self.images_dir, raw_filename)
            image = Image(image_filename, raw_filename)
            for data in r[1]:
                image.add_face(Face(data))
            yield image

    def images(self):
        for i in self._load_images_from(self.test_annotations_file, 'LabelTest'):
            yield i

    def train_set(self):
        raise NotImplementedError()

    def val_set(self):
        raise NotImplementedError()

    def trainval_set(self):
        raise NotImplementedError()

    def test_set(self):
        return [i for i in self._load_images_from(self.test_annotations_file, 'LabelTest')]

    def get_tensorflow_exporter(self):
        pass

    def get_caffe_exporter(self):
        pass

    def get_darknet_exporter(self):
        pass

    def get_coco_exporter(self):
        from morghulis.exporters.coco import BaseCocoExporter
        return BaseCocoExporter

    def download(self):
        MafaDownloader(self.root_dir).download()
