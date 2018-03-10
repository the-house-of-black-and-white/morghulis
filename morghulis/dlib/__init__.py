# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from morghulis.model import Image as BaseImage, BaseFace, BaseDataset

import logging
import os

import xml.etree.ElementTree as ET


log = logging.getLogger(__name__)


class Image(BaseImage):
    def __init__(self, filename, raw_filename, width, height):
        BaseImage.__init__(self, filename, raw_filename)
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __str__(self):
        return 'Image(filename={})'.format(self.filename)


class Face(BaseFace):

    def __init__(self, anno):
        """
        ('xmin', 'ymin', 'xmax', 'ymax')
        :param annotations:
        """
        self._x1 = float(anno[0])
        self._y1 = float(anno[1])
        self._x2 = float(anno[2])
        self._y2 = float(anno[3])

    @property
    def x1(self):
        return self._x1

    @property
    def y1(self):
        return self._y1

    @property
    def w(self):
        return self._x2 - self._x1

    @property
    def h(self):
        return self._y2 - self._y1

    @property
    def center(self):
        pass


class Dlib(BaseDataset):

    @property
    def name(self):
        return 'dlib'

    @property
    def description(self):
        return """This dataset is a collection face images cropped from many publicly available
   image datasets (excluding the FDDB dataset).  In particular, there are images
   from ImageNet, AFLW, Pascal VOC, the VGG dataset, WIDER, and face scrub.

   Care was taken to make sure every face in each image was labeled.
"""

    @property
    def url(self):
        return ''

    def __init__(self, root_dir):
        super(Dlib, self).__init__(root_dir)
        self._annotations = os.path.join(self.root_dir, 'faces_2016_09_30.xml')

    def _xml_to_image(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        raw_filename = root.find('filename').text
        filename = os.path.join(self._images_dir, raw_filename)
        width = int(root.find('size')[0].text)
        height = int(root.find('size')[1].text)
        image = Image(filename, raw_filename, width, height)

        for object in root.findall('object'):
            for part in object.findall('part'):
                if part[0].text == 'head':
                    value = (int(part[1][0].text),
                             int(part[1][1].text),
                             int(part[1][2].text),
                             int(part[1][3].text)
                             )
                    image.add_face(Face(value))
        return image

    def images(self):
        processed_images = set()
        with open(self._train_val_gt, 'r') as trainval:
            for line in trainval:
                image_id = line.strip().split(' ')[0]
                if image_id not in processed_images:
                    processed_images.add(image_id)
                    annotation = os.path.join(self._annotations_dir, image_id + '.xml')
                    yield self._xml_to_image(annotation)

    def get_tensorflow_exporter(self):
        from morghulis.exporters.tf import TensorflowExporter
        return TensorflowExporter

    def get_caffe_exporter(self):
        raise NotImplementedError()

    def get_darknet_exporter(self):
        raise NotImplementedError()

    def get_coco_exporter(self):
        from morghulis.exporters.coco import BaseCocoExporter
        return BaseCocoExporter

    def download(self):
        from morghulis.pascal_faces.downloader import PascalFacesDownloader
        downloader = PascalFacesDownloader(self.root_dir)
        downloader.download()
