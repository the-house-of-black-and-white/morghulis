# coding=utf-8

import json
import logging
import os
from collections import namedtuple

from datetime import datetime

from morghulis.os_utils import ensure_dir

Category = namedtuple('Category', ['id', 'name', 'supercategory'])
License = namedtuple('License', ['id', 'name', 'url'])
Info = namedtuple('Info', ['year', 'version', 'description', 'contributor', 'url', 'date_created'])
Image = namedtuple('Image', ['id', 'width', 'height', 'file_name'])
Annotation = namedtuple('Annotation', ['id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox', 'iscrowd'])

log = logging.getLogger(__name__)


class BaseCocoExporter:
    """
    Exports a dataset to the COCO json format
    http://cocodataset.org/#download
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def _export(self, target_file, images_iterable):
        log.info('Exporting %s to COCO json format', self.dataset.name)
        now = datetime.utcnow()
        images = []
        annotations = []
        img_id = 1
        annotation_id = 1

        for img in images_iterable:
            images.append(Image(img_id, img.width, img.height, img.raw_filename)._asdict())
            for face in img.faces:
                segmentation = [face.poly]
                annotations.append(Annotation(annotation_id, img_id, 0, segmentation, face.area, [face.x1, face.y1, face.w, face.h], 0)._asdict())
                annotation_id += 1
            img_id += 1

        wider_coco = dict(
            info=Info(2016, 'v1.0', self.dataset.description, '', self.dataset.url, now.isoformat())._asdict(),
            images=images,
            annotations=annotations,
            licenses=[License(0, '', '')._asdict()],
            categories=[Category(0, 'face', 'human body part')._asdict()]
        )

        log.info('Saving json file: %s', target_file)
        with open(target_file, 'w') as fp:
            json.dump(wider_coco, fp)

    def export(self, target_dir):
        ensure_dir(target_dir)
        target_file = '{}.json'
        output_filename = os.path.join(target_dir, target_file.format(self.dataset.name))
        self._export(output_filename, self.dataset.images())
