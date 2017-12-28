import logging
import os

from model import Image

log = logging.getLogger(__name__)


class Wider:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self._train_gt = os.path.join(self.root_dir, 'wider_face_split', 'wider_face_train_bbx_gt.txt')
        self._train_images_dir = os.path.join(self.root_dir, 'WIDER_train', 'images')
        self._val_gt = os.path.join(self.root_dir, 'wider_face_split', 'wider_face_val_bbx_gt.txt')
        self._val_images_dir = os.path.join(self.root_dir, 'WIDER_val', 'images')

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
            while filename:
                log.debug(filename)
                image = Image(os.path.join(images_dir, filename))
                face_num = int(f.readline().rstrip())
                log.debug(face_num)
                for _ in range(face_num):
                    anno = f.readline().rstrip().split()
                    log.debug(anno)
                    image.add_face(anno)
                filename = f.readline().rstrip()
                yield image

    def train_set(self):
        for i in self._image_set(self._train_gt, self._train_images_dir):
            yield i

    def val_set(self):
        for i in self._image_set(self._val_gt, self._val_images_dir):
            yield i

    @property
    def train_dir(self):
        return os.path.join(self.root_dir, 'WIDER_train')

    @property
    def val_dir(self):
        return os.path.join(self.root_dir, 'WIDER_val')
