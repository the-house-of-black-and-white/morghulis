from shutil import copy
import logging
import os

try:
    from PIL import Image as PilImage
except ImportError:
    logging.warning("Pillow not installed some features won't work")

log = logging.getLogger(__name__)


class Image:
    def __init__(self, filename):
        self.filename = filename
        self._faces = []
        self._image = None

    def add_face(self, annotations):
        """
        x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
        :param annotations:
        :return:
        """
        face = Face(annotations)

        if face.invalid == 1:
            log.warning('Skipping INVALID %s from %s', face, self)
            return

        # if face.blur > 0:
        #     log.warning('Skipping BLURRED %s from %s', face, self)
        #     return

        n = max(face.w, face.h)
        if n < 20:
            log.warning('Skipping SMALL(<20) %s from %s', face, self)
            return

        self._faces.append(face)

    @property
    def path(self):
        return os.path.abspath(self.filename)

    @property
    def image(self):
        if not self._image:
            self._image = PilImage.open(self.filename)
        return self._image

    @property
    def faces(self):
        return self._faces

    @property
    def width(self):
        return self.image.width

    @property
    def height(self):
        return self.image.height

    @property
    def size(self):
        return self.image.size

    @property
    def format(self):
        return self.image.format

    def copy_to(self, target_dir):
        new_path = os.path.join(target_dir, os.path.basename(self.filename))
        copy(self.path, target_dir)
        return new_path

    def __str__(self):
        return 'Image( filename={} )'.format(self.filename)


class Face:
    def __init__(self, anno):
        """
        x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
        :param annotations:
        """
        self._x1 = float(anno[0])
        self._y1 = float(anno[1])
        self._w = float(anno[2])
        self._h = float(anno[3])
        self._blur = int(anno[4])
        self._expression = int(anno[5])
        self._illumination = int(anno[6])
        self._invalid = int(anno[7])
        self._occlusion = int(anno[8])
        self._pose = int(anno[9])

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

    @property
    def blur(self):
        return self._blur

    @property
    def expression(self):
        return self._expression

    @property
    def illumination(self):
        return self._illumination

    @property
    def invalid(self):
        return self._invalid

    @property
    def occlusion(self):
        return self._occlusion

    @property
    def pose(self):
        return self._pose

    def __str__(self):
        return 'Face( x1={}, y1={}, w={}, h={}, invalid={}, blur={} )'.format(self.x1, self.y1, self.w, self.h,
                                                                              self.invalid, self.blur)


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
