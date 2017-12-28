import os
import logging
from shutil import copy

from PIL import Image as PilImage

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