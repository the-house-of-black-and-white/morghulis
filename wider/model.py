import logging
import os
from abc import ABCMeta, abstractmethod, abstractproperty
from shutil import copy

from PIL import Image as PilImage

log = logging.getLogger(__name__)


class Image:
    def __init__(self, filename):
        self.filename = filename
        self._faces = []
        self._image = None

    def add_face(self, face):
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


class BaseFace:
    __metaclass__ = ABCMeta

    @abstractproperty
    def w(self):
        pass

    @abstractproperty
    def h(self):
        pass

    @abstractproperty
    def center(self):
        pass

    @property
    def invalid(self):
        return 0
