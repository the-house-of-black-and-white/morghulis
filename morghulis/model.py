import logging
import os
from abc import ABCMeta, abstractproperty, abstractmethod
from shutil import copy

from PIL import Image as PilImage

from os_utils import ensure_dir

log = logging.getLogger(__name__)


class Image:
    def __init__(self, filename, raw_filename=None):
        self.filename = filename
        self.raw_filename = raw_filename
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

    @property
    def subdir(self):
        return os.path.dirname(self.raw_filename) + '/' if os.path.dirname(self.raw_filename) and self.raw_filename else None

    def copy_to(self, target_dir, include_subdirs=False):

        if include_subdirs and self.subdir:
            target_dir = os.path.join(target_dir, self.subdir)

        new_path = os.path.join(target_dir, os.path.basename(self.filename))

        if os.path.exists(new_path):
            log.error('%s is being overwritten.', new_path)
            raise Exception(new_path + ' already exists')

        ensure_dir(target_dir)
        copy(self.path, target_dir)

        return new_path

    def link_to(self, target_dir):
        target_dir = os.path.join(target_dir, self.subdir)
        new_path = os.path.join(target_dir, os.path.basename(self.filename))
        if os.path.exists(new_path):
            log.error('%s is being overwritten.', new_path)
            raise Exception(new_path + ' already exists')
        ensure_dir(target_dir)
        os.symlink(self.path, new_path)
        return new_path

    def __str__(self):
        return 'Image(filename={})'.format(self.filename)


class BaseFace:
    __metaclass__ = ABCMeta

    @abstractproperty
    def x1(self):
        pass

    @abstractproperty
    def y1(self):
        pass

    @property
    def x2(self):
        return self.x1 + self.w

    @property
    def y2(self):
        return self.y1 + self.h

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


FORMATS = {
    'tensorflow',
    'darknet',
    'caffe'
}


class BaseDataset:
    __metaclass__ = ABCMeta

    def __init__(self, data_dir):
        self.root_dir = data_dir

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def description(self):
        pass

    @abstractproperty
    def url(self):
        pass

    @abstractmethod
    def images(self):
        pass

    def export(self, target_dir, target_format):
        if target_format not in FORMATS:
            raise ValueError('Format not supported. Valid choices are %s:', FORMATS)

        exporter_class = getattr(self, 'get_{}_exporter'.format(target_format))()
        exporter = exporter_class(self)
        exporter.export(target_dir)

    def get_tensorflow_exporter(self):
        raise NotImplementedError()

    def get_caffe_exporter(self):
        raise NotImplementedError()

    def get_darknet_exporter(self):
        raise NotImplementedError()

    def download(self):
        raise NotImplementedError()


class Composite(BaseDataset):

    def __init__(self, datasets=None, data_dir=None):
        super(Composite, self).__init__(data_dir)
        self.datasets = datasets if datasets else []

    def add(self, dataset):
        self.datasets.append(dataset)

    @property
    def name(self):
        for ds in self.datasets:
            ds.name

    @property
    def description(self):
        pass

    @property
    def url(self):
        pass

    def images(self):
        for ds in self.datasets:
            for img in ds.images():
                yield img

    def get_tensorflow_exporter(self):
        pass

    def get_caffe_exporter(self):
        pass

    def get_darknet_exporter(self):
        pass

    def download(self):
        for ds in self.datasets:
            ds.download()
