import logging
import os

from wider.model import BaseFace, Image

log = logging.getLogger(__name__)


class Face(BaseFace):
    def __init__(self, anno):
        """
        <major_axis_radius minor_axis_radius angle center_x center_y 1>
        :param annotations:
        """
        self._major_axis_radius = float(anno[0])
        self._minor_axis_radius = float(anno[1])
        self._angle = float(anno[2])
        self._center_x = float(anno[3])
        self._center_y = float(anno[4])

    @property
    def x1(self):
        return self._center_x - self._minor_axis_radius

    @property
    def y1(self):
        return self._center_y - self._major_axis_radius

    @property
    def w(self):
        return self._minor_axis_radius * 2

    @property
    def h(self):
        return self._major_axis_radius * 2

    @property
    def center(self):
        return self._center_x, self._center_y

    @property
    def angle(self):
        return self._angle

    @property
    def major_axis_radius(self):
        return self._major_axis_radius

    @property
    def minor_axis_radius(self):
        return self._minor_axis_radius

    def __str__(self):
        return 'Face(x1={}, y1={}, w={}, h={})'.format(self.x1, self.y1, self.w, self.h,
                                                                              self.invalid, self.blur)


class FDDB:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, 'originalPics')
        self.annotations_dir = os.path.join(self.root_dir, 'FDDB-folds')
        self.annotation_files = [os.path.join(self.annotations_dir, f) for f in os.listdir(self.annotations_dir) if
                                 'ellipseList' in f]

    def _image_set(self, annotation_file):
        with open(annotation_file) as f:
            filename = f.readline().rstrip()
            while filename:
                log.debug(filename)
                image = Image(os.path.join(self.images_dir, filename + '.jpg'), filename + '.jpg')
                face_num = int(f.readline().rstrip())
                log.debug(face_num)
                for _ in range(face_num):
                    annotations = f.readline().rstrip().split()
                    log.debug(annotations)
                    face = Face(annotations)
                    image.add_face(face)
                filename = f.readline().rstrip()
                yield image

    def images(self):
        for annotation_file in self.annotation_files:
            for i in self._image_set(annotation_file):
                yield i
