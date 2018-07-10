import logging
import os

from morghulis.model import BaseFace, Image as BaseImage, BaseDataset

try:
    import cv2
except:
    logging.warning('OpenCV not found')

log = logging.getLogger(__name__)


class Image(BaseImage):
    def __init__(self, filename, raw_filename=None):
        BaseImage.__init__(self, filename, raw_filename)

    # def draw_faces(self, image=None, color=(0, 0, 255), thickness=3):
    #     image = BaseImage.draw_faces(self, image, color, thickness)
    #     for f in self.faces:
    #         cv2.ellipse(image, f.center, (int(f.x2), int(f.y2)), color, thickness)
    #     return image


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
        return 'Face(x1={}, y1={}, w={}, h={})'.format(self.x1, self.y1, self.w, self.h)


class FDDB(BaseDataset):
    def __init__(self, root_dir):
        super(FDDB, self).__init__(root_dir)
        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, 'originalPics')
        self.annotations_dir = os.path.join(self.root_dir, 'FDDB-folds')
        self._initialize_dirs()

    @property
    def name(self):
        return 'FDDB'

    @property
    def description(self):
        return 'Face Detection Data Set and Benchmark'

    @property
    def url(self):
        return 'http://vis-www.cs.umass.edu/fddb/'

    def _initialize_dirs(self):
        if os.path.exists(self.annotations_dir):
            self.annotation_files = [os.path.join(self.annotations_dir, f) for f in os.listdir(self.annotations_dir) if
                                     'ellipseList' in f]
            self.fold_files = [os.path.join(self.annotations_dir, f) for f in os.listdir(self.annotations_dir) if
                                     'ellipseList' not in f]
        else:
            log.warning('Annotation dir %s not found. Check the root_dir or download the dataset first',
                        self.annotations_dir)

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

    def folds(self):
        for fold_file in self.fold_files:
            yield fold_file[-6:-4], fold_file

    def download(self):
        from morghulis.fddb.downloader import FddbDownloader
        downloader = FddbDownloader(self.root_dir)
        downloader.download()
        self._initialize_dirs()

    def get_tensorflow_exporter(self):
        from morghulis.fddb.tensorflow_exporter import TensorflowExporter
        return TensorflowExporter

    def get_darknet_exporter(self):
        from morghulis.fddb.darknet_exporter import DarknetExporter
        return DarknetExporter
