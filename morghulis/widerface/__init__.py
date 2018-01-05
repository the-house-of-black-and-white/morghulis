# coding=utf-8
import logging
import os
import re
from morghulis.model import Image as BaseImage, BaseFace, BaseDataset

log = logging.getLogger(__name__)

# http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/paper.pdf
# subset partitions based on scale

HARD_SUBSET_CATEGORIES = {'Traffic', 'Festival', 'Parade', 'Demonstration', 'Ceremony', 'People Marching', 'Basketball',
                          'Shoppers', 'Matador Bullfighter', 'Car Accident', 'Election Campain', 'Concerts',
                          'Award Ceremony', 'Picnic', 'Riot', 'Funeral', 'Cheering', 'Soldier Firing', 'Car Racing',
                          'Voter'}

MEDIUM_SUBSET_CATEGORIES = {'Stock Market', 'Hockey', 'Students Schoolkids', 'Ice Skating', 'Greeting', 'Football',
                            'Running', 'people driving car', 'Soldier Drilling', 'Photographers', 'Sports Fan', 'Group',
                            'Celebration Or Party', 'Soccer', 'Interview', 'Raid', 'Baseball', 'Soldier Patrol',
                            'Angler', 'Rescue'}

EASY_SUBSET_CATEGORIES = {'Gymnastics', 'Handshaking', 'Waiter Waitress', 'Press Conference', 'Worker Laborer',
                          'Parachutist Paratrooper', 'Sports Coach Trainer', 'Meeting', 'Aerobics', 'Row Boat',
                          'Dancing', 'Swimming', 'Family Group', 'Balloonist', 'Dresses', 'Couple', 'Jockey', 'Tennis',
                          'Spa', 'Surgeons'}

CATEGORY_RE = re.compile(r".*/(\d+)--(\w+)/.*")


class Event:
    def __init__(self, filename):
        match = next(re.finditer(CATEGORY_RE, filename))
        self._id = match.group(1)
        self._category = match.group(2).replace('_', ' ')

    @property
    def id(self):
        return self._id

    @property
    def category(self):
        return self._category

    def __str__(self):
        return 'Event(id={}, category={})'.format(self.id, self.category)


class Image(BaseImage):
    def __init__(self, filename, raw_filename=None):
        BaseImage.__init__(self, filename, raw_filename)
        self.event = Event(filename)

    def is_hard(self):
        return self.event.category in HARD_SUBSET_CATEGORIES

    def is_easy(self):
        return self.event.category in EASY_SUBSET_CATEGORIES

    def is_medium(self):
        return self.event.category in MEDIUM_SUBSET_CATEGORIES

    def __str__(self):
        return 'Image(filename={}, event={})'.format(self.filename, self.event)


class Face(BaseFace):
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

    def is_large(self):
        return self.h > 300

    def is_medium(self):
        return 50 <= self.h <= 300

    def is_small(self):
        return self.h < 50

    def is_partially_occluded(self):
        """
        A face is defined as ‘partially occluded’ if 1%-30% of the total face area is occluded.
        :return:
        """
        return self.occlusion == 1

    def is_heavily_occluded(self):
        """
        A face with occluded area over 30% is labeled as ‘heavily occluded’
        :return:
        """
        return self.occlusion == 2

    def has_typical_pose(self):
        """
        A face with occluded area over 30% is labeled as ‘heavily occluded’
        :return:
        """
        return self.pose == 0

    def has_atypical_pose(self):
        """
        Face is annotated as atypical under two conditions:
            either the roll or pitch degree is larger than 30-degree;
            or the yaw is larger than 90-degree
        :return:
        """
        return self.pose == 1

    def is_valid(self):
        if self.invalid or self.w <= 0.0 or self.h <= 0.0:
            return False
        return True

    def __str__(self):
        return 'Face(x1={}, y1={}, w={}, h={}, invalid={}, blur={})'.format(self.x1, self.y1, self.w, self.h,
                                                                            self.invalid, self.blur)


class Wider(BaseDataset):

    @property
    def name(self):
        return 'WiderFace'

    @property
    def description(self):
        return 'WIDER FACE: A Face Detection Benchmark'

    @property
    def url(self):
        return 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/'

    def images(self):
        pass

    def __init__(self, root_dir):
        super(Wider, self).__init__(root_dir)
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
                image = Image(os.path.join(images_dir, filename), filename)
                face_num = int(f.readline().rstrip())
                log.debug(face_num)
                for _ in range(face_num):
                    anno = f.readline().rstrip().split()
                    log.debug(anno)
                    face = Face(anno)
                    if face.is_valid():
                        image.add_face(face)
                    else:
                        log.debug('Skipping INVALID %s from %s', face, image)
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

    def download(self):
        from morghulis.widerface.downloader import WiderFaceDownloader
        downloader = WiderFaceDownloader(self.root_dir)
        downloader.download()

    def get_tensorflow_exporter(self):
        from morghulis.widerface.tensorflow_exporter import TensorflowExporter
        return TensorflowExporter

    def get_caffe_exporter(self):
        from morghulis.widerface.caffe_exporter import CaffeExporter
        return CaffeExporter

    def get_darknet_exporter(self):
        from morghulis.widerface.darknet_exporter import DarknetExporter
        return DarknetExporter
