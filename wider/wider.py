import hashlib
import io
import logging
import os

try:
    import tensorflow as tf
except ImportError:
    logging.warning("Tensorflow not installed some features won't work", exc_info=True)

try:
    from PIL import Image as PilImage
except ImportError:
    logging.warning("Pillow not installed some features won't work", exc_info=True)

log = logging.getLogger(__name__)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


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
        self._faces.append(face)

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
    def format(self):
        return self.image.format

    def tf_example(self):

        with tf.gfile.GFile(self.filename, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        self._image = PilImage.open(encoded_jpg_io)

        if self.format != 'JPEG':
            raise ValueError('Image format not JPEG')

        key = hashlib.sha256(encoded_jpg).hexdigest()
        width = self.width
        height = self.height
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []

        for face in self.faces:
            xmins.append(max(0.005, (face.x1 / width)))
            ymins.append(max(0.005, (face.y1 / height)))
            xmaxs.append(min(0.995, (face.x1 + face.w) / width))
            ymaxs.append(min(0.995, (face.y1 + face.h) / height))
            classes_text.append('face')
            classes.append(1)
            poses.append("unspecified".encode('utf8'))
            truncated.append(int(1) if face.occlusion > 0 else int(0))
            difficult_obj.append(int(0))

        _, filename = os.path.split(self.filename)

        feature_dict = {
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'image/filename': bytes_feature(filename.encode('utf8')),
            'image/source_id': bytes_feature(filename.encode('utf8')),
            'image/key/sha256': bytes_feature(key.encode('utf8')),
            'image/encoded': bytes_feature(encoded_jpg),
            'image/format': bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': float_list_feature(xmins),
            'image/object/bbox/xmax': float_list_feature(xmaxs),
            'image/object/bbox/ymin': float_list_feature(ymins),
            'image/object/bbox/ymax': float_list_feature(ymaxs),
            'image/object/class/text': bytes_list_feature(classes_text),
            'image/object/class/label': int64_list_feature(classes),
            'image/object/difficult': int64_list_feature(difficult_obj),
            'image/object/truncated': int64_list_feature(truncated),
            'image/object/view': bytes_list_feature(poses),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example


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

    @staticmethod
    def _generate_tf_records(output_filename, examples):
        writer = tf.python_io.TFRecordWriter(output_filename)
        for idx, example in enumerate(examples):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples))
            try:
                tf_example = example.tf_example()
                writer.write(tf_example.SerializeToString())
            except Exception:
                logging.warning('Invalid example: %s, ignoring.', example.filename)
        writer.close()

    def generate_tf_records(self, output_dir):

        output_filename = os.path.join(output_dir, 'wider_train.record')
        log.info('Loading train set, it might take a while')
        examples = [ex for ex in self.train_set()]
        log.info('Generating tf_record for train set: %s example(s)', len(examples))
        self._generate_tf_records(output_filename, examples)

        output_filename = os.path.join(output_dir, 'wider_val.record')
        log.info('Loading val set, it might take a while')
        examples = [ex for ex in self.val_set()]
        log.info('Generating tf_record for val set: %s example(s)', len(examples))
        self._generate_tf_records(output_filename, examples)
