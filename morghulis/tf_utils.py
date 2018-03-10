from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def read_detections_from(tf_record, min_confidence=0.0):
    example = tf.train.Example()
    for record in tf.python_io.tf_record_iterator(tf_record):
        example.ParseFromString(record)
        f = example.features.feature

        filename = f['image/filename'].bytes_list.value[0]
        im_width = f['image/width'].int64_list.value[0]
        im_height = f['image/height'].int64_list.value[0]
        scores = f['image/detection/score'].float_list
        xmin_list = f['image/detection/bbox/xmin'].float_list
        xmax_list = f['image/detection/bbox/xmax'].float_list
        ymin_list = f['image/detection/bbox/ymin'].float_list
        ymax_list = f['image/detection/bbox/ymax'].float_list
        detections = []
        for score, xmin, xmax, ymin, ymax in zip(scores.value, xmin_list.value, xmax_list.value, ymin_list.value, ymax_list.value):
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

            if score >= min_confidence:
                detections.append((left, top, (right - left), (bottom - top), score))

        yield filename, detections