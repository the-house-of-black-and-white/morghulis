# Morghulis

**Morghulis** is an attempt to create a common API for face datasets.
There are many face datasets available. Each of them has its own conventions and annotation format, but at the end, 
they all consist of a set of images with the respective annotated faces.

To make things worse the existent object detection libraries: [Detectron](https://github.com/facebookresearch/Detectron)
, [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [Darknet's YOLO](https://pjreddie.com/darknet/yolo/),
to name a few, all use different formats for train/eval/test. Detectron uses [COCO json format](http://cocodataset.org/#download),
Tensorflow uses [tf records](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md), and so on. 

Once Morghulis loads a dataset, it can be easily exported to different formats
 
Currently the following datasets are supported:
 
 * [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) - 32,203 images and 393,703 faces.
 * [FDDB](http://vis-www.cs.umass.edu/fddb/) - 2,845 images and 5,171 faces.
 * [AFW](https://www.ics.uci.edu/~xzhu/face/) - 205 images and 473 faces.
 * [PASCAL faces](https://www.ics.uci.edu/~xzhu/face/) - 850 images and 1335 faces.
 * **TODO** [MAFA](http://www.escience.cn/people/geshiming/mafa.html) - 30,811 images and 35,806 masked faces.
 * **TODO** [IJB-C](https://www.nist.gov/itl/iad/image-group/ijb-c-dataset-request-form-0)
 * **TODO** [Caltech faces](http://www.vision.caltech.edu/html-files/archive.html) - 450 frontal face images of 27 or so unique people

## Usage

TODO

### Docker

```bash
# Download wider face
docker run --rm -it \
    -v ${PWD}/datasets:/datasets \
    housebw/morghulis \
    ./download_dataset.py  --dataset widerface --output_dir /datasets/widerface

# Download fddb    
docker run --rm -it \
    --volumes-from ds \
    housebw/morghulis \
    ./download_dataset.py  --dataset fddb --output_dir /ds/fddb/

# Generate TF records for fddb
docker run --rm -it \
    --volumes-from ds \
    housebw/morghulis \
    ./export.py --dataset=fddb --format=tensorflow --data_dir=/ds/fddb/ --output_dir=/ds/fddb/tensorflow/
    
# Generate COCO json files for widerface
docker run --rm -it \
    -v ${PWD}/datasets:/ds \
    housebw/morghulis \
    ./export.py --dataset=widerface --format=coco --data_dir=/ds/widerface/ --output_dir=/ds/widerface/coco/
    
```

### API

Use a `Wider` or `FDDB` dataset object to download and export to different formats:

```python
data_dir = '/datasets/WIDER'

ds = Wider(data_dir) # FDDB(data_dir)

# downloads train, validation sets and annotations
ds.download()

# generate darknet (YOLO)
ds.export(darknet_output_dir, target_format='darknet')

# generate tensorflow tf records
ds.export(tf_output_dir, target_format='tensorflow')

# generates COCO json file (useful for Detectron)
ds.export(coco_output_dir, target_format='coco')
```
