# pyWiderFace

Simple python API for the [WIDER FACE dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/).

## Usage

First download:

* Wider Face [training images](https://drive.google.com/file/d/0B6eKvaijfFUDQUUwd21EckhUbWs/view?usp=sharing)
* Wider Face [validation images](https://drive.google.com/file/d/0B6eKvaijfFUDd3dIRmpvSk8tLUk/view?usp=sharing)
* [Face annotations](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip)

Unzip them in the same directory and you should end up with the following structure:

```bash
   .
   |-wider_face_split
   |-WIDER_train
   |---images
   |-WIDER_val
   |---images

```

Now you can create a new `Wider` object.

```python
wider = Wider(data_dir)
```

### Darknet (YOLO) exporter

```python
wider = Wider(data_dir)
exporter = DarknetExporter(wider)
exporter.export(output_dir)
```

### Tensorflow exporter

Creates 2 `tfrecords`: `widerface-train.tfrecord` and `widerface-val.tfrecord`

```python
wider = Wider(data_dir)
exporter = TensorflowExporter(wider)
exporter.export(output_dir)
```

### [TODO] Caffe exporter

Creates LMDB files for caffe training

```python
wider = Wider(data_dir)
exporter = CaffeExporter(wider)
exporter.export(output_dir)
```
