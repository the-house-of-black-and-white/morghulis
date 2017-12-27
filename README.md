# pyWiderFace

Simple python API for the [WIDER FACE dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/).

```python
wider = Wider(data_dir)
```

## Darknet (YOLO) exporter

```python
wider = Wider(data_dir)
exporter = DarknetExporter(wider)
exporter.export(output_dir)
```

## Tensorflow exporter

Creates 2 `tfrecords`: `widerface-train.tfrecord` and `widerface-val.tfrecord`

```python
wider = Wider(data_dir)
exporter = TensorflowExporter(wider)
exporter.export(output_dir)
```

## **TODO** Caffe exporter

Creates LMDB files for caffe training

```python
wider = Wider(data_dir)
exporter = CaffeExporter(wider)
exporter.export(output_dir)
```
