# morghulis

Python API for face datasets:
 
 * [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
 * [FDDB](http://vis-www.cs.umass.edu/fddb/)
 
## Usage

### WIDER FACE

Use a `Wider` or `FDDB` dataset object to download and export to different formats:

```python
data_dir = '/datasets/WIDER'

ds = Wider(data_dir) 

# ds = FDDB(data_dir)

# downloads train, validation sets and annotations
ds.download()

# generate darknet (YOLO)
darknet = DarknetExporter(ds)
darknet.export(darknet_output_dir)

# generate tensorflow tf records
tensorflow = TensorflowExporter(ds)
tensorflow.export(tf_output_dir)

# generates LMDB files for caffe
caffe = CaffeExporter(ds)
exporter.export(caffe_output_dir)
```
