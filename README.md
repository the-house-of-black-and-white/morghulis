# morghulis

Python API for face detection datasets:
 
 * [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
 * [FDDB](http://vis-www.cs.umass.edu/fddb/)
 * [AFW](https://www.ics.uci.edu/~xzhu/face/)
 
## Usage


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

	modified:   README.md
	modified:   morghulis/fddb/__init__.py
	modified:   morghulis/model.py
	modified:   morghulis/widerface/__init__.py
	modified:   morghulis/widerface/darknet_exporter.py

### API

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
