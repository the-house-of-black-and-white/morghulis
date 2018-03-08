# morghulis

Python API for face detection datasets:
 
 * [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
 * [FDDB](http://vis-www.cs.umass.edu/fddb/)
 * [AFW](https://www.ics.uci.edu/~xzhu/face/)
 * [PASCAL faces](https://www.ics.uci.edu/~xzhu/face/)
 
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
