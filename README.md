# 3D Directional Encoding for Point Cloud Analysis

## Installation
source install.sh

## Dataset
Follow the [PointNeXt](https://guochengqian.github.io/PointNeXt/) to download the datasets.


## Train
- ScanObjetNN:
```
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/denet.yaml 
```
- S3DIS:
```
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/s3dis/denet.yaml 
```
- ShapeNetPart:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/denet.yaml 
```
## Test
Pretrained models: [Google Drive](https://drive.google.com/drive/folders/1yYeb0Nii0EmNin-saTB6DLa6j5ejgnPb)
- ScanObjetNN:
```
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/denet.yaml --mode test --pretrained_path /path/to/pretrained/network
```
- S3DIS:
```
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/s3dis/denet.yaml --mode test --pretrained_path /path/to/pretrained/network
```
- ShapeNetPart:
```
CUDA_VISIBLE_DEVICES=0 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/denet.yaml --mode test --pretrained_path /path/to/pretrained/network 
```
## Acknowledgement
This repository is based on the codes of [PointNeXt](https://github.com/guochengqian/PointNeXt), [PointMetaBase](https://github.com/linhaojia13/PointMetaBase) and [DeLA](https://github.com/Matrix-ASC/DeLA).
