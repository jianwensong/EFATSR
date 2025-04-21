# Efficient Frequency Feature Aggregation Transformer for Image Super-Resolution

## Dependencies
- Python 3.9
- PyTorch 1.10.0

```
cd code
pip install -r requirements.txt
python setup.py develop
```
## Datasets
- EFATSR

|  Training Set   | Testing Set   |
|  ----  | ----  |
|  DIV2K | Set5 + Set14 + BSD100 + Urban100 + Manga109  |

- EFATSSR

|  Training Set   | Testing Set   |
|  ----  | ----  |
|  Flickr1024 + Middlebury | KITTI2012 + KITTI2015 + Middlebury + Flickr1024  |

Refer to the datasets folder for the complete data. 

## Implementation of EFATSR
### Train

```shell
#scale factor 2
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/EFATSR/EFATSR_x2.yml --launcher pytorch
#scale factor 3
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/EFATSR/EFATSR_x3.yml --launcher pytorch
#scale factor 4
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/EFATSR/EFATSR_x4.yml --launcher pytorch
```
### Test
```shell
#scale factor 2
python scripts/test_SISR.py --scale 2 --model_path './experiments/pretrained_models/EFATSR_x2.pth'
#scale factor 3
python scripts/test_SISR.py --scale 3 --model_path './experiments/pretrained_models/EFATSR_x3.pth'    
#scale factor 4
python scripts/test_SISR.py --scale 4 --model_path './experiments/pretrained_models/EFATSR_x4.pth'  
```

## Implementation of EFATSSR
### Train

```shell
#scale factor 2
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/EFATSSR/EFATSSR_x2.yml --launcher pytorch
#scale factor 4
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/EFATSSR/EFATSSR_x4.yml --launcher pytorch
```
### Test
```shell
#scale factor 2
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/EFATSSR/EFATSSR_x2.yml --launcher pytorch
#scale factor 4
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/EFATSSR/EFATSSR_x4.yml --launcher pytorch
```
