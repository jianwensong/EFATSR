### Datasets for EFATSR
The structure of the training set is as follows:
```
|-- datasets
    |-- SISR
        |-- train
            |-- DIV2K_train_HR
            |-- DIV2K_train_LR_bicubic
                |-- X2
                |-- X3
                |-- X4
```
The structure of the testing sets is as follows:
```
|-- datasets
    |-- SISR
        |-- test
            |-- Set5
                |-- HR
                |-- LR_bicubic
                    |-- X2
                    |-- X3
                    |-- X4
            |-- Set14
                |-- HR
                |-- LR_bicubic
                    |-- X2
                    |-- X3
                    |-- X4
            |-- B100
                |-- HR
                |-- LR_bicubic
                    |-- X2
                    |-- X3
                    |-- X4
            |-- Urban100
                |-- HR
                |-- LR_bicubic
                    |-- X2
                    |-- X3
                    |-- X4
            |-- Manga109
                |-- HR
                |-- LR_bicubic
                    |-- X2
                    |-- X3
                    |-- X4
```
### Datasets for EFATSSR
The structure of the training set is as follows:
```
|-- datasets
    |-- STSR
        |-- train
            |-- Flickr1024        
                |-- HR
                |-- train_bicubic_x2
                |-- train_bicubic_x4
```
The structure of the testing sets is as follows:
```
|-- datasets
    |-- STSR
        |-- test
            |-- KITTI2012
                |-- hr
                |-- lr_x2
                |-- lr_x4
            |-- KITTI2015
                |-- hr
                |-- lr_x2
                |-- lr_x4
            |-- Middlebury
                |-- hr
                |-- lr_x2
                |-- lr_x4
            |-- Flickr1024
                |-- hr
                |-- lr_x2
                |-- lr_x4
```
### Download links
EFATSR:
| Dataset  | Description |
| -------- | -------- |
| [DIV2K Train HR](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)   | Train Data (HR images)   |
| [DIV2K Train LR x2](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip)   | Train Data x2 (LR images)   |
| [DIV2K Train LR x3](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip)   | Train Data x3 (LR images)    |
| [DIV2K Train LR x4](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip)   | Train Data x4 (LR images)   |
| [Test](https://1drv.ms/u/s!Ary1RQsbjs98lqsTe2KvPpduJfnDkQ?e=We1EUZ)   | Testsets for Set5+Set14+BSD100+Urban100+Manga109|

EFATSSR:
| Dataset  | Description |
| -------- | -------- |
| [Train](https://1drv.ms/u/s!Ary1RQsbjs98lNoSSRyoLxDbZkDa3A?e=GJ0A3i)   | Tradin Data (HR images + LR images)   |
| [Test](https://1drv.ms/u/s!Ary1RQsbjs98lqsUyO_DZddUset5iA?e=SqqU3b)   | Testsets for KITTI2012 + KITTI2015 + Middlebury + Flickr1024 (LR images)   |


