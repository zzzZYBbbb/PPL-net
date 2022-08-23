# PPL-net

## Introduction
PPL-net is a line segment extraction method that combines point features and line features. It uses advanced feature extraction method based on convolutional networks, and then uses a point-line verification framework to solve the problem of over-extraction and local ambiguity.
<p align="center">
<p>

## Results

### Comparison of average results of five algorithms in data set

| Methods | AP | AR | F-measure| Number| Length| FPS|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| [LSD](https://ieeexplore.ieee.org/document/4731268/) | 0.675 | 0.412 | 0.232| 788.37| 25.96| 18.88|
| [EDlines](https://ieeexplore.ieee.org/document/6116138) | 0.746 | 0.521 | 0.222 | **957.36** | 27.78 | **36.59** |
| [Linelet](https://ieeexplore.ieee.org/document/7926451)| 0.767 | 0.498 | 0.249| 955.91| 23.18| 0.09|
| [AFM](https://ieeexplore.ieee.org/document/8954315)| 0.892 | 0.547 | 0.259| 606.17| **31.67**| 23.98|
|Ours| **0.906** | **0.569** | **0.278**| 661.22| 29.65| 24.06|

### Precision and Recall Curves
<p align="center">
<img src="figures/PR curve.jpg"  width="500">
</p>

## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions.


## 1.Data preparation
### 1.1 Downloading data
- Wireframe Dataset: https://github.com/huangkuns/wireframe
- YorkUrban Dataset: http://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/

Please follow the above links to download Wireframe and YorkUrban datasets. For Wireframe dataset, we only need the file named pointlines.zip which contains images and line segment annotations for training and testing. 

Once the files are downloaded, please unzip them into <AFM_root>/data/wireframe_raw and <AFM_root>/data/york_raw  respectively. The structures of wireframe_raw and york_raw folder are as follows:
```
wireframe_raw/
    - pointlines/*.pkl
    - train.txt
    - test.txt

york_raw/
    - filename0_rgb.png
    - filename0.mat
    ...
    - filename{N}_rgb.png
    - filename{N}.mat
```

### 1.2. Data Pre-processing
Please run the following commands
```
cd <AFM_root>/data/
python preparation_wireframe.py
python preparation_york.py
```

## 2. Hyper-parameter configurations
We use the [YACS](https://github.com/rbgirshick/yacs) to control the hyper parameters. Our configuration files for U-Net [(afm_unet.yaml)](experiments/afm_unet.yaml) and a-trous Residual Unet [(afm_atrous.yaml)](experiments/afm_atrous.yaml) are saved in the "<AFM_root>/experiments" folder.

In each yaml file, the SAVE_DIR is used to store the network weights and experimental results. The weights are saved in SAVE_DIR/weights and the results are saved in SAVE_DIR/results/DATASET_name.

The TEST configuration is for outputing results in testing phase with different ways (e.g. save or display). We currently provide two output modes "display" and "save". 
You can custom more output methods in [modeling/output/output.py](modeling/output/output.py). 

## 3. Inference with pretrained models
The pretrained models for U-Net and atrous Residual U-Net can be downloaded from [this link](https://drive.google.com/file/d/1AnLWs91vQdsJm6jJhB7MAvbIIQc0hJL2/view?usp=sharing). Please place the weights into "<AFM_root>/experiments/unet/weight" and "<AFM_root>/experiments/atrous/weight" respectively. 

- For testing, please run the following command

```
python test.py --config-file experiments/afm_atrous.yaml --gpu 0
```


## 4. Training
Please run the following command 
```
python train.py --config-file experiments/afm_atrous.yaml --gpu 0
```
to train a network. To speedup training procedure, our code will save the generated attraction field maps into <AFM_root>/data/wireframe/.cache when you run training code in the first time.
```
