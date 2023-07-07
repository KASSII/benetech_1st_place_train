# 1st Place Solution of the Benetech - Making Graphs Accessible Kaggle Competition

This repository is the training code for the kaggle Benetech Competition 1st Solution.  
For more information on the solution, please check the kaggle [discussion](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/418786).

## Preparation  
1. Clone this repository  
2. Download the dataset from this [link](https://www.kaggle.com/datasets/kashiwaba/benetech-trainingdata) and place it under the datas/input distribution.  

The folder structure is as follows  
(The folder hierarchy is deep for the convenience of uploading, so be careful to arrange the folders as shown below.)
```
benetech_1st_place_train/  
　├ datas/  
　│　├ input/  
　│　│  ├ ICDAR_2022/  
　│　│  │  ├ manual_annotation/  
　│　│  │  └ useful/  
　│　│  ├ benetech-making-graphs-accessible/  
　│　│  │  ├ train/  
　│　│  │  ├ extracted_4fold_axis_reading.csv  
　│　│  │  ├ extracted_clean_4fold.csv  
　│　│  │  ├ generated_4fold_axis_reading.csv  
　│　│  │  └ generated_clean_4fold.csv  
　│　│  ├ generated_synthetic/  
　│　│  │  ├ dot/  
　│　│  │  ├ horizontal_bar/  
　│　│  │  ├ line/  
　│　│  │  ├ scatter/  
　│　│  │  └ vertical_bar/  
　│　│  └ dataset_for_yolox/  
　│　│     ├ annotations/  
　│　│     ├ train2017/  
　│　│     └ val2017/  
　│　└ output/  
　├ classification_train/  
　├ deplot_train/  
　├ YOLOX_train/  
　└ README.md  
```

## Train
### 1. classification model
1. Install dependent libraries  
`> cd classification_train`  
`> pip install -r requirements.txt`  
2. Run the training script  
`> python train_convnext.py --output ../datas/output/classification/convnext`  
`> python train_swin.py --output ../datas/output/classification/swin`  
3. Results are output to `benetech_1st_place_train/datas/output/classification/convnext` and `benetech_1st_place_train/datas/output/classification/swin`  

### 2. Bar/Line/Dot model
1. Install dependent libraries  
`> cd deplot_train`  
`> pip install -r requirements.txt`  

#### a. horizontal_bar & line (All Chart-type Train)  
1. Run the training script  
`> python AllChartTrain_v1/train.py`  
2. Results are output to `benetech_1st_place_train/datas/output/AllChartTrain_v1/<time_stamp>`  

#### b. line (Specific Chart-type Train)
1. Run the All Chart-type Train training script  
`> python AllChartTrain_v1/train.py`  

2. Rewrite config file.  
   Rewrite the file path (the timestamp part) in `benetech_1st_place_train/deplot_train/SpecificChartTrain_Line/config.py (l.15)` to match your environment. (Rewrite the folder name with the result of step 1 in the output.)
3. Run the Specific Chart-type Train training script  
`> python SpecificChartTrain_Line/train.py`  

4. Results are output to `benetech_1st_place_train/datas/output/SpecificChartTrain_Line/<time_stamp>`  

#### c. vertical_bar (Specific Chart-type Train)
1. Run the All Chart-type Train training script  
   ※ Note that the script of the AllChart-type train, which is the base for vertical_bar, is different from the others (in terms of the percentage of data used and minor source code differences). The details of the differences can be seen by taking the differences.  
`> python AllChartTrain_v2/train.py`  

2. Rewrite config file.  
   Rewrite the file path (the timestamp part) in `benetech_1st_place_train/deplot_train/SpecificChartTrain_VerticalBar_1st/config.py (l.15)` to match your environment. (Rewrite the folder name with the result of step 1 in the output.)

3. Run the Specific Chart-type Train 1st training script  
`> python SpecificChartTrain_VerticalBar_1st/train.py`  

4. Rewrite config file.  
   Rewrite the file path (the timestamp part) in `benetech_1st_place_train/deplot_train/SpecificChartTrain_VerticalBar_2nd/config.py (l.15)` to match your environment. (Rewrite the folder name with the result of step 3 in the output.)

5. Run the Specific Chart-type Train 2nd training script  
`> python SpecificChartTrain_VerticalBar_2nd/train.py`  

4. Results are output to `benetech_1st_place_train/datas/output/SpecificChartTrain_VerticalBar_2nd/<time_stamp>`  


### 3. Scatter model
#### a. label text reading model  
1. Install dependent libraries  
`> cd deplot_train`  
`> pip install -r requirements.txt`  

2. Run the training script  
`> python AxisReading/train.py`  

3. Results are output to `benetech_1st_place_train/datas/output/AxisReading/<time_stamp>`  

#### b. scatter detect model
1. Install dependent libraries  
`> cd YOLOX_train`  
`> pip install -r requirements.txt`  
`> pip install -v -e .`  

2. Download pre-training model  
Download the yolox_l trained model from the YOLOX formula site below.After downloading, create a folder called weights directly under the YOLOX_train folder and save it there.  
https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth
```
benetech_1st_place_train/  
　├ YOLOX_train/  
　│　├ weights/  
　│　|  └ yolox_l.pth  
　│　├ ...   
　├ ...  
```


3. Run the training script  
`> python tools/train.py -f exps/example/custom/yolox_l_benetech_scatter_point_001_1280.py -d 1 -b 4 --fp16 -o -c ./weights/yolox_l.pth --cache`  

3. Results are output to `benetech_1st_place_train/YOLOX_train/YOLOX_outputs/yolox_l_benetech_scatter_point_001_1280`  

## Inference  
All scripts and processes required for inference are available in the [kaggle Notebook](https://www.kaggle.com/code/kashiwaba/benetech-1st-place-inference). Please refer there for details.  

## Pre-trained model
The weight files trained by the training code in this repository are also available on the kaggle platform, as are the inference scripts.  
1. classification model  
* [classification model link](https://www.kaggle.com/datasets/kashiwaba/benetech-classificationmodel)

2. Bar/Line/Dot model
* [horizontal_bar & line (All Chart-type Train) model link](https://www.kaggle.com/datasets/kashiwaba/benetech-allcharttrain-deplot)  
* [line (Specific Chart-type Train) model link](https://www.kaggle.com/datasets/kashiwaba/benetech-specificcharttrain-line-deplot)  
* [vertical_bar (Specific Chart-type Train) model link](https://www.kaggle.com/datasets/kashiwaba/benetech-specificcharttrain-verticalbar-deplot)  

3. Scatter model
* [label text reading model link](https://www.kaggle.com/datasets/kashiwaba/benetech-scatter-axis-reading)  
* [scatter detect model](https://www.kaggle.com/datasets/kashiwaba/benetech-scatter-point-detect-yolox)  

## Hardware  
I ran this source code in the following environment.  
* Ubuntu 22.04  
* intel Xeon(R) W-2223 CPU @ 3.60GHz x 8  
* NVIDIA GeForce RTX 3090  x1  
* CUDA11.7  