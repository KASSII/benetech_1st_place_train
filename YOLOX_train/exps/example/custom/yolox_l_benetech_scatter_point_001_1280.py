#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "../datas/input/dataset_for_yolox/scatter_detection_dataset001"
        self.train_ann = "train_annotations.json"
        self.val_ann = "val_annotations.json"

        self.num_classes = 1

        self.max_epoch = 50
        self.data_num_workers = 0
        self.eval_interval = 1
        self.input_size = (1280, 1280)
        self.test_size = (1280,1280)        
        self.save_history_ckpt = False
