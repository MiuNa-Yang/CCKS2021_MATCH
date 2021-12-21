# -*- coding:utf-8 -*-
# @Time   : 2021/12/17 12:09
# @Author : xinhongyang
# @File   : options
import os
import torch


class Train_Config(object):
    BERT_PATH = "./pretrained_model/nezha_base"
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    BATCH_SIZE = 64
    TRAIN_FILE_PATH = "./resources/train.txt"
    MAX_LEN = 64
    LEARNING_RATE = 1e-7
    EPOCH = 3


train_config = Train_Config()
