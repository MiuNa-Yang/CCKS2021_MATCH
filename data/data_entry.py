# -*- coding:utf-8 -*-
# @Time   : 2021/12/21 10:42
# @Author : xinhongyang
# @File   : data_entry
from data.TextDataset import TextDataset4train
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer


def select_dataloader(train_config):
    tokenizer = BertTokenizer.from_pretrained("./pretrained_model/nezha_base/vocab.txt")
    dataset = TextDataset4train(tokenizer, train_file_path=train_config.TRAIN_FILE_PATH, max_len=train_config.MAX_LEN)

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, train_config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, train_config.BATCH_SIZE, shuffle=True, drop_last=True)

    return train_loader, val_loader
