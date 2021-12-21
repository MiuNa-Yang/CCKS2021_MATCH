# -*- coding:utf-8 -*-
# @Time   : 2021/12/17 14:03
# @Author : xinhongyang
# @File   : nezhabase

import torch
from .NeZhaBERT import NeZhaPreTrainedModel, NeZhaModel
from torch import nn
from transformers import BertConfig


class NeZhaCLS(nn.Module):
    def __init__(self, bert_path):
        super(NeZhaCLS, self).__init__()
        self.num_labels = 3
        self.bert_config = BertConfig.from_pretrained(bert_path)
        self.bert = NeZhaModel(config=self.bert_config)
        self.fc_cls = nn.Linear(self.bert_config.hidden_size, self.num_labels)
        # self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                input_embeds=None,
                labels=None):

        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

        # 相当于[CLS]的输出
        pooled_output = bert_output[1]  # [1, hidden_size]
        logit = self.fc_cls(pooled_output)  # [1, num_labels]

        # 可以将预测和训练继承在一个Model里
        if labels is not None:  # 标签不为空，则为训练，可以直接在这里得到loss

            pass

        return logit

# 将两个句子拼接后，放入bert，然后通过CLS对标签进行预测。
# 这个任务会不会用文本相似度 + 标签 预测起来可以更加猛？
