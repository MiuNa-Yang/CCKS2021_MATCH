# -*- coding:utf-8 -*-
# @Time   : 2021/12/17 14:03
# @Author : xinhongyang
# @File   : nezhabase

import torch
from .NeZhaBERT import NeZhaPreTrainedModel, NeZhaModel
from torch import nn
from transformers import BertConfig


class NeZhaCLS(NeZhaPreTrainedModel):
    def __init__(self, config):
        super(NeZhaPreTrainedModel, self).__init__(config)
        self.num_labels = 3

        self.bert = NeZhaModel(config=self.bert_config)
        self.fc_cls = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

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
                                token_type_ids=token_type_ids,
                                head_mask=head_mask,
                                input_embeds=input_embeds)

        pooled_output = bert_output[1]

        logit = self.fc_cls(pooled_output)

        # outputs = (logit,) + pooled_output[2:]
        #
        # if labels is not None:
        #     # loss_fct = LabelSmoothingLoss(smoothing=0.01)
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(logit.view(-1, self.num_labels), labels.view(-1))
        #     outputs = (loss,) + outputs

        return logit
