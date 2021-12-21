# -*- coding:utf-8 -*-
# @Time   : 2021/12/17 12:07
# @Author : xinhongyang
# @File   : train

import torch
from models.nezhabase import NeZhaCLS
from options import train_config
from data.data_entry import select_dataloader
from transformers import AdamW, get_cosine_schedule_with_warmup


class Trainer:
    def __init__(self, train_config_):
        self.config = train_config_
        self.model = NeZhaCLS(bert_path=self.config.BERT_PATH)
        self.device = self.config.DEVICE
        self.train_dataloader, self.val_dataloader = select_dataloader(self.config)
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=1e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train_per_epoch(self, epoch):

        self.model.train()

        for idx, sample in enumerate(self.train_dataloader):

            input_ids = sample[0].to(self.device)
            input_mask = sample[1].to(self.device)
            input_seg = sample[2].to(self.device)
            labels = sample[3].to(self.device)

            # [batch_size, num_labels]
            logit = self.model(input_ids=input_ids,
                               attention_mask=input_mask,
                               token_type_ids=input_seg,
                               labels=labels)

            # loss写在外面吧
            # labels [batch_size, num_labels]
            loss = self.loss_fn(logit, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print('Train: Epoch {} batch {} Loss {}'.format(epoch, idx, loss))

    def val_per_epoch(self, epoch):

        self.model.eval()

        val_loss = 0.0

        for idx, sample in enumerate(self.val_dataloader):

            input_ids = sample[0].to(self.device)
            input_mask = sample[1].to(self.device)
            input_seg = sample[2].to(self.device)
            labels = sample[3].to(self.device)

            # [batch_size, num_labels]
            logit = self.model(input_ids=input_ids,
                               attention_mask=input_mask,
                               token_type_ids=input_seg,
                               labels=labels)

            # loss写在外面吧
            # labels [batch_size, num_labels]
            val_loss += self.loss_fn(logit, labels)

        print('Val: Epoch {} Loss {}'.format(epoch, val_loss / (len(self.val_dataloader * self.config.BATCH_SIZE))))

    def train(self):

        for e in range(1, self.config.EPOCH):

            self.train_per_epoch(e)
            self.val_per_epoch(e)


def main():
    trainer = Trainer(train_config)
    trainer.train()


if __name__ == "__main__":
    main()
