# -*- coding:utf-8 -*-
# @Time   : 2021/12/17 14:25
# @Author : xinhongyang
# @File   : TextDataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os


class TextDataset4pretrain(Dataset):
    def __init__(self, tokenizer, train_file_path, test_file_path, block_size):
        with open(train_file_path, encoding='utf-8') as f:
            train_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        with open(test_file_path, encoding='utf-8') as f:
            test_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        total_lines = train_lines + test_lines
        # print(total_lines)
        line_encodings = tokenizer(total_lines, add_special_tokens=True, truncation=True, max_length=block_size)

        self.examples = line_encodings["input_ids"]
        # self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]
        self.examples = [torch.tensor(e, dtype=torch.long) for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class TextDataset4train(Dataset):
    def __init__(self, tokenizer, train_file_path, max_len):
        input_ids = []
        input_mask = []
        input_seg = []
        labels = []
        with open(train_file_path, encoding="utf-8") as f:
            for line_ in f.read().splitlines():
                tmp_seq1, tmp_seq2, label = line_.split("\t")

                ids_1 = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(tmp_seq1) + ['[SEP]'])
                ids_2 = tokenizer.convert_tokens_to_ids( tokenizer.tokenize(tmp_seq2) + ['[SEP]'])
                ids = ids_1 + ids_2
                seg = [0] * len(ids_1) + [1] * len(ids_2)
                mask = [1] * len(ids)

                if len(ids) > max_len:
                    ids = ids[: max_len]
                    seg = seg[: max_len]
                    mask = mask[: max_len]

                while len(ids) < max_len:
                    ids.append(0)
                    seg.append(0)
                    mask.append(0)

                # label_tmp = [0, 0, 0]
                # label_tmp[int(label)] = 1
                # labels.append(label_tmp)

                labels.append(int(label))
                input_ids.append(ids)
                input_mask.append(mask)
                input_seg.append(seg)

        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.input_mask = torch.tensor(input_mask, dtype=torch.long)
        self.input_seg = torch.tensor(input_seg, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.input_mask[idx], self.input_seg[idx], self.labels[idx]


if __name__ == "__main__":
    config = {
        "train_file_path": "../resources/train.txt",
        "test_file_path": "../resources/test.txt",
        "vocab_file": "../pretrained_model/nezha_wwn/vocab.txt"
    }

    # Tokenzier = BertTokenizer.from_pretrained(config["vocab_file"])
    # dataset = TextDataset4pretrain(Tokenzier, config["train_file_path"], config["test_file_path"], block_size=100)
    #
    # dataloader = DataLoader(dataset, batch_size=32, drop_last=True)
    #
    # for e in dataloader:
    #     print(e)
    #     break
    #
    # tokenizer = BertTokenizer.from_pretrained(config["vocab_file"])
    with open("../resources/train.txt", encoding='utf-8') as f:
        # train_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        for line in f.read().splitlines():
            print(line.split("\t"))
            break
    #
    # with open("../resources/test.txt", encoding='utf-8') as f:
    #     test_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    #
    # total_lines = train_lines + test_lines
    # line_encodings = tokenizer(total_lines, add_special_tokens=True, truncation=True, max_length=100)
    #
    # examples = line_encodings["input_ids"]
    # examples = [torch.tensor(e, dtype=torch.long) for e in examples]
    # print(examples)
    # print(train_lines[0])
