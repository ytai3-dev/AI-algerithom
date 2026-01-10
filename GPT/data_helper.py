"""
@file   : data_helper.py
@time   : 2026-01-03
"""
import torch
from torch.utils.data import Dataset


def load_data(path):
    all_data = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            all_data.append(line)
    return all_data



class MyDataset(Dataset):
    def __init__(self, data, vocab2id):
        self.data = data
        self.vocab2id = vocab2id
        self.max_len = 510

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item]
        text = text[:self.max_len]

        # 转成id
        input_ids = []
        input_ids.append(self.vocab2id['START'])
        for v in text:
            idx = self.vocab2id.get(v, self.vocab2id['UNK'])
            input_ids.append(idx)
        input_ids.append(self.vocab2id['END'])
        return input_ids


def padding_to_max(input_ids, max_len, padding_value=0):
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + (max_len-len(input_ids)) * [padding_value]
    return input_ids


def collate_fn(batch):
    # print(batch)   # [[x, x, ,x, ,x ], [x,x, ,x,x,,x,x], [input_ids], [input_ids], [input_ids]]
    max_len = max([len(item) for item in batch])

    input_ids_list = []
    for item in batch:
        item = padding_to_max(item, max_len=max_len)
        input_ids_list.append(item)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    return input_ids

