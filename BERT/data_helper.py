"""
@file   : data_helper.py
@time   : 2025-12-31
"""
import torch
import jieba
import json
from torch.utils.data import Dataset


def load_data(path):
    all_data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # '{k:v, k:v, ...}'
            line = json.loads(line)
            all_data.append(line)
    return all_data



class MyDataset(Dataset):
    def __init__(self, data, vocab2id):
        self.data = data  # [{'tokens': [xxx], 'mask_position': [xxx], 'mask_labels': [xxx]}, ...]
        self.vocab2id = vocab2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        cur_data = self.data[item]

        tokens = cur_data['tokens']
        mask_position = cur_data['mask_position']
        mask_labels = cur_data['mask_labels']

        input_ids = []
        input_ids.append(self.vocab2id['CLS'])  # CLS .... SEP ... SEP
        for x in tokens:
            temp = self.vocab2id.get(x, self.vocab2id['UNK'])
            input_ids.append(temp)

        # mask_position = [3, 6, 19, 30]   =》 [4, 7, 20, 31]
        new_mask_position = [i+1 for i in mask_position]

        new_mask_labels = []
        for x in mask_labels:
            temp = self.vocab2id.get(x, self.vocab2id['UNK'])
            new_mask_labels.append(temp)
        return {"input_ids": input_ids, 'mask_position': new_mask_position, 'mask_label': new_mask_labels}


def padding_to_maxlen(l1, max_len, padding_value=0):
    if len(l1) < max_len:
        l1 = l1 + [padding_value] * (max_len - len(l1))
    else:
        l1 = l1[:max_len]
    return l1


def collate_fn(batch):
    # padding + tensor
    input_ids_max_len = max([len(b['input_ids']) for b in batch])   # [32, 28, ...]
    mask_max_len = max([len(b['mask_position']) for b in batch])   # [32, 28, ...]

    all_input_ids = []
    all_mask_position = []
    all_mask_label = []
    # 对数据三部分进行padding
    for b in batch:
        input_ids = padding_to_maxlen(b['input_ids'], input_ids_max_len, padding_value=0)
        mask_label = padding_to_maxlen(b['mask_label'], mask_max_len, padding_value=0)
        mask_position = padding_to_maxlen(b['mask_position'], mask_max_len, padding_value=0)
        all_input_ids.append(input_ids)
        all_mask_label.append(mask_label)
        all_mask_position.append(mask_position)

    # 转tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_mask_label = torch.tensor(all_mask_label, dtype=torch.long)
    all_mask_position = torch.tensor(all_mask_position, dtype=torch.long)
    return all_input_ids, all_mask_position, all_mask_label





