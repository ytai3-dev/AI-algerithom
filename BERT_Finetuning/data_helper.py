"""
@file   : data_helper.py
@time   : 2025-12-27
"""
import jieba
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self,dataframe, tokenizer, is_train=True):
        self.reviews=dataframe['text'].tolist()
        self.tokenizer = tokenizer

        self.is_train = is_train
        if self.is_train:
            self.labels=dataframe['target'].tolist()

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review=self.reviews[item]
        review=str(review)

        data = self.tokenizer(review)
        # print(data)
        # input_ids
        # attention_mask
        # token_type_ids

        if self.is_train:
            label=self.labels[item]
            data['label'] = label
            return data
        else:
            return data


class Collater:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def pad_to_max_len(self, input_ids, max_len, padding_value=0):
        if len(input_ids)<max_len:
            input_ids=input_ids+(max_len-len(input_ids))*[padding_value]
        else:
            input_ids=input_ids[:max_len]
        return input_ids

    def collate_fn(self, batch):
        # batch: [{input_ids:xxx, attention_mask:xx, token_type_ids:xx, label: 0}, {}, {}]
        # 1. 计算 batch 内最大长度
        lengths = [len(item['input_ids']) for item in batch]
        max_len = max(lengths)

        # 2. 限制最大长度
        if max_len > 512:
            max_len = 512

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_labels = []

        # 3. padding
        for item in batch:
            input_ids = self.pad_to_max_len(item['input_ids'], max_len)
            attention_mask = self.pad_to_max_len(item['attention_mask'], max_len)
            token_type_ids = self.pad_to_max_len(item['token_type_ids'], max_len)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)

            if self.is_train:
                label = item['label']
                all_labels.append(label)

        input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)

        # 4. 转 tensor
        if self.is_train:
            labels = torch.tensor(all_labels, dtype=torch.long)
            return input_ids, attention_mask, token_type_ids, labels
        else:
            return input_ids, attention_mask, token_type_ids

