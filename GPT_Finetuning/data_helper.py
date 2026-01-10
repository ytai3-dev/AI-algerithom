"""
@file   : data_helper.py
@time   : 2026-01-10
"""
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        cur_data = self.data[item]
        question = cur_data['instruction']
        answer = cur_data['response']

        # print(cur_data)
        # {'instruction': '你眼中的户晨风先生是个什么样的人？', 'response': '户晨风先生是一位观点犀利且幽默的评论员，他的评论常常能够一针见血地指出问题的核心，同时又不失风趣，让人在思考的同时也能会心一笑。'}
        # '你是谁'   "我是户晨风"
        # [START]  你  是  谁  [SEP]  我  是  户  晨  风  [END]
        #    0     0   0   0    1   1    1   1  1   1    0
        #                       我   是  户  晨  风 [END]

        # [START]  你  是  谁  [SEP]  我  是  户  晨  风
        #    你  是  谁  [SEP]  我  是  户  晨  风  [END]

        question_input_ids = self.tokenizer.encode(question, add_special_tokens=False)
        answer_input_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        question_input_ids = [self.tokenizer.bos_token_id]+ question_input_ids + [self.tokenizer.sep_token_id]
        loss_mask = [0] * (len(question_input_ids) - 1)
        answer_input_ids = answer_input_ids + [self.tokenizer.eos_token_id]
        loss_mask = loss_mask + [1] * len(answer_input_ids) + [0]
        input_ids = question_input_ids + answer_input_ids
        return {"input_ids": input_ids, 'loss_mask': loss_mask}



def padding_to_max_len(x, max_len, padding_value=0):
    if len(x) >= max_len:
        x = x[:max_len]
    else:
        x = x + (max_len - len(x)) * [0]
    return x


def collate_fn(batch):
    # batch: [{"input_ids": input_ids, 'loss_mask': loss_mask}, {"input_ids": input_ids, 'loss_mask': loss_mask},...]
    max_len = max([len(item['input_ids']) for item in batch])
    if max_len > 1024:
        max_len = 1024

    all_input_ids = []
    all_attention_mask = []
    all_loss_mask = []

    # max_len = 8
    # input_ids = [32, 12, 53, 2]
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = [1] * len(input_ids)
        attention_mask = padding_to_max_len(attention_mask, max_len)
        input_ids = padding_to_max_len(input_ids, max_len)
        loss_mask = item['loss_mask']
        loss_mask = padding_to_max_len(loss_mask, max_len)
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_loss_mask.append(loss_mask)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    loss_mask = torch.tensor(all_loss_mask, dtype=torch.long)
    return input_ids, attention_mask, loss_mask

