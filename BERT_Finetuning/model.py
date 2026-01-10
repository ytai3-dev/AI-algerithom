"""
@file   : model.py
@time   : 2025-12-27
"""
import torch
from torch import nn
from config import set_args
from transformers.models.bert import BertModel
from transformers import RobertaModel
args = set_args()



class Classify(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(Classify, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        out = self.linear3(x)
        return out


class Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(args.pretrain_model)  # 实现了bert 并且加载了它的权重
        # self.roberta = RobertaModel.from_pretrained(args.pretrain_model)

        self.output = Classify(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print(out)  #  一般会输出两个东西: (1) 把每个token的最后一层向量输出  (2) 把cls向量单独输出出来
        last_hidden_state = out.last_hidden_state
        cls_output = out.pooler_output   # base: (batch_size, 768)    large: (batch_size, 1024)
        # print(last_hidden_state.size())   # batch_size, max_len, hidden_size  # torch.Size([2, 21, 768])
        # print(cls_output.size())   # batch_size, hidden_size   # torch.Size([2, 768])

        logits = self.output(cls_output)
        return logits

