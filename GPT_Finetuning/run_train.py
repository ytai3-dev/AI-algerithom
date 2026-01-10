"""
@file   : run_train.py
@time   : 2026-01-10
"""
import json
from config import set_args
from data_helper import MyDataset, collate_fn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer

# 加载数据 -> Dataset -> DataLoader -> 模型 -> 损失 -> 优化器  ->  训练过程 -> 验证过程 -> 推理



if __name__ == '__main__':
    args = set_args()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)

    # 1. 加载数据
    train_data = json.load(open(args.train_data_path, 'r', encoding='utf8'))
    test_data = json.load(open(args.test_data_path, 'r', encoding='utf8'))

    # 2. 实现Dataset
    train_dataset = MyDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    for batch in train_dataloader:
        input_ids, attention_mask, loss_mask = batch
        print(input_ids.size())
        print(attention_mask.size())
        print(loss_mask.size())
        exit()





