"""
@file   : run_train.py
@time   : 2025-12-27
"""
import os
import torch
import json
import pandas as pd
from tqdm import tqdm
from torch import nn
# from model import Model
from torch.optim import AdamW
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from data_helper import MyDataset, Collater
from config import set_args
from model import Model
from transformers.models.bert import BertTokenizer

from transformers import RobertaTokenizer, RobertaModel
# from transformers import AutoModel, AutoTokenizer


def evaluate(model, test_dataloader):
    model.eval()   # 告诉模型 接下来要做验证
    all_predict = []
    all_target = []
    for batch in tqdm(test_dataloader):
        if torch.cuda.is_available():
            batch = [t.cuda() for t in batch]
        input_ids, attention_mask, token_type_ids, label = batch
        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)

        _, pred = torch.max(logits, dim=1)
        pred = pred.cpu().detach().numpy().tolist()
        label = label.cpu().detach().numpy().tolist()
        all_predict.extend(pred)
        all_target.extend(label)

    f1score = f1_score(all_target, all_predict)
    return f1score


if __name__ == '__main__':
    # 加载数据 -> 实现自己Dataset -> 实现DataLoader -> 写模型 -> 优化器、损失 -> 训练过程 -> 验证过程 -> 推理
    args = set_args()

    os.makedirs(args.output_dir, exist_ok=True)   # 创建output文件夹

    # step1: 加载数据
    train_df = pd.read_csv(args.train_data_path)
    test_df = pd.read_csv(args.test_data_path)
    print("训练集数量:", train_df.shape)   # 训练集数量: (6989, 2)
    print("测试集数量:", test_df.shape)   # 测试集数量: (777, 2)

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model)
    # step2: 实现Dataset
    train_dataset = MyDataset(train_df, tokenizer)
    test_dataset = MyDataset(test_df, tokenizer)
    # print(train_dataset[12])

    # step3: 实现dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Collater().collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=Collater().collate_fn)

    # step4: 实例化模型
    model = Model(num_classes=2)
    if torch.cuda.is_available():
        model.cuda()

    loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # step5: 优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)   # 优化器  更新梯度
    best_f1score = 0
    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.is_available():
                batch = [t.cuda() for t in batch]
            input_ids, attention_mask, token_type_ids, labels = batch
            # print(input_ids.size())   # (batch_size, max_len)
            # print(attention_mask.size())   #  (batch_size, max_len)
            # print(labels.size())   # (batch_size,)
            logits = model(input_ids, attention_mask, token_type_ids)   # 等价于 logits = model.forward(input_ids, attention_mask, token_type_ids)
            # print(logits.size())
            # 算损失
            loss = loss_func(logits, labels)
            optimizer.zero_grad()  # 清空当前优化器中梯度
            loss.backward()  # 反向求导 得到每个参数的梯度

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)  # 梯度裁剪
            optimizer.step()  # 把上一步梯度 更新到参数上
            print("epoch:{}, step:{}, loss:{}".format(epoch, step, loss.item()))  # loss.item(): 和梯度断开

        f1score = evaluate(model, test_dataloader)
        f =  open(args.output_dir + '/log.txt', 'a', encoding='utf8')
        f.write("epoch:{}, eval_f1_score:{}\n".format(epoch, round(f1score, 4)))
        f.close()

        if f1score > best_f1score:
            best_f1score = f1score
            torch.save(model.state_dict(), './output/best_model.bin')

