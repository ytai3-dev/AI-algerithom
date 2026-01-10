"""
@file   : run_train.py
@time   : 2026-01-03
"""
import torch
import json
from config import set_args
from model import GPT
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from data_helper import load_data, MyDataset, collate_fn


def calc_loss(logits, labels):
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    # print(logits.size())  # batch_size, max_len, vocab_size
    # print(labels.size())   # batch_size, max_len

    batch_size, max_len, vocab_size = logits.size()
    logits = logits.view(-1, vocab_size)  #
    labels = labels.reshape(-1)
    # print(logits.size())   # (2000, vocab_size)
    # print(labels.size())   # (2000,)
    loss = loss_func(logits, labels)
    return loss


def evaluate(model, test_dataloader):
    model.eval()

    total_num = 0
    correct_num = 0
    for batch in tqdm(test_dataloader):
        # print(batch.size())   # (batch_size, max_len)
        # [START] A B C E F [END]
        # 输入: [START]  A  B  C  E  F
        # 标签:    A     B  C  E  F [END]
        input_ids = batch[:, :-1]  # 包含第一位
        labels = batch[:, 1:]  # 从第二位
        logits = model(input_ids)
        # print(logits.size())   # batch_size, max_len, vocab_size
        # print(labels.size())   # batch_size, max_len
        batch_size, max_len, vocab_size = logits.size()
        logits = logits.view(-1, vocab_size)  #  (2000, vocab_size)
        labels = labels.reshape(-1)     # (2000,)

        _, pred = torch.max(logits, dim=1)

        pred = pred.cpu().detach().numpy().tolist()
        labels = labels.cpu().detach().numpy().tolist()

        for t, p in zip(labels, pred):
            if t != 0:
                total_num += 1
                if t == p:
                    correct_num += 1
    acc = correct_num / total_num
    return acc


if __name__ == '__main__':
    # 1. 写配置文件
    args = set_args()

    # 2. 加载数据
    train_data = load_data(args.train_data_path)
    test_data = load_data(args.test_data_path)
    vocab2id = json.load(open(args.vocab2id_path, 'r', encoding='utf8'))

    # 3. 实现dataset
    # 4. 实现Dataloader
    train_dataset = MyDataset(train_data, vocab2id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = MyDataset(test_data, vocab2id)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 5. 实现模型
    model = GPT(len(vocab2id))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)  # 优化器  更新梯度
    best_acc = 0
    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # print(batch.size())   # (batch_size, max_len)
            # [START] A B C E F [END]
            # 输入: [START]  A  B  C  E  F
            # 标签:    A     B  C  E  F [END]
            input_ids = batch[:, :-1]   # 包含第一位
            labels = batch[:, 1:]   # 从第二位
            logits = model(input_ids)
            # print(logits.size())   # batch_size, max_len, vocab_size
            # print(labels.size())   # batch_size, max_len
            loss = calc_loss(logits, labels)

            optimizer.zero_grad()  # 清空当前优化器中梯度
            loss.backward()  # 反向求导 得到每个参数的梯度

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)  # 梯度裁剪
            optimizer.step()  # 把上一步梯度 更新到参数上
            print("epoch:{}, step:{}, loss:{}".format(epoch, step, loss.item()))  # loss.item(): 和梯度断开

        test_acc = evaluate(model, test_dataloader)
        f = open(args.output_dir + '/log.txt', 'a', encoding='utf8')
        f.write("epoch:{}, test_acc:{}\n".format(epoch, round(test_acc, 4)))
        f.close()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), './output/best_model.bin')

