"""
@file   : run_train.py
@time   : 2025-12-31
"""
import torch
import json
from torch import nn
from config import set_args
from model import BERT
from torch.utils.data import DataLoader
from data_helper import load_data, MyDataset, collate_fn
from tqdm import tqdm

import swanlab   # pip install swanlab
swanlab.login(api_key="xx", save=True)

def evalute(model, test_dataloader):
    model.eval()
    total_num = 0
    correct_num = 0
    for batch in tqdm(test_dataloader):
        input_ids, mask_position, mask_label = batch
        logits = model(input_ids, mask_position)
        # print(res.size())  # batch_size, mask_max_len, vocab_size
        logits = logits.view(-1, logits.size(-1))  # (batch_size*mask_max_len, vocab_size)

        _, pred = torch.max(logits, dim=1)   # [[0.3, 0.2, 0.3], [0.3, 0.4, 0.1]]
        pred_label = pred.numpy().tolist()
        true_label = mask_label.numpy().tolist()

        for p, t in zip(pred_label, true_label):
            if t != 0:
                total_num += 1
                if p == t:
                    correct_num += 1
    accuracy = correct_num / total_num
    return round(accuracy, 5)


if __name__ == '__main__':
    # 配置config.py -> 加载数据 -> 实现自己Dataset -> 实现DataLoader -> 写模型 -> 优化器、损失 -> 训练过程 -> 验证过程 -> 推理
    # 1. 配置config.py
    args = set_args()

    swanlab.init(
        # 设置项目
        project="theo-bert",
        experiment_name='bert',
        # 跟踪超参数与实验元数据
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size
        }
    )

    # 2. 加载数据
    train_data = load_data(args.train_data_path)
    test_data = load_data(args.test_data_path)
    test_data = test_data[:10]
    vocab2id = json.load(open(args.vocab2id_path, 'r', encoding='utf8'))

    # 3. 实现自己Dataset
    # 4. 实现自己DataLoader    batch -> padding -> tensor
    train_dataset = MyDataset(train_data, vocab2id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataset = MyDataset(test_data, vocab2id)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = BERT(len(vocab2id))

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    global_step = 0
    best_acc = 0
    for epoch in range(args.num_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            global_step += 1
            input_ids, mask_position, mask_label = batch
            logits = model(input_ids, mask_position)
            # print(res.size())  # batch_size, mask_max_len, vocab_size
            logits = logits.view(-1, logits.size(-1))    # (batch_size*mask_max_len, vocab_size)
            mask_label = mask_label.view(-1)  # (batch_size*mask_max_len,)
            loss = loss_func(logits, mask_label)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            print("epoch:{}, step:{}, loss:{}".format(epoch, step, loss))
            swanlab.log({"loss": loss.item()}, step=global_step)

        test_acc = evalute(model, test_dataloader)
        swanlab.log({"accuracy": test_acc}, step=global_step)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), './output/best_model.bin')


