"""
@file   : run_train.py
@time   : 2026-01-10
"""
import os
import json
import torch
from tqdm import tqdm
from torch.optim import AdamW
from config import set_args
from data_helper import MyDataset, collate_fn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer



# 加载数据 -> Dataset -> DataLoader -> 模型 -> 损失 -> 优化器  ->  训练过程 -> 验证过程 -> 推理
def get_model(path, tokenizer):
    model = GPT2LMHeadModel.from_pretrained(path, pad_token_id=tokenizer.eos_token_id)
    return model


def evaluate(model, test_dataloader, tokenizer):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids, attention_mask, loss_mask = batch
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            loss_mask = loss_mask.to(model.device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            label = input_ids[:, 1:].contiguous()
            loss_mask = loss_mask[:, 1:].contiguous()
            logits = out.logits[:, :-1, :].contiguous()
            loss_func = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits.view(-1, logits.size(-1)), label.view(-1))
            loss = loss * loss_mask.view(-1)
            total_loss += loss.sum().item()
    test_loss =  total_loss / len(test_dataloader.dataset)
    return test_loss


if __name__ == '__main__':
    args = set_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)

    # 1. 加载数据
    train_data = json.load(open(args.train_data_path, 'r', encoding='utf8'))
    test_data = json.load(open(args.test_data_path, 'r', encoding='utf8'))

    
    # 2. 实现Dataset
    train_dataset = MyDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = MyDataset(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.pretrain_model, tokenizer)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    # 总的训练步数
    total_step = len(train_dataloader) * args.Num_Epochs
    print("Total training steps:", total_step)

    global_step = 0
    for epoch in range(args.Num_Epochs):
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            input_ids, attention_mask, loss_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            loss_mask = loss_mask.to(device)

            # [START]  你  是  谁  [SEP]  我  是  户  晨  风  [END]
            #    0     0   0   0    1   1    1   1  1   1    0
            #                       我   是  户  晨  风 [END]
            # print(input_ids.size())  # torch.Size([2, 130])
            # print(attention_mask.size())  # torch.Size([2, 130])
            # print(loss_mask.size())  # torch.Size([2, 130])
            out = model(input_ids=input_ids, attention_mask=attention_mask)

            label = input_ids[:, 1:].contiguous()    # batch_size, max_len-1
            loss_mask = loss_mask[:, 1:].contiguous()   # batch_size, max_len-1
            logits = out.logits[:, :-1, :].contiguous()  # batch_size, max_len-1, vocab_size

            # 计算loss
            loss = loss_func(logits.view(-1, logits.size(-1)), label.view(-1))
            loss = loss * loss_mask.view(-1)  
            loss = loss.sum() / (loss_mask.sum() + 1e-5)
            print("global_step:{}, epoch:{}, step:{}, loss:{}".format(global_step, epoch, step, loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 验证
        test_loss = evaluate(model, test_dataloader, tokenizer)
        with open(f'./{args.output_dir}/test_loss.txt', 'a', encoding='utf8') as f:
            f.write(f'Epoch {epoch}: Test Loss: {test_loss}\n')
        
        # 保存模型
        model.save_pretrained(f'./{args.output_dir}/finetune_model_epoch_{epoch}')
        tokenizer.save_pretrained(f'./{args.output_dir}/finetune_model_epoch_{epoch}')


                







