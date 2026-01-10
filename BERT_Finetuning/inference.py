"""
@file   : inference.py
@time   : 2025-12-27
"""
import json
import torch
import jieba
from model import Model
import pandas as pd
from tqdm import tqdm
from config import set_args
from torch.utils.data import DataLoader
from data_helper import MyDataset, Collater
from transformers.models.bert import BertTokenizer


if __name__ == '__main__':
    args = set_args()

    model = Model(num_classes=2)
    model.load_state_dict(torch.load('./output/best_model.bin', map_location='cpu'))

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    df = pd.read_csv("./data/test.csv")
    df = df[['id', 'text']]

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model)
    test_dataset = MyDataset(df, tokenizer, is_train=False)
    a = Collater(is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=Collater(is_train=False).collate_fn)


    all_predict = []
    for batch in tqdm(test_dataloader):
        if torch.cuda.is_available():
            batch = [t.cuda() for t in batch]
        input_ids, attention_mask, token_type_ids = batch
        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)

        _, pred = torch.max(logits, dim=1)
        pred = pred.cpu().detach().numpy().tolist()
        all_predict.extend(pred)

    submit_df = pd.DataFrame()
    submit_df['id'] = df['id']
    submit_df['target'] = all_predict
    submit_df.to_csv("./output/submit_df.csv", index=False)


