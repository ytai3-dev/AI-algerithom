"""
@file   : run_data_process.py
@time   : 2026-01-03
"""
import json
import pandas as pd
from collections import Counter


def build_vocab(news):
    all_vocab = []
    for x in news:
        vocab = list(x)   # '今日数据趣谈' => ['今', '日', '数', '据', ....]
        all_vocab.extend(vocab)

    vocab2count = dict(Counter(all_vocab))
    vocab2count = sorted(vocab2count.items(), key=lambda x: x[1], reverse=True)
    # print(vocab2count)   # [(v, c), (v, c), ...]
    # print(len(vocab2count))   # 4789

    vocab2id = {}
    vocab2id['PAD'] = 0
    vocab2id['UNK'] = 1
    vocab2id['START'] = 2
    vocab2id['END'] = 3

    i = 3
    for x in vocab2count:
        i += 1
        vocab2id[x[0]] = i
    return vocab2id


def save_data(data, save_path):
    data = [item.strip() for item in data]
    with open(save_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(data))

if __name__ == '__main__':
    df = pd.read_csv("./data/data.csv")
    news = df['news']

    # 构建词表
    vocab2id = build_vocab(news)

    train_data = news[:4500]
    test_data = news[-500:]

    save_data(train_data, save_path='./data/train_data.jsonl')   # json line
    save_data(test_data, save_path='./data/test_data.jsonl')
    json.dump(vocab2id, open("./data/vocab2id.json", 'w', encoding='utf8'), ensure_ascii=False)
