import pandas as pd
import random
import json
from tqdm import tqdm
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
    vocab2id['MASK'] = 2
    vocab2id['CLS'] = 3

    i = 3
    for x in vocab2count:
        i += 1
        vocab2id[x[0]] = i
    return vocab2id


def rum_mask(data, save_path, dtype='train'):
    final_result = []
    for item in tqdm(data, desc=dtype):
        item = item[:510]
        result = {}
        vocab_list = list(item)  # ['', '', '', '', '' ,'']
        mask_num = int(len(vocab_list) * 0.3)
        mask_position = random.sample([i for i in range(len(vocab_list))], k=mask_num)
        mask_position.sort()
        mask_labels = []
        for pos in mask_position:
            label = vocab_list[pos]
            vocab_list[pos] = 'MASK'
            mask_labels.append(label)

        result['tokens'] = vocab_list
        result['mask_position'] = mask_position
        result['mask_labels'] = mask_labels
        s = json.dumps(result, ensure_ascii=False)
        # {'tokens': [xxx], 'mask_position': [xxx], 'mask_labels': [xxx]}
        final_result.append(s)
    with open(save_path, 'w', encoding='utf8') as f:  # "{k: v, k:v ,}"
        f.write('\n'.join(final_result))


if __name__ == '__main__':
    # MASK位置预测
    df = pd.read_csv("./data/data.csv")
    news = df['news'].tolist()
    print(len(news))   # 5000

    # 构建一个词表
    vocab2id = build_vocab(news)

    train_data = news[:4500]
    test_data = news[-500:]

    # mask
    rum_mask(train_data, save_path='./data/train_data.jsonl', dtype='train processing')
    rum_mask(test_data, save_path='./data/test_data.jsonl', dtype='test processing')

    json.dump(vocab2id, open("./data/vocab2id.json", 'w', encoding='utf8'), ensure_ascii=False)

