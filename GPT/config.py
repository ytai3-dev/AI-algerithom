"""
@file   : config.py
@time   : 2026-01-03
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='./data/train_data.jsonl', help='训练集')
    parser.add_argument('--test_data_path', type=str, default='./data/test_data.jsonl', help='测试集')
    parser.add_argument('--vocab2id_path', type=str, default='./data/vocab2id.json', help='词表')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练多少轮')
    parser.add_argument('--learning_rate', type=float, default=3e-3, help='学习率')

    return parser.parse_args()
