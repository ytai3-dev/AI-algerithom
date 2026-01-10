"""
@file   : config.py
@time   : 2026-01-10
"""

import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='./data/train_data.json')
    parser.add_argument('--test_data_path', type=str, default='./data/test_data.json')
    parser.add_argument('--pretrain_model', type=str, default='./gpt2_pretrain')
    parser.add_argument('--batch_size', type=int, default=2)

    return parser.parse_args()

