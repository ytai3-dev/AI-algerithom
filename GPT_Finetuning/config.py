"""
@file   : config.py
@time   : 2026-01-10
"""

import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./output")

    parser.add_argument('--train_data_path', type=str, default='./data/train_data.json')
    parser.add_argument('--test_data_path', type=str, default='./data/test_data.json')
    parser.add_argument('--pretrain_model', type=str, default='./gpt2_pretrain')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--Num_Epochs', type=int, default=10)
    return parser.parse_args()

