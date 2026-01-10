"""
@file   : config.py
@time   : 2025-12-27
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--train_data_path', type=str, default='./data/train_data.csv')
    parser.add_argument('--test_data_path', type=str, default='./data/test_data.csv')

    parser.add_argument('--pretrain_model', type=str, default='./bert_base_pretrain')
    parser.add_argument('--num_epochs', type=int, default=5, help='训练多少轮')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    return parser.parse_args()

