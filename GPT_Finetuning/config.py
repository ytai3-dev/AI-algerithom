"""
@file   : config.py
@time   : 2026-01-10
"""

import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='./data/xx.jsonl')
    parser.add_argument('--test_data_path', type=str, default='./data/xx.jsonl')
    parser.add_argument('--batch_size', type=str, default=2)
    return parser.parse_args()

