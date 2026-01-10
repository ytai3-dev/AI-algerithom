"""
@file   : model.py
@time   : 2026-01-03
"""
import torch
import numpy as np
from torch import nn


'''
# 我爱北京天安门
# [START] 我 爱 北 京 天 安  门 [END]
#   我    爱 北 京 天 安 门  [END]

# 问答
# 你  是  谁 
# v1  v2  v3
# p1  p2  p3
#         我

# 你  是   谁   我
# v1  v2   v3  v4
# p1  p2   p3  p4
#              是

# 你  是   谁   我  是
#                  v5
#                  p5
#                  张

# 你  是   谁   我  是  张
#                     v6
#                     p6
# ......
'''


def get_attention_mask_with_padding(q, k):
    batch_size, q_len = q.size()
    batch_size, k_len = k.size()

    # k: batch_size, max_len
    res = torch.eq(k, 0).unsqueeze(1)   # batch_size, 1, max_len
    res = res.expand((batch_size, q_len, k_len))
    return res

def get_sub_sequence_mask(seq):
    # print(seq.size())   # (batch_size, max_len)
    batch_size, seq_len = seq.size()

    temp = torch.ones(seq_len, seq_len)   # np.ones(5, 5)
    temp = torch.triu(temp, diagonal=1)   # (max_len, max_len)
    # (1, max_len, max_len)
    mask = temp.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class Embedding(nn.Module):
    def __init__(self, vocab_size, max_position):
        super(Embedding, self).__init__()
        embedding_dim = 768
        self.vocab_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_position, embedding_dim)
        self.layernorm = nn.LayerNorm(embedding_dim)

    def forward(self, input_ids):
        # 1. 词嵌入
        v_embed = self.vocab_embedding(input_ids)
        # print(v_embed.size())   # torch.Size([2, 98, 768])

        # 2. 位置嵌入
        # print(v_embed.size())   # torch.Size([2, 98, 768])
        seq_len = v_embed.size(1)
        position = torch.arange(seq_len, dtype=torch.long)

        batch_size = input_ids.size(0)
        position = position.unsqueeze(0).repeat(batch_size, 1)
        p_embed = self.position_embedding(position)
        # print(p_embed.size())   # torch.Size([2, 98, 768])

        embed = v_embed + p_embed
        # print(embed.size())   # torch.Size([2, 98, 768])

        embed = self.layernorm(embed)
        return embed



class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.head_num = 12
        self.head_dim = 64
        self.WQ = nn.Linear(768, self.head_dim*self.head_num)
        self.WK = nn.Linear(768, self.head_dim*self.head_num)
        self.WV = nn.Linear(768, self.head_dim*self.head_num)
        self.softmax = nn.Softmax(dim=-1)
        self.layernorm = nn.LayerNorm(768)


    def forward(self, Q, K, V, attention_mask):
        batch_size = Q.size(0)
        # # (2, 98, 768) => (2, 98, 12, 64) batch_size, max_len, head_num, head_dim => (2, 12, 98, 64)
        # batch_size, head_num, max_len, head_dim
        q = self.WQ(Q).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
        k = self.WK(K).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
        v = self.WV(V).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)

        # (2, 12, 98, 64) * (2, 12, 64, 98) => (2, 12, 98, 98)
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)

        # print(scores.size())   # batch_size, head_num, max_len, max_len
        # print(attention_mask.size())   # batch_size, max_len, max_len
        # batch_size, max_len, max_len => batch_size, 1, max_len, max_len => batch_size, 12, max_len, max_len
        mask = attention_mask.unsqueeze(1).repeat(1, self.head_num, 1, 1)
        scores.masked_fill_(mask, -1e9)   # 原地操作   masked_fill_ 根据mask的true false填充score
        scores = torch.softmax(scores, dim=-1)


        # (2, 12, 98, 98) * (2, 12, 98, 64) => (2, 12, 98, 64)
        out = torch.matmul(scores, v)   # torch.Size([2, 12, 98, 64])
        out = out.transpose(1, 2).contiguous()
        # print(out.size())   # torch.Size([2, 98, 12, 64])
        out = out.view(batch_size, -1, self.head_dim * self.head_num)
        # print(out.size())   # torch.Size([2, 98, 768])
        return self.layernorm(out)


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(768, 2048)
        self.linear2 = nn.Linear(2048, 768)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # print(out.size())   # torch.Size([2, 98, 768])
        return out


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super(GPT, self).__init__()
        max_position = 512
        self.embedding = Embedding(vocab_size, max_position=max_position)
        self.multihead_attention = MultiHeadAttention()
        self.feedforword = FeedForward()
        self.classify_head = nn.Linear(768, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)   #

        padding_mask = get_attention_mask_with_padding(input_ids, input_ids)  # padding mask
        subseq_mask = get_sub_sequence_mask(input_ids)

        mask = (padding_mask + subseq_mask)
        mask = torch.ge(mask, 1)   # great equal

        for i in range(12):
            out = self.multihead_attention(x, x, x, mask)
            # print(out.size())   # torch.Size([2, 98, 768])
            x = self.feedforword(out)

        # print(x.size())   # torch.Size([4, 511, 768])
        logits = self.classify_head(x)
        # print(logits)   # batch_size, max_len, vocab_size
        return logits

