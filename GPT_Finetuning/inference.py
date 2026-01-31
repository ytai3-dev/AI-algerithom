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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = "./output/finetune_model_epoch_9"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = get_model(checkpoint, tokenizer)
    model.to(device)

    model.eval()
    while True:
        text = input("User: ")
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids = [tokenizer.bos_token_id]+ input_ids + [tokenizer.sep_token_id]
        input_ids = torch.tensor([input_ids]).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, max_length=100, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("Bot:", output_text)
