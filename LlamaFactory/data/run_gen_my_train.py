# step1: 将数据转成人家规定的格式
# step2: 在data中dataset_info.json配置数据的路径
#   "huchengfeng": {
#     "file_name": "huchengfeng.json"
#   },
# step3: 在examples中写一个自己yaml文件
  
import json

if __name__ == '__main__':
    all_data = []
    with open('/root/autodl-tmp/Qwen3_fintuning/data/question_answer.jsonl', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = json.loads(line)
            messages = line['messages']
            question = messages[0]['content']
            answer = messages[1]['content']

            item = {
                "instruction": question,
                "input": "",
                "output": answer
            }
            all_data.append(item)
    json.dump(all_data, open('./huchengfeng.json', 'w', encoding='utf8'), ensure_ascii=False)

    

