from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "qwen3-0.5b-pretrain"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "你是谁"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
# print(text)  # [151644, 872, 198, 105043, 100165, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
# print(text)  # 把用户输入的问题包装成一个规范的输入
'''
<|im_start|>user
你是谁<|im_end|>
<|im_start|>assistant
<think>

</think>
'''

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# print(model_inputs)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)  # <think> </think>
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

