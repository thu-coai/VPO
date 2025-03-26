import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--begin', type=int)
parser.add_argument('--end', type=int)
parser.add_argument('--gpu', type=int)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import pandas as pd
from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer


# TODO query path
with open('query.json', encoding='utf-8') as f:
    data = json.load(f)[args.begin: args.end]

sys_prompt = """In this task, your goal is to expand the user's short query into a detailed and well-structured English prompt for generating short videos.

Please ensure that the generated video prompt adheres to the following principles:

1. **Harmless**: The prompt must be safe, respectful, and free from any harmful, offensive, or unethical content.  
2. **Aligned**: The prompt should fully preserve the user's intent, incorporating all relevant details from the original query while ensuring clarity and coherence.  
3. **Helpful for High-Quality Video Generation**: The prompt should be descriptive and vivid to facilitate high-quality video creation. Keep the scene feasible and well-suited for a brief duration, avoiding unnecessary complexity or unrealistic elements not mentioned in the query.

User Query:{}

Video Prompt:"""

tmp = [{'messages': [{'role': 'user', 'content': sys_prompt.format(i.strip())}]} for i in data]


# TODO change model path
model_path = ''

llm = LLM(model=model_path, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.padding_side = "left"
prompts = []

for i in tmp:
    try:
        prompts.append(tokenizer.apply_chat_template(i['messages'], add_generation_prompt=True, tokenize=True))
    except Exception as e:
        print(e)
        continue    
print("data numbers: ", len(prompts))
print(tokenizer.decode(prompts[0]))

sampling_params = SamplingParams(
    temperature=0.9, 
    top_p=0.9,
    max_tokens=2048,
    n=4,
    stop=['<|eot_id|>', '</s>']
)

outputs = llm.generate(prompt_token_ids=prompts, sampling_params=sampling_params)
# Print the outputs.
res = []
for output, i in zip(outputs, data):
    a = {
        'query': i
    }
    a['output'] = [output.outputs[j].text.strip() for j in range(4)]
    res.append(a)

with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)
