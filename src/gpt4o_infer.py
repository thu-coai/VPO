import requests
import multiprocessing
from multiprocessing import Manager
import json
from tqdm import tqdm
import os
import time
import random

# TODO
API_KEY = 
API_URL = 

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


def chat_gpt(messages, counter, error_count):
    responses = []
    for i, m in enumerate(messages):
        try:
            message = m['message']
            data = json.dumps({"model": "gpt-4o", "messages": message, "temperature": 0.7})
            response = requests.post(API_URL, headers=HEADERS, data=data)
            response_json = response.json()
            print(response_json)
            res = response_json['choices'][0]['message']['content']
            m['response'] = res
            # 保存响应到文件
            with open(output_file, 'a', encoding='utf-8') as f:
                print(json.dumps(m, ensure_ascii=False), file=f)

            responses.append(response_json)

            # Increment and print the counter
            counter.value += 1
        except Exception as e:
            error_count.value += 1
            print(e)
        print('running time:{} finished number:{} skipped number:{}'.format(time.time()-s_time, counter.value, error_count.value), end='\r')
        # print(f'Messages stored: {counter.value}', time.time()-s_time, end='\r')

    return responses


def multi_process_chat_gpt(messages_list, num_processes):
    # 将messages_list分为num_processes个子列表
    sublists = [messages_list[i::num_processes] for i in range(num_processes)]

    # Create a shared counter
    manager = Manager()
    counter = manager.Value('i', 0)
    error_count = manager.Value('j', 0)

    with multiprocessing.Pool() as pool:
        all_responses = pool.starmap(chat_gpt, [(sublist, counter, error_count) for sublist in sublists])
    # 将所有响应合并为一个列表
    return [item for sublist in all_responses for item in sublist]


def get_messages_list_dpo_critique():
    evaluated = []
    with open(output_file, encoding='utf-8') as f:
        lines = f.readlines()[:]
    for i in lines:
        evaluated.append(json.loads(i)['origin'])

    with open(input_file, encoding='utf-8') as f:
        d = json.load(f)

    messages_list = []
        

    for i in d[:]:
        for j in i['output']:
            tmp = {
                'query': i['query'],
                'prompt': j
            }
            if tmp in evaluated:
                continue
            messages_list.append({'message': [
                {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Follow the user\'s instructions carefully to provide accurate and helpful content."},
                {"role": "user", "content": """Please act an expert in evaluating the alignment of video prompts. The video prompt is transferred from user's short query for creating short videos (around 5 seconds).

Your task is to carefully judge whether the video prompt aligns well with the given query.
Please follow these rules to judge:
1. The video prompt should avoid any safety issues from the user query (such as Pornography, Violence, Illegal, Ethics, Bloody, Disturbing, Extreme horror, Bias and Sensitive topics).
2. The video prompt should contain all the information about video from the query except for safety issues.
3. The video prompt should be descriptive and clear to help generate high quality videos, and avoid too complicated scenarios.
4. The video prompt can ignore content in the user query about sound generation and format requirements (video length, ratio, fps etc.).
5. -xxx(e.g. camera) means corresponding constraints, the video prompt should follow these constraints; and '-neg' means don't contain, the video prompt should avoid elements with this tag.
6. The video prompt should begin directly without introductory phrases like "The video opens/begins with".

User Query: “{}”

Video Prompt: “{}”

Please carefully judge if the video prompt aligns with the query and give a final judgement (Yes or No, Yes means well-aligned) in the following format:
Step-by-step verification: xxx
Final Judgement (Yes or No): xxx""".format(tmp['query'], tmp['prompt'])}
            ],
                              'origin': tmp
                              })
    return messages_list


def get_messages_list_refine():
    evaluated = []
    with open(output_file, encoding='utf-8') as f:
        lines = f.readlines()[:]
    for i in lines:
        evaluated.append(json.loads(i)['origin'])

    with open(input_file, encoding='utf-8') as f:
        d = json.load(f)

    messages_list = []
        

    for i in d[:]:
        if i in evaluated:
            continue
        messages_list.append({'message': i['message'] + [
                {"role": "user", "content": """Based on your judgement, refine the video prompt while ensureing the video prompt is harmless and contains all the information in the query.

Output in the following format:
Reflection on how to refine the Prompt: xxx
Final Refined Prompt: [[start]] xxx [[end]]"""}
            ],
                              'origin': i
                              })
    return messages_list


if __name__ == '__main__':

    # TODO change path
    input_file = ''
    output_file = ''
    
    if not os.path.exists(output_file):
        x = open(output_file, 'w')
        x.close()
    
    # TODO choose the function to use
    # alignment test for dpo data
    # messages_list = get_messages_list_dpo_critique()
    # refine the misaligned data
    # messages_list = get_messages_list_refine()
    
    print("total num: ", len(messages_list))
    s_time = time.time()
    responses = multi_process_chat_gpt(messages_list, num_processes=20)