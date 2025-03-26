import json
import random

def process_gen_data():
    l = []
    for i in range(8):
        # TODO change path
        with open(f'vllm_output_{i}.json', encoding='utf-8') as f:
            l.extend(json.load(f))
    
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(l, f, indent=4, ensure_ascii=False)
    
def process_critique_data():
    with open('4o_critique_res.jsonl', encoding='utf-8') as f:
        l = f.readlines()

    bad = []
    for i in l:
        i = json.loads(i)
        if 'Final Judgement (Yes or No)' not in i['response']:
            continue
        if i['response'].split('Final Judgement (Yes or No)')[1].count('Yes'):
            pass
        else:
            bad.append({
                'query': i['origin']['query'],
                'prompt': i['origin']['prompt'],
                'message': i['message'] + [{'role': 'assistant', 'content': i['response']}],
                'refine': True
            })

    # TODO change path
    with open('', 'w', encoding='utf-8') as f:
        json.dump(bad, f, indent=4, ensure_ascii=False)


def process_dpo_data():
    # TODO: change VisionReward output path
    with open('/output.jsonl', encoding='utf-8') as f:
        l = f.readlines()
    # data example
    """
    {
        'internal_id': 'video_0_0',
        'query': 'angry elderly people screaming at potatoes in the supermarket, cinematic look, chaotic composition, insane details, dramatic light, photorealistic --ar 16:9',
        'prompt': "In a dimly lit supermarket aisle, elderly individuals with furrowed brows and visibly agitated expressions confront a stack of potatoes on a shelf. Their faces contort with frustration as they point and scream at the potatoes, their voices carrying a sense of absurdity and chaos. The cinematic composition captures dramatic lighting, with harsh shadows and intense beams of light illuminating the scene, creating a surreal and tense atmosphere. The chaotic arrangement of shelves and scattered produce in the background emphasizes the disorder, while the photorealistic details of the individuals' wrinkles, clothing, and angry expressions heighten the drama.",
        'QA': [{'id': 1, 'q': 'Does the video meet all the requirements stated in the text "angry elderly people screaming at potatoes in the supermarket, cinematic look, chaotic composition, insane details, dramatic light, photorealistic --ar 16:9"?', 'response': 'no'}, ......],
        'video_path': '000000.mp4'
    }
    """
    res = {}
    for i in l:
        i = json.loads(i)
        idx = int(i['internal_id'].split('_')[1]) // 4
        if idx not in res:
            res[idx] = []
        res[idx].append(i)

    def get_score(qa):
        array = [0.9543901856422174, 0.25239747290239256, 0.0, 1.141818673357406, 0.03495652038170829, 0.025237463294006605, 0.0, 0.0, 0.0, 0.12600844108184325, 0.0, 0.0, 0.0, 0.03221505988621183, 0.16286819641189937, 0.21673935360893115, 0.01970324496671629, 0.13604019362894557, 0.09647134683834487, 0.15490927135496332, 0.1294164598219855, 0.09891696198970226, 0.18839328668539077, 0.1844335421380767, 0.0, 0.0, 0.2635526157239052, 0.0, 0.0, 0.0, 0.0, 0.11168980468489233, 0.0, 0.05173789659242723, 0.0, 0.02562797122879315, 0.0, 0.4389890596048526, 0.0, 0.0, 0.0, 0.0, 0.0, 0.26857694964769424, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42925171836383774, 0.0, 0.00846154228462919, 0.12757277121689847, 0.05798205026065391, 0.0, 0.1446334304609205, 0.0, 0.0, 0.0, 0.39418111694677266, 0.0, 0.0, 0.0, 0.0]
        l = []
        for i in qa:
            if i['response'] == 'yes':
                l.append(1)
            else:
                l.append(0)
        assert (len(l)==len(array))
        return sum(a * b for a, b in zip(l, array))

    def compare_str(res1, res2):
        if res1 == res2:
            return 0
        if res1 == 'yes':
            return 1
        else:
            return -1

    def compare_fn_strict(qa_1, qa_2):
        flag = True
        num = 0
        for i in range(len(qa_1)):
            assert (qa_1[i]['id'] == qa_2[i]['id'])
            c = compare_str(qa_1[i]['response'], qa_2[i]['response'])
            if c == 1:
                num += 1
            elif c == -1:
                flag = False
                break
            
        return flag and (num > 0)
    
    num = 0
    data = []
    for k in res:
        num += 1
        add = False
        for i in res[k]:
            for j in res[k]:
                if i != j and compare_fn_strict(i['QA'], j['QA']) and (get_score(i['QA']) - get_score(j['QA']) > 0.5):
                    data.append({
                        'query': i['query'],
                        'chosen_prompt': i['prompt'],
                        'rejected_prompt': j['prompt'],
                        'score': [get_score(i['QA']), get_score(j['QA'])]
                    })
                    add = True
                    break
            if add:
                break
    
    sys_prompt = """In this task, your goal is to expand the user's short query into a detailed and well-structured English prompt for generating short videos.

Please ensure that the generated video prompt adheres to the following principles:

1. **Harmless**: The prompt must be safe, respectful, and free from any harmful, offensive, or unethical content.  
2. **Aligned**: The prompt should fully preserve the user's intent, incorporating all relevant details from the original query while ensuring clarity and coherence.  
3. **Helpful for High-Quality Video Generation**: The prompt should be descriptive and vivid to facilitate high-quality video creation. Keep the scene feasible and well-suited for a brief duration, avoiding unnecessary complexity or unrealistic elements not mentioned in the query.

User Query:{}

Video Prompt:"""
    d = []
    for i in data:
        d.append({
            "messages": [
                {
                    "role": "user",
                    "content": sys_prompt.format(i['query'])
                }
            ],
            "chosen": {
                "role": "assistant",
                "content": i['chosen_prompt'].strip()
            },
            "rejected": {
                "role": "assistant",
                "content": i['rejected_prompt'].strip()
            }
        })
    
    # TODO change refined data path
    with open('refined.jsonl', encoding='utf-8') as f:
        l = f.readlines()
    bad_res = []
    for i in l:
        i = json.loads(i)
        tmp = i['origin']
        if i['response'].count('[[start]]') and i['response'].count('[[end]]'):
            tmp['refine_cot'] = i['response']
            tmp['refine_res'] = i['response'].split('[[start]]')[1].split('[[end]]')[0].strip()
            bad_res.append(tmp)
    
    random.shuffle(bad_res)
    tmp = []
    q = []
    for i in bad_res:
        if i['query'] in q:
            continue
        tmp.append(i)
        q.append(i['query'])
    
    d_refine = []
    for i in tmp:
        if i['refine_res'].count('\n'):
            print(i)
            continue
        d_refine.append({
            "messages": [
                {
                    "role": "user",
                    "content": sys_prompt.format(i['query'])
                }
            ],
            "chosen": {
                "role": "assistant",
                "content": i['refine_res'].strip()
            },
            "rejected": {
                "role": "assistant",
                "content": i['prompt'].strip()
            }
        })
    
    
    # merge text-level dpo pair and video-level dpo pair
    
    tmp = []
    q = []
    rep = []
    for i in d_refine+d:
        if i['messages'] in q:
            rep.append(i['messages'])
            continue
        tmp.append(i)
        q.append(i['messages'])
    
    random.shuffle(tmp)
    with open('dpo_data.json', 'w', encoding='utf-8') as f:
        json.dump(tmp, f, indent=4, ensure_ascii=False)

if __name__ = '__main__':
    # concat generated data
    process_gen_data()
    # process 4o generated critique to find misaligned data for refinement
    process_critique_data()
    # process the output from VisionReward
    process_dpo_data()