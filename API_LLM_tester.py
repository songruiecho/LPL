import requests
import json

from scipy.signal.windows import blackman
from sympy.codegen.tests.test_algorithms import wurlitzer

from LLM_tester import *
import time


def chat(prompt):
    url = "https://api.xi-ai.cn/v1/chat/completions"
    api_key = [
        'sk-xvu9m3Oxwo9RSt167718D2D1Fb464902BdF138C1F388C749',
        'sk-HMjMaMLCYxpscmYoC3019dFe884149B6977066562bA5Cd4b',
        'sk-9UOnq9q2aseQrOldBb0aFdBb3b7342A89881587c0e54B9A8',
        'sk-4vlbSD6DlZFYjMtXE1A3772f35A440769a2761B72b42395f'
    ]
    random_key = random.choice(api_key)
    payload = json.dumps({
        # "model": "gpt-3.5-turbo",
        "model": "gpt-4o-mini",
        # "model": "qwen2-72b-instruct",
        "messages": [
            {
               "role": "user",
               "content": prompt
            },

        ],
        "max_tokens": 30,    # 修改最大长度
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Bearer {}'.format(random_key),
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response = json.loads(response.text)
    try:
        text = response['choices'][0]['message']['content']
    except:
        print(response)
        text = ''
    return text

def main(cfg, testset):
    id2label = cfg.label_space[cfg.task]
    with open('filtered_samples/llama3.2-3b_{}_CPP_norm.txt'.format(testset), 'r') as wf:
        filtered_samples = [each.strip().split('\t') for each in wf.readlines()]
        ids = [int(each[0]) for each in filtered_samples]
        labels = [each[1] for each in filtered_samples]
    try:
        with open('LLM_results/{}_3.5.txt'.format(testset), 'r') as rf:
            iddd = len(rf.readlines())
    except:
        iddd = 0
    batches = get_batch(cfg, testset)[:500]
    with open('LLM_results/{}_3.5.txt'.format(testset), 'a') as wf:
        for i in tqdm(range(iddd, len(batches))):
            text = batches[i][0][0]
            label = batches[i][1][0]
            try:
                response = chat(text)
                out = response.strip().lower()
                wf.write(label+ '\t' + out + '\n')
            except Exception as e:
                print(f"[Error at sample {i}] {e}")
                wf.write(label + '\t' + 'ERROR' + '\n')


def main2(cfg, testset):
    id2label = cfg.label_space[cfg.task]
    with open('filtered_samples/llama3.1-8b_{}_CPP_norm.txt'.format(testset), 'r') as rf:
        filtered_samples = [each.strip().split('\t') for each in rf.readlines()]
        ids = [int(each[0]) for each in filtered_samples]
        if cfg.task == 'NER':
            labels = [each[1] for each in filtered_samples]
        else:
            labels = [id2label[int(each[1])] for each in filtered_samples]
        CPP_scores = [each[3] for each in filtered_samples]
        filtered_ids_labels = [(ids[i], labels[i]) for i in range(len(CPP_scores)) if float(CPP_scores[i]) > cfg.model2confidence]
        ids, labels = zip(*filtered_ids_labels) if filtered_ids_labels else ([], [])
    batches = get_batch_from_selected_samples(cfg, ids, labels, testset)[:200]
    # for batch in tqdm(batches):
    #     print(batch[0][0])
    #     print(batch[1][0])
    # exit(111)
    try:
        with open('LLM_results/{}_LPL.txt'.format(testset), 'r') as rf:
            iddd = len(rf.readlines())
    except:
        iddd = 0
    with open('LLM_results/{}_LPL.txt'.format(testset), 'a') as wf:
        for i in tqdm(range(iddd, len(batches))):
            text = batches[i][0][0]
            label = batches[i][1][0]
            try:
                response = chat(text)
                out = response.strip().lower()
                wf.write(label+ '\t' + out + '\n')
            except Exception as e:
                print(f"[Error at sample {i}] {e}")
                wf.write(label + '\t' + 'ERROR' + '\n')

def cal_accuracy(cfg, testset):
    right_counts, right_counts2 = 0, 0
    with open('LLM_results/{}_3.5.txt'.format(testset), 'r') as rf:
        for each in rf.readlines():
            each = each.strip().split('\t')
            try:
                label = each[0]
                preds = each[1]
            except:
                continue
            if label in preds:
                right_counts += 1
    with open('LLM_results/{}_LPL_3.5.txt'.format(testset), 'r') as rf:
        for each in rf.readlines():
            each = each.strip().split('\t')
            if len(each) == 2:
                label = each[0]
                preds = each[1]
                if label in preds:
                    right_counts2 += 1
    print('base model {}; LPL {}'.format(round(right_counts/500, 4), round(right_counts2/500, 4)))

def ner_acc(preds, labels):
    # organization:person: Grete Stroem; location: Norway
    '''
    :param preds:
    :param labels:
    :return:
    '''
    # 解析预测标签和真实标签
    parsed_preds = [pred.lower().split(';') for pred in preds]
    parsed_labels = [label.lower().split(';') for label in labels]
    correct_count, total_count = 0, 1e-8
    for pred, label in zip(parsed_preds,parsed_labels):
        # 将列表转换为集合
        label = [each.strip() for each in label if '' != each]
        pred = [each.strip() for each in pred if '' != each and ':;' not in each]
        set_pred = set(pred)
        set_label = set(label)
        # 计算交集和并集
        intersection = set_pred.intersection(set_label)
        union = set_pred.union(set_label)
        correct_count += len(intersection)
        total_count += len(union)
    return correct_count/total_count


def cal_ner_accuracy(cfg, testset):
    preds1, labels1 = [], []
    preds2, labels2 = [], []
    # with open('LLM_results/{}.txt'.format(testset), 'r') as rf:
    #     for each in rf.readlines():
    #         each = each.strip().replace('entity:', '').split('\t')
    #         if len(each) == 2:
    #             print(each)
    #             labels1.append(each[0].lower())
    #             preds1.append(each[1].lower())
    with open('LLM_results/{}_LPL.txt'.format(testset), 'r') as rf:
        for each in rf.readlines():
            each = each.strip().replace('entity:', '').split('\t')
            if len(each) == 2:
                labels2.append(each[0].lower())
                preds2.append(each[1].lower())
    # acc1 = ner_acc(preds1, labels1)
    acc2 = ner_acc(preds2, labels2)
    print('base model {}; LPL {}'.format(0, acc2))

if __name__ == '__main__':
    cfg = Config()
    cfg.batch = 1
    cfg.task = 'TD'
    # main(cfg, 'conll')
    # main2(cfg, 'conll')
    cal_accuracy(cfg, 'implicit_hate')
    # main2(cfg, 'ener')
    cal_accuracy(cfg, 'toxigen')
    # main(cfg, 'contract_nli')
    # main(cfg, 'wanli')
    # main2(cfg, 'anli')
    # main2(cfg, 'contract_nli')
    # main2(cfg, 'wanli')
    # cal_accuracy(cfg, 'anli')
    # cal_accuracy(cfg, 'contract_nli')
    # cal_accuracy(cfg, 'wanli')