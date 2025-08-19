import copy
import json
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForMultipleChoice, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaTokenizerFast, LlamaForCausalLM, LlamaForSequenceClassification
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
# import zhconv
# from accelerate import Accelerator, infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.corpus import words
import torch.multiprocessing as mp
from cfg import Config
from dataloader import *
import re
import string
from sklearn.metrics import accuracy_score, f1_score
import traceback
from collections import defaultdict
from scorer import tfidf_scorer
from pesudo_generater import *

# 辅助函数，检查字符是否可打印
def printable(char):
    return char in string.printable

def clean_text(text):
    # 去除非打印字符
    text = ''.join(c for c in text if printable(c))
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除或替换其他特殊字符（例如，将所有数字替换为#）
    # text = re.sub(r'\d+', '#', text)

    # 去除或替换Unicode字符（这里只是示例，通常不建议这样做）
    # 注意：这可能会删除所有非ASCII字符，通常不是最佳选择
    # text = ''.join(c for c in text if ord(c) < 128)
    # 去除前导和尾随空格
    text = text.strip()

    return text

def load_base_model_tokenizer(cfg):
    if 'gpt2' in cfg.LLM:
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path, trust_remote_code=True, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(cfg.LLM_path, trust_remote_code=True)
    if 'Mistral' in cfg.LLM:
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path, trust_remote_code=True, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(cfg.LLM_path, trust_remote_code=True)
    # if 'opt' in cfg.LLM:
    #     tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path, trust_remote_code=True, padding_side='left')
    #     tokenizer.pad_token = ' '
    #     model = OPTForCausalLM.from_pretrained(cfg.LLM_path, trust_remote_code=True)

    if 'llama3' in cfg.LLM.lower():
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path, trust_remote_code=True,
                                                   padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(cfg.LLM_path, trust_remote_code=True)
        # model = torch.load(cfg.LLM_path+'/consolidated.00.pth', map_location='cpu')
        # print(model)
        # torch.save(model['state_dict'], cfg.LLM_path+'/pytorch_model.bin')
        # exit(111)

    if 'qwen' in cfg.LLM.lower():
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path, trust_remote_code=True, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(cfg.LLM_path, trust_remote_code=True)

    if 'deepseek' in cfg.LLM:
        model = AutoModelForCausalLM.from_pretrained(cfg.LLM_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path, trust_remote_code=True, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.config.pad_token_id = tokenizer.config.eos_token_id


    model = model.half().cuda()
    print("(*^_^*) model load finished on {}!!!! ".format(model.device))
    model.eval()
    return model, tokenizer

def generate_prompt(cfg, context_shots, test_sample):
    '''
        :param testset:    test dataset
        :return:
        '''
    Instructions = {
        'SA': 'Solve the sentiment analysis task. Options for sentiment: negative, positive, neutral. \n',
        'TD': 'Solve the toxic detection task. Options for toxicity: benign, toxic. \n',
        'NLI': 'Solve the NLI task. Options for entailment relationship: entailment, neutral, contradiction. \n',
        'NER': 'Solve the NER task, identifying the organization, person, location entities from given text. \n'
    }
    prompt = Instructions[cfg.task]
    for shot in context_shots:
        if cfg.task == 'NLI':
            prompt = prompt + '{} Prediction: {}\n'.format(' '.join(shot[0].replace('\n', '').split()[:cfg.max_sen_len]), shot[1])
        elif cfg.task == 'NER':
            prompt = prompt + 'Text: {}\nEntity: {}\n'.format(' '.join(shot[0].replace('\n', '').split()[:cfg.max_sen_len]), shot[1])
        else:
            prompt = prompt + 'Text: {} Prediction: {}\n'.format(' '.join(shot[0].replace('\n', '').split()[:cfg.max_sen_len]), shot[1])
    test_text = ' '.join(test_sample[0].split(' ')[:cfg.max_sen_len])
    if cfg.task == 'NLI':
        prompt = prompt + '{} Prediction: '.format(test_text)
    elif cfg.task == 'NER':
        prompt = prompt + 'Text: {}\nEntity: '.format(test_text)
    else:
        prompt = prompt + 'Text: {} Prediction: '.format(test_text)
    prompt = clean_text(prompt)
    return prompt

def get_batch(cfg, testset):
    '''
    :param testset:    test dataset
    :return:
    '''
    # load train&test set
    ICL_datas, ICL_labels = [], []
    IID_datas, OOD_datas = load_datasets(cfg)
    OOD_datas, IID_datas = OOD_datas[testset], IID_datas[testset]
    # 加载context样本
    if cfg.if_sim_first:
        file = open('ICL_samples/{}/{}/{}_sim.json'.format(cfg.score_func, cfg.task, testset), 'r', encoding='utf-8')
    else:
        file = open('ICL_samples/{}/{}/{}_unsim.json'.format(cfg.score_func, cfg.task, testset), 'r', encoding='utf-8')
    context_samples = json.load(file)
    assert len(context_samples) == len(OOD_datas)
    for test_sample, context in zip(OOD_datas, context_samples):
        context_shots = []
        for i in range(cfg.shots):
            if cfg.task == 'NER':
                context_shots.append(IID_datas[context[i]])
            else:
                for label in context.keys():
                    text = IID_datas[context[label][i]]
                    try:
                        text[0] = ' '.join(text[0].split()[:80])
                        context_shots.append(text)
                    except:
                        continue
        # 根据context_shots、test_sample以及对应的instructions构建ICL的prompt
        prompt = generate_prompt(cfg, context_shots, test_sample)
        ICL_datas.append(prompt)
        ICL_labels.append(test_sample[1])

    # 分batch
    batches = []
    for i in range(0, len(ICL_labels), cfg.batch):
        batch = ICL_datas[i:i + cfg.batch]
        batch_labels = ICL_labels[i:i + cfg.batch]  # 标签
        batches.append([batch, batch_labels])
    print('process datasets ({}) with batch ({}) ......'.format(len(ICL_labels), len(batches)))
    return batches


def select_random_numbers(lst, idx, n=1):
    filtered_list = [x for i, x in enumerate(lst) if i not in idx]
    # 随机选择n个不重复的数
    selected_numbers = random.sample(filtered_list, n)
    return selected_numbers[0]

def get_batch_from_selected_samples(cfg, selected_idx, pred_labels, testset):
    ''' 从迭代过程中选择出来的样本中随机地构建batch数据
    :param selected_idx:
    :param pred_labels:
    :param testset:
    :return: batches，和get_batch一样的batches结果
    '''
    id2label = cfg.label_space[cfg.task]
    ICL_datas, ICL_labels = [], []
    # 首先加载testset对应的数据集
    test_data = []
    if cfg.task != 'NER':
        test_frame = pd.read_csv('datasets/{}/{}/test.tsv'.format(cfg.task, testset), sep='	')
        if cfg.task == 'NLI':
            t_premises, t_hypothesis, t_labels = list(test_frame['Premise'].values), list(test_frame['Hypothesis'].values), list(test_frame['Label'].values)
            for i in range(len(t_premises)):
                test_data.append(['Premise: ' + t_premises[i] + ' Hypothesis: ' + t_hypothesis[i], id2label[t_labels[i]]])
        else:
            for text, label in zip(list(test_frame['Text'].values), list(test_frame['Label'].values)):
                test_data.append([text, id2label[label]])
    else:
        test_texts, test_tags = load_ner_by_space('datasets/NER/{}/sample.tsv'.format(testset))
        for i in range(len(test_texts)):
            label_str = ''
            label = test_tags[i]
            for k in label.keys():
                if len(label[k]) > 0:
                    for entity in label[k]:
                        label_str = label_str + k + ':' + entity + ';'
            test_data.append([test_texts[i], label_str])

    OOD_datas = test_data[:5000]
    # 将标签映射到对应的样本，并进行few-shot选取
    label2idx = defaultdict(list)
    for idx, label in zip(selected_idx, pred_labels):   # 将selected_idx中的索引映射到pred_labels中的标签
        label2idx[label].append(idx)

    # 利用word2vec将selected_idx以及OOD_datas映射到向量空间并计算相似度并选择每个OOD_data中top-shots相似的样本进行ICL
    selexted_texts = [test_data[id][0] for id in selected_idx]
    OOD_texts = [each[0] for each in OOD_datas]
    # bert_model = AutoModel.from_pretrained('/root/autodl-tmp/models/SimCSE/').cuda()
    # bert_tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/SimCSE/')
    # sim_matrix = bert_scorer(selexted_texts, OOD_texts, bert_model, bert_tokenizer).T
    sim_matrix = tfidf_scorer(selexted_texts, OOD_texts).T
    print(sim_matrix.shape)

    for i, test_sample in enumerate(OOD_datas):
        sorted_idx = np.argsort(sim_matrix[i, :])[::-1]
        if cfg.task == 'SA':
            # 遍历sorted_idx从中寻找每个类别最相似的样本
            sims = {"negative": [], "positive": [], "neutral": []}
            for idx in sorted_idx:
                if len(sims["negative"]) >= cfg.shots and len(sims["positive"]) >= cfg.shots and len(sims["neutral"]) >= cfg.shots:
                    break
                if len(sims[pred_labels[idx]]) < cfg.shots and selected_idx[idx] != i:   # 说明不等于当前的数据
                # if len(sims[pred_labels[idx]]) < cfg.shots:   # 说明不等于当前的数据
                    sims[pred_labels[idx]].append(selected_idx[idx])

        if cfg.task == 'TD':
            # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
            sims = {"benign": [], "toxic": []}
            for idx in sorted_idx:
                if len(sims["benign"]) >= cfg.shots and len(sims["toxic"]) >= cfg.shots:
                    break
                if len(sims[pred_labels[idx]]) < cfg.shots and selected_idx[idx] != i:   # 说明不等于当前的数据
                # if len(sims[pred_labels[idx]]) < cfg.shots:   # 说明不等于当前的数据
                    sims[pred_labels[idx]].append(selected_idx[idx])

        if cfg.task == 'NLI':
            # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
            sims = {"entailment": [], "neutral": [], "contradiction":[]}
            for idx in sorted_idx:
                if len(sims["entailment"]) >= cfg.shots and len(sims["neutral"]) >= cfg.shots and len(sims['contradiction']) >= cfg.shots:
                    break
                if len(sims[pred_labels[idx]]) < cfg.shots and selected_idx[idx] != i:   # 说明不等于当前的数据
                # if len(sims[pred_labels[idx]]) < cfg.shots:   # 说明不等于当前的数据
                    sims[pred_labels[idx]].append(selected_idx[idx])

        context_datas = []
        if cfg.task != 'NER':
            for shot in range(cfg.shots):
                for label in sims.keys():
                    try:
                        context_datas.append([test_data[sims[label][shot]][0], label])
                    except:
                        continue
        else:
            for iii in sorted_idx[:cfg.shots]:
                # context_datas.append([test_data[iii][0], pred_labels[iii]])
                context_datas.append(test_data[iii])
        prompt = generate_prompt(cfg, context_datas, test_sample)
        ICL_datas.append(prompt)
        ICL_labels.append(test_sample[1])
    # 分batch
    batches = []
    for i in range(0, len(ICL_labels), cfg.batch):
        batch = ICL_datas[i:i + cfg.batch]
        batch_labels = ICL_labels[i:i + cfg.batch]  # 标签
        batches.append([batch, batch_labels])
    print('process datasets ({}) with batch ({}) ......'.format(len(ICL_labels), len(batches)))
    return batches

def extract_entity_from_output(text):
    pattern = r"(organization|person|location):\s*([^;\n]+)[;\n]?"
    order = ['organization', 'person', 'location']
    if 'organization' in text or 'person' in text or 'location' in text:
        matches = re.findall(pattern, text)
        matches = list(set(matches))
        matches_sorted = sorted(matches, key=lambda x: order.index(x[0]))
        label = []
        for each in matches_sorted:
            if each[1] in text:  # 保证生成样本的标签是属于
                label.append(each[0] + ':' + each[1])
        label = ';'.join(label) + ';'
    else:
        label = ';'
    return label

def ICL_ner_acc(preds, labels):
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
        label = [each for each in label if '' != each]
        pred = [each for each in pred if '' != each]
        set_pred = set(pred)
        set_label = set(label)
        # 计算交集和并集
        intersection = set_pred.intersection(set_label)
        union = set_pred.union(set_label)
        correct_count += len(intersection)
        total_count += len(union)
    return correct_count/total_count

def ICL(cfg, test, model, tokenizer):
    print('load LLM {}......'.format(cfg.LLM))
    batches = get_batch(cfg, test)
    preds, labels = [], []
    for batch in tqdm(batches):
        try:
            inputs = tokenizer(batch[0], return_tensors="pt", padding=True)
            input_ids = inputs.input_ids
            with torch.no_grad():
                if cfg.task != 'NER':
                    generate_ids = model.generate(input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(), pad_token_id=0, max_new_tokens=5)
                else:
                    generate_ids = model.generate(input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(), pad_token_id=0, max_new_tokens=30)
                output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except Exception as e:
            continue
        if cfg.task != 'NER':
            for out, inp in zip(output, batch[0]):
                out = out.replace(inp, '').lower()
                if cfg.task == 'TD':
                    if cfg.label_space[cfg.task][0] in out:
                        preds.append(0)
                    elif cfg.label_space[cfg.task][1] in out:
                        preds.append(1)
                    else:
                        preds.append(100)
                else:
                    if cfg.label_space[cfg.task][0] in out:
                        preds.append(0)
                    elif cfg.label_space[cfg.task][1] in out:
                        preds.append(1)
                    elif cfg.label_space[cfg.task][2] in out:
                        preds.append(2)
                    else:
                        preds.append(100)
            swapped_label_space = {value: key for key, value in cfg.label_space[cfg.task].items()}
            labels.extend([swapped_label_space[each] for each in batch[1]])
        else:
            for out, inp in zip(output, batch[0]):
                out = out.replace(inp, '')
                out = out.split('\n')[0].strip()
                preds.append(out)
            labels.extend(batch[1])
    if cfg.task == 'NER':
        acc = ICL_ner_acc(preds, labels)
    else:
        swapped_label_space = {value: key for key, value in cfg.label_space[cfg.task].items()}
        labels = [swapped_label_space[each] for each in labels]
        acc = accuracy_score(preds, labels)
    print('===================final acc:{}============='.format(acc))


if __name__ == '__main__':
    cfg = Config()
    for model_name in ['llama3.2-3b', 'llama3.1-8b', 'Qwen2.5-3b', 'Mistral-7B-v0.3']:
        cfg.batch = 20
        cfg.LLM = model_name
        cfg.LLM_path = '/root/autodl-tmp/models/{}'.format(model_name)
        model, tokenizer = load_base_model_tokenizer(cfg)
        for task in ['NER']:
            cfg.task = task
            for score_func in ['random', 'tfidf', 'bert']:
            # for score_func in ['random']:
                for sim in [True]:
                    cfg.score_func = score_func
                    cfg.if_sim_first = sim
                    if cfg.task == 'TD':
                        for dataset in ['implicit_hate', 'toxigen']:
                            print('==========={}-{}-{}-{}========'.format(model_name, dataset, score_func, sim))
                            ICL(cfg, dataset, model, tokenizer)
                    if cfg.task == 'SA':
                        for dataset in ['dynasent', 'semeval', 'sst5']:
                            print('==========={}-{}-{}-{}========'.format(model_name, dataset, score_func, sim))
                            ICL(cfg, dataset, model, tokenizer)
                    if cfg.task == 'NLI':
                        for dataset in ['anli', 'contract_nli', 'wanli']:
                            print('==========={}-{}-{}-{}========'.format(model_name, dataset, score_func, sim))
                            ICL(cfg, dataset, model, tokenizer)
                    if cfg.task == 'NER':
                        for dataset in ['conll', 'ener', 'wnut']:
                            print('==========={}-{}-{}-{}========'.format(model_name, dataset, score_func, sim))
                            ICL(cfg, dataset, model, tokenizer)
        del model
