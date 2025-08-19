import os

from cfg import Config
import pandas as pd
from scorer import *
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm
import json
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from sklearn.utils import resample
from pesudo_generater import *

# 转换numpy.int64为标准的int
def convert_to_builtin_types(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    raise TypeError(f'Not serializable: {repr(obj)}')

def load_datasets(arg):
    id2label = arg.label_space[arg.task]
    test_datas, train_datas = {}, {}
    for dataset in arg.test_datasets[arg.task]:
        test_datas[dataset] = ''
        train_datas[dataset] = ''

    if arg.task == 'SA':
        train_data = []
        train_frame = pd.read_csv('datasets/{}/amazon/train.tsv'.format(arg.task), sep='	').fillna(' ')
        for text, label in zip(list(train_frame['Text'].values), list(train_frame['Label'].values)):
            train_data.append([text, id2label[label]])
        for key in test_datas.keys():
            test_data = []
            test_frame = pd.read_csv('datasets/SA/{}/test.tsv'.format(key), sep='	').fillna(' ')
            for text, label in zip(list(test_frame['Text'].values), list(test_frame['Label'].values)):
                test_data.append([text, id2label[label]])
            test_datas[key] = test_data[:5000]
            train_datas[key] = train_data

    if arg.task == 'TD':
        train_frame = pd.read_csv('datasets/TD/civil_comments/train.tsv', sep='	')
        train_data = []
        for text, label in zip(list(train_frame['Text'].values), list(train_frame['Label'].values)):
            train_data.append([text, id2label[label]])
        for key in test_datas.keys():
            test_data = []
            test_frame = pd.read_csv('datasets/TD/{}/test.tsv'.format(key), sep='	', on_bad_lines='skip')
            for text, label in zip(list(test_frame['Text'].values), list(test_frame['Label'].values)):
                test_data.append([text, id2label[label]])
            test_datas[key] = test_data[:5000]
            train_datas[key] = train_data

    if arg.task == 'NLI':
        train_frame = pd.read_csv('datasets/{}/mnli/train.tsv'.format(arg.task), sep='	')
        train_data = []
        premises, hypothesis, labels = list(train_frame['Premise'].values), list(train_frame['Hypothesis'].values), list(train_frame['Label'].values)
        for i in range(len(premises)):
            train_data.append([premises[i]+'###'+hypothesis[i], id2label[labels[i]]])
        for key in test_datas.keys():
            test_data = []
            test_frame = pd.read_csv('datasets/{}/{}/test.tsv'.format(arg.task, key), sep='	')
            t_premises, t_hypothesis, t_labels = list(test_frame['Premise'].values), list(
                test_frame['Hypothesis'].values), list(test_frame['Label'].values)
            for i in range(len(t_premises)):
                test_data.append([t_premises[i] + '###' + t_hypothesis[i], id2label[t_labels[i]]])
            test_datas[key] = test_data[:5000]
            train_datas[key] = train_data
    if arg.task == 'NER':
        train_data = []
        train_texts, train_tags = load_ner_by_space('datasets/NER/{}/train.tsv'.format(arg.source_datasets['NER']))
        for i in range(len(train_texts)):
            label_str = ''
            label = train_tags[i]
            for k in label.keys():
                if len(label[k]) > 0:
                    for entity in label[k]:
                        label_str = label_str + k + ':' + entity + ';'
            train_data.append([train_texts[i], label_str])
        for key in test_datas.keys():
            test_data = []
            test_texts, text_tags = load_ner_by_space('datasets/{}/{}/sample.tsv'.format(arg.task, key))
            for i in range(len(test_texts)):
                label_str = ''
                label = train_tags[i]
                for k in label.keys():
                    if len(label[k]) > 0:
                        for entity in label[k]:
                            label_str = label_str + k + ':' + entity + ';'
                test_data.append([test_texts[i], label_str])
            test_datas[key] = test_data[:5000]
            train_datas[key] = train_data

    return train_datas, test_datas

def load_filtered_datasets(arg):
    id2label = arg.label_space[arg.task]

    test_datas, train_datas = {}, {}
    for dataset in arg.test_datasets[arg.task]:
        test_datas[dataset] = ''
        train_datas[dataset] = ''

    # 从filtered_samples中加载训练样本
    for key in test_datas.keys():
        train_data, test_data = [], []
        train_frame = pd.read_csv('datasets/{}/{}/test.tsv'.format(arg.task, key), sep='	').fillna(' ')
        texts = train_frame['Text'].values
        # 加载高置信度的样本
        with open('filtered_samples/{}.txt'.format(key), 'r') as wf:
            filtered_samples = [each.strip().split('\t') for each in wf.readlines()]
            ids = [int(each[0]) for each in filtered_samples]
            labels = [int(each[1]) for each in filtered_samples]
            texts = texts[ids]
            # labels = train_frame['Label'].values[ids]
        for text, label in zip(texts, labels):
            # print(text, label)
            train_data.append([text, id2label[label]])
        train_datas[key] = train_data

        # 加载测试集
        test_frame = pd.read_csv('datasets/{}/{}/test.tsv'.format(arg.task, key), sep='	')
        for text, label in zip(list(test_frame['Text'].values), list(test_frame['Label'].values)):
            test_data.append([text, id2label[label]])
        test_datas[key] = test_data[:5000]

    return train_datas, test_datas


def samples_selecter(arg):
    # 根据不同的文本评估器选取context中的样本，并存入ICL_samples
    if arg.select_from_filter:
        train_datas, test_datas = load_filtered_datasets(arg)
    else:
        train_datas, test_datas = load_datasets(arg)
    # 获取测试集合中每个元素的相似性得分
    for key in test_datas.keys():
        test_texts, train_texts = [], []
        print('generate {}..............'.format(key))
        for test_data in test_datas[key]:
            test_texts.append(test_data[0].replace('###', ' '))    # ### for NLI tasks
        for train_data in train_datas[key]:
            train_texts.append(train_data[0].replace('###', ' '))    # ### for NLI tasks
        #
        # draw_LDE(train_texts, test_texts, key)
        # continue
        if arg.score_func == 'random':
            sim_matrix = random_scorer(train_texts, test_texts)

        if arg.score_func == 'bm25':
            sim_matrix = BM25_scorer(train_texts, test_texts).T

        if arg.score_func == 'tfidf':
            sim_matrix = tfidf_scorer(train_texts, test_texts).T

        if arg.score_func == 'bert':
            bert_model = AutoModel.from_pretrained('/root/autodl-tmp/models/roberta/').cuda()
            bert_tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/roberta/')
            sim_matrix = bert_scorer(train_texts, test_texts, bert_model, bert_tokenizer).T

        all_sim_records, all_unsim_records = [], []

        for i in tqdm(range(sim_matrix.shape[0])):
            sorted_idx = np.argsort(sim_matrix[i,:])[::-1]
            if arg.task == 'SA':
                # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
                sims = {"negative": [], "positive": [], "neutral": []}
                unsims = {"negative": [], "positive": [], "neutral": []}
                for idx in sorted_idx:
                    if len(sims["negative"]) >= 10 and len(sims["positive"]) >= 10 and len(sims["neutral"]) >= 10:
                        break
                    if len(sims[train_datas[key][idx][1]]) < 10:
                        sims[train_datas[key][idx][1]].append(idx)

                re_sorted_idx = sorted_idx[::-1]
                for idx in re_sorted_idx:
                    if len(unsims["negative"]) >= 10 and len(unsims["positive"]) >= 10 and len(unsims["neutral"]) >= 10:
                        break
                    if len(unsims[train_datas[key][idx][1]]) < 10:
                        unsims[train_datas[key][idx][1]].append(idx)

            if arg.task == 'TD':
                # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
                sims = {"benign": [], "toxic": []}
                unsims = {"benign": [], "toxic": []}
                for idx in sorted_idx:
                    if len(sims["benign"]) >= 10 and len(sims["toxic"]) >= 10:
                        break
                    if len(sims[train_datas[key][idx][1]]) < 10:
                        sims[train_datas[key][idx][1]].append(idx)

                re_sorted_idx = sorted_idx[::-1]
                for idx in re_sorted_idx:
                    if len(unsims["benign"]) >= 10 and len(unsims["toxic"]) >= 10:
                        break
                    if len(unsims[train_datas[key][idx][1]]) < 10:
                        unsims[train_datas[key][idx][1]].append(idx)

            if arg.task == 'NLI':
                # 遍历sorted_idx从中寻找每个类别最相似（最不像）的样本
                sims = {'entailment':[], 'neutral':[], 'contradiction':[]}
                unsims = {'entailment':[], 'neutral':[], 'contradiction':[]}
                for idx in sorted_idx:
                    if len(sims["entailment"]) >= 10 and len(sims["neutral"]) >= 10 and len(sims["contradiction"]) >= 10:
                        break
                    if len(sims[train_datas[key][idx][1]]) < 10:
                        sims[train_datas[key][idx][1]].append(idx)

                re_sorted_idx = sorted_idx[::-1]
                for idx in re_sorted_idx:
                    if len(sims["entailment"]) >= 10 and len(sims["neutral"]) >= 10 and len(sims["contradiction"]) >= 10:
                        break
                    if len(unsims[train_datas[key][idx][1]]) < 10:
                        unsims[train_datas[key][idx][1]].append(idx)

            if arg.task == 'NER':
                sims = sorted_idx[:10].tolist()
                unsims = sorted_idx[-10:].tolist()

            all_sim_records.append(sims)
            all_unsim_records.append(unsims)
        # 写入文件
        if arg.select_from_filter:
            filtered = 'filter_'
        else:
            filtered = ''
        with open('ICL_samples/{}/{}/{}{}_sim.json'.format(arg.score_func, arg.task, filtered, key), 'w', encoding='utf-8') as f:
            json.dump(all_sim_records, f, default=convert_to_builtin_types, ensure_ascii=False,
                      indent=4)  # indent参数用于增加可读性
        with open('ICL_samples/{}/{}/{}{}_unsim.json'.format(arg.score_func, arg.task, filtered, key), 'w', encoding='utf-8') as f:
            json.dump(all_unsim_records, f, default=convert_to_builtin_types, ensure_ascii=False,
                      indent=4)  # indent参数用于增加可读性


if __name__ == '__main__':
    arg = Config()
    arg.task = 'NER'
    arg.score_func = 'bert'
    samples_selecter(arg)