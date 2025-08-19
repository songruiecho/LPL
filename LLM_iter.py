# 用于LLM参与迭代的代码，我们设置迭代次数为3，第一次迭代为初始迭代，利用训练集样本进行随机选取，然后选择高置信度样本进行处理
# 随后两次进行LLM参与的迭代，并选择新的样本加入到伪标签集合，用于下一次迭代；
# 最后三次迭代的结果进行多数投票以提升预测的稳定性。
import traceback

import numpy as np

import pesudo_selector
from cfg import Config
import LLM_tester
from pesudo_selector import LLM_based_sample_selector, CPP_based_sample_selector
from sklearn.metrics import accuracy_score, f1_score


# def iterator(cfg, dataset, model, tokenizer):
#     print('load LLM {}......'.format(cfg.LLM))
#     id2label = cfg.label_space[cfg.task]
#     total_selected_idx, total_selected_labels = [], []
#     preds_for_iters = []      # 每次迭代的预测样本
#     # 初始化已经由DeBERTa选取的样本
#     with open('filtered_samples/{}.txt'.format(dataset), 'r') as wf:
#         filtered_samples = [each.strip().split('\t') for each in wf.readlines()]
#         ids = [int(each[0]) for each in filtered_samples]
#         labels = [id2label[int(each[1])] for each in filtered_samples]
#     if cfg.task != 'NLI':
#         total_selected_idx.extend(ids)
#         total_selected_labels.extend(labels)
#     for current_iter in range(cfg.iter_times):
#         print('=============start {}-th iter with {} samples==================='.format(current_iter, len(total_selected_idx)))
#         if current_iter == 0:
#             batches = LLM_tester.get_batch_from_selected_samples(cfg, ids, labels, dataset)
#         else:
#             batches = LLM_tester.get_batch_from_selected_samples(cfg, total_selected_idx, total_selected_labels, dataset)
#         current_selected_idx, pred_labels, (preds, labels) = LLM_based_sample_selector(cfg, batches, model, tokenizer)
#         for idx, label in zip(current_selected_idx, pred_labels):
#             if idx not in total_selected_idx:
#                 total_selected_idx.append(idx)
#                 total_selected_labels.append(label)
#         preds_for_iters.append(np.array(preds))
#     # 将所有预测结果堆叠成一个二维数组
#     final_preds = np.vstack(preds_for_iters)
#     # 进行投票，找出每一列中出现次数最多的值
#     final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=final_preds)
#     acc = accuracy_score(final_preds, labels)
#     print('===========最终预测结果：{}========================'.format(acc))

def iterator(cfg, dataset, model, tokenizer):
    ''' 每次选择固定的置信度前cfg.model2confidence百分比对应的样本进行更新，防止不同LLMs置信度区间不一致造成的复杂度
    :param cfg:
    :param dataset:
    :param model:
    :param tokenizer:
    :return:
    '''
    print('load LLM {}......'.format(cfg.LLM))
    id2label = cfg.label_space[cfg.task]
    total_selected_idx, total_selected_labels = [], []
    preds_for_iters = []      # 每次迭代的预测样本
    # 初始化已经由DeBERTa选取的样本
    with open('filtered_samples/{}_CPP_norm.txt'.format(dataset), 'r') as rf:
        filtered_samples = [each.strip().split('\t') for each in rf.readlines()]
        ids = [int(each[0]) for each in filtered_samples]
        labels = [id2label[int(each[1])] for each in filtered_samples]
        CPP_scores = [each[3] for each in filtered_samples]
        filtered_ids_labels = [(ids[i], labels[i]) for i in range(len(CPP_scores)) if float(CPP_scores[i]) > cfg.model2confidence]
        ids, labels = zip(*filtered_ids_labels) if filtered_ids_labels else ([], [])
    total_selected_idx.extend(ids)
    total_selected_labels.extend(labels)
    for current_iter in range(cfg.iter_times):
        print('=============start {}-th iter with {} samples==================='.format(current_iter, len(total_selected_idx)))
        if current_iter == 0:
            batches = LLM_tester.get_batch_from_selected_samples(cfg, ids, labels, dataset)
        else:
            batches = LLM_tester.get_batch_from_selected_samples(cfg, total_selected_idx, total_selected_labels, dataset)
        current_selected_idx, pred_labels, (preds, labels), confidences = LLM_based_sample_selector(cfg, batches, model, tokenizer)
        # 将样本的idx, preds, conf存储进“iter_resutls”文件夹, 命名规则为iter{current_iter}_{dataset}.txt
        filtered_samples = [(idx, pred, conf) for idx, pred, label, conf in zip(current_selected_idx, preds, labels, confidences)]
        # 写入文件
        with open('iter_results/{}_{}_iter{}.txt'.format(cfg.LLM, dataset, current_iter), "w") as f:
            for idx, pred, conf in filtered_samples:
                f.write(f"{idx}\t{pred}\t{conf}\n")
        if current_iter < cfg.iter_times-1:
            CPP_based_sample_selector(dataset, cfg.task, current_iter, model, tokenizer, cfg.LLM)  # 计算CPP并写入iter_CPP文件
            # 读取iter_CPP并根据阈值选取top_idx
            top_percent_idx, top_pred_labels = [], []
            with open('iter_results/{}_{}_CPP_iter{}.txt'.format(cfg.LLM, dataset, current_iter), "r") as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:  # 确保有置信度列
                        idx, _, label, conf = parts[:4]  # 忽略 pred
                        if float(conf) > cfg.gamma:
                            top_percent_idx.append(int(idx))
                            top_pred_labels.append(id2label[int(label)])   # 注意这里是label_str而不是int
            for idx, label in zip(top_percent_idx, top_pred_labels):
                if idx not in total_selected_idx:
                    total_selected_idx.append(idx)
                    total_selected_labels.append(label)
        preds_for_iters.append(np.array(preds))
    # 将所有预测结果堆叠成一个二维数组
    final_preds = np.vstack(preds_for_iters)
    # 进行投票，找出每一列中出现次数最多的值
    final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=final_preds)
    acc = accuracy_score(final_preds, labels)
    print('===========最终预测结果：{}========================'.format(acc))

def ner_iterator(cfg, dataset, model, tokenizer):
    ''' 每次选择固定的置信度前cfg.model2confidence百分比对应的样本进行更新，防止不同LLMs置信度区间不一致造成的复杂度
    :param cfg:
    :param dataset:
    :param model:
    :param tokenizer:
    :return:
    '''
    print('load LLM {}......'.format(cfg.LLM))
    id2label = cfg.label_space[cfg.task]
    total_selected_idx, total_selected_labels = [], []
    preds_for_iters = []      # 每次迭代的预测样本
    # 初始化已经由DeBERTa选取的样本
    with open('filtered_samples/{}_{}_CPP_norm.txt'.format(cfg.LLM, dataset), 'r') as rf:
        filtered_samples = [each.strip().split('\t') for each in rf.readlines()]
        ids = [int(each[0]) for each in filtered_samples]
        labels = [each[1] for each in filtered_samples]
        CPP_scores = [each[3] for each in filtered_samples]
        filtered_ids_labels = [(ids[i], labels[i]) for i in range(len(CPP_scores)) if float(CPP_scores[i]) > cfg.model2confidence]
        ids, labels = zip(*filtered_ids_labels) if filtered_ids_labels else ([], [])
    total_selected_idx.extend(ids)
    total_selected_labels.extend(labels)
    for current_iter in range(cfg.iter_times):
        print('=============start {}-th iter with {} samples==================='.format(current_iter, len(total_selected_idx)))
        if current_iter == 0:
            batches = LLM_tester.get_batch_from_selected_samples(cfg, ids, labels, dataset)
        else:
            batches = LLM_tester.get_batch_from_selected_samples(cfg, total_selected_idx, total_selected_labels, dataset)
        current_selected_idx, preds, labels, confidences = LLM_based_sample_selector(cfg, batches, model, tokenizer)
        confidences = (confidences - confidences.min()) / (confidences.max() - confidences.min())
        confidences = confidences.tolist()
        # 将样本的idx, preds, conf存储进“iter_resutls”文件夹, 命名规则为iter{current_iter}_{dataset}.txt
        filtered_samples = [(idx, pred, conf) for idx, pred, label, conf in zip(current_selected_idx, preds, labels, confidences)]
        # 写入文件
        with open('iter_results/{}_{}_iter{}.txt'.format(cfg.LLM, dataset, current_iter), "w") as f:
            for idx, pred, conf in filtered_samples:
                f.write(f"{idx}\t{pred}\t{conf}\n")
        if current_iter < cfg.iter_times-1:
            # 读取iter并根据阈值选取top_idx
            top_percent_idx, top_pred_labels = [], []
            with open('iter_results/{}_{}_iter{}.txt'.format(cfg.LLM, dataset, current_iter), "r") as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:  # 确保有置信度列
                        idx, label, conf = parts[:3]  # 忽略 pred
                        if float(conf) > cfg.gamma:
                            top_percent_idx.append(int(idx))
                            top_pred_labels.append(label)   # 注意这里是label_str而不是int
            print(top_percent_idx, top_pred_labels)
            for idx, label in zip(top_percent_idx, top_pred_labels):
                if idx not in total_selected_idx:
                    total_selected_idx.append(idx)
                    total_selected_labels.append(label)
        preds_for_iters.append(np.array(preds))
    # 将所有预测结果堆叠成一个二维数组
    # final_preds = np.vstack(preds_for_iters)
    # 进行投票，找出每一列中出现次数最多的值
    # final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=final_preds)
    # acc = accuracy_score(final_preds, labels)
    # print('===========最终预测结果：{}========================'.format(acc))

if __name__ == '__main__':
    cfg = Config()
    # for model_name in ['llama3.2-3b', 'llama3.1-8b', 'Qwen2.5-3b', 'Mistral-7B-v0.3']:
    for model_name in ['llama3.2-3b']:
        cfg.batch = 15
        cfg.LLM = model_name
        cfg.confidence = cfg.model2confidence
        cfg.LLM_path = '/root/autodl-tmp/models/{}'.format(model_name)
        model, tokenizer = LLM_tester.load_base_model_tokenizer(cfg)
        for task in ['NER']:
            cfg.task = task
            if cfg.task == 'TD':
                for dataset in ['adv_civil', 'implicit_hate', 'toxigen']:
                    print('==========={}-{}========'.format(model_name, dataset))
                    try:
                        # LLM_tester.ICL(cfg, dataset, model, tokenizer)
                        iterator(cfg, dataset, model, tokenizer)
                    except:
                        continue
            if cfg.task == 'SA':
                # for dataset in ['amazon']:
                for dataset in ['dynasent', 'semeval', 'sst5']:
                    print('==========={}-{}========'.format(model_name, dataset))
                    try:
                        # LLM_tester.ICL(cfg, dataset, model, tokenizer)
                        iterator(cfg, dataset, model, tokenizer)
                    except:
                        continue
            if cfg.task == 'NLI':
                for dataset in ['anli', 'contract_nli', 'wanli']:
                    print('==========={}-{}========'.format(model_name, dataset))
                    try:
                        # LLM_tester.ICL(cfg, dataset, model, tokenizer)
                        iterator(cfg, dataset, model, tokenizer)
                    except:
                        traceback.print_exc()
                        continue
            if cfg.task == 'NER':
                for dataset in ['conll']:
                    print('==========={}-{}========'.format(model_name, dataset))
                    try:
                        # LLM_tester.ICL(cfg, dataset, model, tokenizer)
                        ner_iterator(cfg, dataset, model, tokenizer)
                    except:
                        traceback.print_exc()
                        continue
        del model