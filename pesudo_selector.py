# 训练基于DeBERTa的选择器，用于从测试样本中选取置信度高的测试用例
import traceback

import torch
from click.core import batch
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaV2Tokenizer, DebertaV2Model, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup, \
    AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score
# import LLM_tester
import numpy as np
import cfg
from LLM_tester import *
Args = cfg.Config()

# 定义数据集
class TSVDataset(Dataset):
    def __init__(self, tokenizer, data, max_len, is_nli=False):
        self.tokenizer = tokenizer
        self.data = data
        if not is_nli:
            self.texts = self.data['Text'].values
        else:
            self.texts = ['Premise: ' + p + ' ' + 'Hypothesis: ' + h for p, h in zip(self.data['Premise'].values, self.data['Hypothesis'].values)]
        self.labels = self.data['Label'].values
        self.max_len = max_len
        self.ids = list(range(len(self.texts)))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = str(self.texts[idx])
            label = int(self.labels[idx])
            id = int(self.ids[idx])
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long),
                'id':torch.tensor(idx, dtype=torch.long)
            }
        except:
            # 打印错误信息（可选），然后返回None或特定的错误标记
            # print(f"Error processing data at index {idx}: {e}")
            traceback.print_exc()
            return None

# 使用自定义的collate_fn来处理DataLoader中的None值
def custom_collate_fn(batch):
    batch = [data for data in batch if data is not None]  # 过滤掉None值
    if len(batch) == 0:  # 如果整个批次都是None，则返回None
        return None
    return torch.utils.data.dataloader.default_collate(batch)


class DebertaClassifier(torch.nn.Module):
    def __init__(self, model_name, num_classes):
        super(DebertaClassifier, self).__init__()
        self.deberta = DebertaV2Model.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.deberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # 取[CLS]位输出
        return logits


def train_selector(dataset='amazon'):
    # 加载tokenizer和模型
    tokenizer = DebertaV2Tokenizer.from_pretrained('/root/autodl-tmp/models/deberta_v3/')
    model = DebertaClassifier('/root/autodl-tmp/models/deberta_v3/', num_classes=3)
    is_nli = False
    if dataset == 'amazon':
        train_file = 'datasets/SA/amazon/train.tsv'
    if dataset == 'civil_comments':
        train_file = 'datasets/TD/civil_comments/train.tsv'
    if dataset == 'mnli':
        train_file = 'datasets/NLI/mnli/train.tsv'
        is_nli = True

    # test_file = 'datasets/SA/dynasent/test.tsv'
    max_len = 128
    batch_size = 128
    learning_rate = 2e-5
    adam_epsilon = 1e-8
    num_epochs = 3
    gradient_accumulation_steps = 10

    datas = pd.read_csv(train_file, sep='	')
    train_data, val_data = train_test_split(datas, test_size=0.1)

    train_set = TSVDataset(tokenizer, train_data, max_len, is_nli)
    val_set = TSVDataset(tokenizer, val_data, max_len, is_nli)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    print(len(train_loader), len(val_loader))

    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)

    # 学习率调度器
    total_steps = len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # 训练模型
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    model.to('cuda')  # 如果你有GPU，则使用cuda
    val_accuracy, best_accuracy = 0, 0
    for epoch in range(num_epochs):
        total_loss, total_acc = 0, 0
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (step+1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                print('Epoch:{} steps: {}/{}'.format(epoch, step, total_steps))

            # if (step+1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to('cuda')
                    attention_mask = batch['attention_mask'].to('cuda')
                    labels = batch['label'].to('cuda')

                    logits = model(input_ids, attention_mask)
                    preds = torch.argmax(logits, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            val_accuracy = accuracy_score(all_labels, all_preds)

        # 保存最优模型参数
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(best_accuracy)
            torch.save(model.state_dict(), 'models/{}'.format(dataset))


def bert_based_sample_init(dataset='sst5'):
    if dataset in ['dynasent', 'semeval', 'sst5', 'amazon']:
        task = 'SA'
        save_model = 'models/{}'.format('amazon')
        num_labels = 3
        is_nli = False
    if dataset in ['adv_civil', 'implicit_hate', 'toxigen', 'civil_comments']:
        task = 'TD'
        save_model = 'models/{}'.format('civil_comments')
        num_labels = 2
        is_nli = False
    if dataset in ['anli', 'contract_nli', 'wanli', 'mnli']:
        task = 'NLI'
        num_labels = 3
        save_model = 'models/{}'.format('mnli')
        is_nli = True

    print('do dataset {}'.format(dataset))
    tokenizer = DebertaV2Tokenizer.from_pretrained('/root/autodl-tmp/models/deberta_v3/')
    model = DebertaClassifier('/root/autodl-tmp/models/deberta_v3/', num_classes=num_labels).cuda()
    model.load_state_dict(torch.load(save_model, map_location='cuda:0'))

    test_file = 'datasets/{}/{}/test.tsv'.format(task, dataset)
    datas = pd.read_csv(test_file, sep='	', nrows=5000)
    test_set = TSVDataset(tokenizer, datas, 128, is_nli=is_nli)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, collate_fn=custom_collate_fn)

    # 置信度区间统计
    confidence_bins = np.linspace(0, 1, 11)  # 10 个区间（0.0-0.1, ..., 0.9-1.0）
    bin_correct = np.zeros(10)  # 每个区间内预测正确的样本数
    bin_total = np.zeros(10)  # 每个区间内的总样本数

    with torch.no_grad(), open('filtered_samples/{}.txt'.format(dataset), 'w', encoding='utf-8') as f:
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')
            text_ids = batch['id']
            logits = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            max_probs, max_indices = torch.max(probs, dim=1)  # 获取置信度和预测类别
            # 写入 TXT 文件
            for tid, pred, conf, true_label in zip(text_ids, max_indices.cpu().numpy(), max_probs.cpu().numpy(), labels.cpu().numpy()):
                f.write(f"{tid}\t{pred}\t{conf:.4f}\n")
                bin_idx = min(int(conf * 10), 9)  # 确保最大值不会超出索引范围
                bin_total[bin_idx] += 1
                if pred == true_label:
                    bin_correct[bin_idx] += 1
    # 计算并打印每个区间的准确率
    print("\n📊 **置信度区间准确率统计**")
    for i in range(10):
        lower, upper = confidence_bins[i], confidence_bins[i + 1]
        acc = bin_correct[i] / bin_total[i] if bin_total[i] > 0 else 0.0
        print(f"⚡ 区间 {lower:.1f} - {upper:.1f} : 准确率 {acc:.4f} ({int(bin_correct[i])}/{int(bin_total[i])})")

    print(f"\n✅ 预测结果已保存")


def LLM_based_sample_selector(args, batches, model, tokenizer):
    print('load LLM {}......'.format(args.LLM))
    preds, labels, selected_idx, pred_labels, return_confidences = [], [], [], [], []
    for step in tqdm(list(range(len(batches)))):
        batch = batches[step]
        try:
            inputs = tokenizer(batch[0], return_tensors="pt", padding=True)
            labels.extend(batch[1])
            input_ids = inputs.input_ids.cuda()
            with torch.no_grad():
                if args.task == 'NER':
                    results = model.generate(input_ids, attention_mask=inputs.attention_mask.cuda(), pad_token_id=tokenizer.pad_token_id, max_new_tokens=50, output_scores=True, return_dict_in_generate=True)
                    generated_ids = results.sequences  # shape: [batch_size, sequence_length]
                    # 解码为字符串（批量）
                    input_len = input_ids.shape[1]
                    generated_ids = results.sequences  # [batch_size, input_len + new_tokens]
                    new_token_ids = generated_ids[:, input_len:]  # 只保留新生成的 token
                    outputs = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
                    outputs = [each.split('\n')[0].strip() for each in outputs]
                    for i in range(len(outputs)):
                        selected_idx.append(step * args.batch + i)
                    preds.extend(outputs)
                    # 计算置信度
                    log_probs = []
                    for i, logits in enumerate(results.scores):
                        log_softmax = torch.log_softmax(logits, dim=-1)  # shape: [batch_size, vocab_size]
                        token_id = new_token_ids[:, i]  # 当前步每个样本生成的 token
                        token_log_prob = log_softmax[torch.arange(log_softmax.size(0)), token_id]
                        log_probs.append(token_log_prob)
                    log_probs = torch.stack(log_probs, dim=1)  # shape: [batch_size, num_generated_tokens]
                    # 每个样本的平均 log prob，越大越好
                    mean_log_prob = log_probs.mean(dim=1)
                    confidence_score = torch.exp(mean_log_prob).detach().cpu().numpy()  # ∈ (0, 1)，越大越好
                    return_confidences.append(confidence_score)
                else:
                    results = model.generate(input_ids, attention_mask=inputs.attention_mask.cuda(), pad_token_id=tokenizer.pad_token_id, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
                    generated_tokens = results.sequences[:, -1]
                    scores = results.scores[0]  # 只生成一个新 token，取第一个 logits
                    # 计算生成 token 的概率分布
                    probs = torch.softmax(scores, dim=-1)  # [16, 50257]
                    # 获取生成 token 的索引
                    generated_token_indices = generated_tokens
                    # 从概率分布中提取生成 token 的置信度
                    confidence_scores = probs[torch.arange(probs.size(0)), generated_token_indices]
                    return_confidences.append(confidence_scores.cpu().numpy())
                    # 打印生成 token 及其置信度
                    if args.task != 'NER':
                        for i, (token, confidence) in enumerate(zip(generated_tokens, confidence_scores)):
                            token_str = tokenizer.decode(token).strip().lower()
                            if token_str == 'entail':
                                token_str = 'entailment'
                            selected_idx.append(step * args.batch + i)  # 选择出的高质量样本/返回所有样本
                            pred_labels.append(token_str.strip())
                            if args.task == 'TD':
                                if token_str in args.label_space[args.task][0]:
                                    preds.append(0)
                                elif token_str in args.label_space[args.task][1]:
                                    preds.append(1)
                                else:
                                    preds.append(100)
                            else:
                                if token_str in args.label_space[args.task][0]:
                                    preds.append(0)
                                elif token_str in args.label_space[args.task][1]:
                                    preds.append(1)
                                elif token_str in args.label_space[args.task][2]:
                                    preds.append(2)
                                else:
                                    preds.append(100)

        except:
            traceback.print_exc()
            continue
    if args.task != 'NER':
        swapped_label_space = {value: key for key, value in args.label_space[args.task].items()}
        labels = [swapped_label_space[each] for each in labels]
        acc = accuracy_score(preds, labels)
    else:
        acc = ICL_ner_acc(preds, labels)
    print('===================acc for current iter:{}============='.format(acc))
    assert len(selected_idx) == len(preds)
    return selected_idx, preds, labels, np.concatenate(return_confidences)

def CPP_based_sample_selector(dataset, task, current_iter, model, tokenizer, LLM):
    with open('iter_results/{}_{}_iter{}.txt'.format(LLM, dataset, current_iter), "r") as rf:
        labeled_datas = [each.strip().split('\t') for each in rf.readlines()]
    id2labels, id2label_strs, id2scores= {}, {}, {}
    for data in labeled_datas:
        if len(data) < 3:  # 确保数据格式正确
            continue
        try:
            sample_id, label, label_str, score = int(data[0]), int(data[1]), Args.label_space[task][int(data[1])], data[2]
        except:
            continue    # 一些非正常标签跳过
        id2labels[sample_id] = label
        id2label_strs[sample_id] = label_str
        id2scores[sample_id] = float(score)

    if dataset in ['dynasent', 'semeval', 'sst5', 'amazon']:
        task = 'SA'
        save_model = 'models/{}'.format('amazon')
        num_labels = 3
        is_nli = False
    if dataset in ['adv_civil', 'implicit_hate', 'toxigen', 'civil_comments']:
        task = 'TD'
        save_model = 'models/{}'.format('civil_comments')
        num_labels = 2
        is_nli = False
    if dataset in ['anli', 'contract_nli', 'wanli', 'mnli']:
        task = 'NLI'
        num_labels = 3
        save_model = 'models/{}'.format('mnli')
        is_nli = True

    test_file = 'datasets/{}/{}/test.tsv'.format(task, dataset)
    datas = pd.read_csv(test_file, sep='	', nrows=5000)
    # 从datas中构建batch
    batches = []
    if task == 'SA':
        for idx, text in enumerate(list(datas['Text'].values)):
            if idx in id2label_strs.keys():
                text1 = 'Question: What is the label of text "{}", negative, positive or neutral? \n Ansawer: '.format(text)
                text2 = 'Question: What is the label of {} text "{}", negative, positive or neutral? \n Ansawer: '.format(
                    id2label_strs[idx], text)
                # batch中封装的数据分别为：扰动的文本text1-text2，标签字符串，标签（整数），真实的标签（整数），样本对应的置信度得分，idx
                batches.append(
                    [text1, text2, id2labels[idx], id2label_strs[idx], int(datas["Label"].iloc[idx]), id2scores[idx],
                     idx])
    elif task == 'TD':
        for idx, text in enumerate(list(datas['Text'].values)):
            if idx in id2label_strs.keys():
                text1 = 'Question: What is the label of text "{}", toxic or not? \n Ansawer: '.format(text)
                text2 = 'Question:  What is the label of {} text "{}", toxic or not? \n Ansawer: '.format(id2label_strs[idx],
                                                                                                   text)
                batches.append(
                    [text1, text2, id2labels[idx], id2label_strs[idx], int(datas["Label"].iloc[idx]), id2scores[idx],
                     idx])
    elif task == 'NLI':
        texts = ['Premise: ' + p + ' ' + 'Hypothesis: ' + h for p, h in
                 zip(datas['Premise'].values, datas['Hypothesis'].values)]
        for idx, text in enumerate(texts):
            if idx in id2label_strs.keys():
                text = ' '.join(text.split()[:128])
                text1 = 'Question: What is the label of text "{}", entailment, neutral or contradiction? \n Ansawer: '.format(text)
                text2 = 'Question: What is the label of {} text "{}", entailment, neutral or contradiction? \n Ansawer: '.format(id2label_strs[idx], text)
                batches.append(
                    [text1, text2, id2labels[idx], id2label_strs[idx], int(datas["Label"].iloc[idx]), id2scores[idx], idx])
    # 按照batch计算困惑度
    all_cpps, all_conf, all_pp, all_pseudo_labels, all_labels, all_idx = [], [], [], [], [], []
    batch_size = 8
    for i in tqdm(range(0, len(batches), batch_size)):
        batch = batches[i:i+batch_size]
        batch_texts1 = [each[0] for each in batch]
        batch_texts2 = [each[1] for each in batch]
        batch_pseudo_labels = [each[2] for each in batch]
        batch_label_strs = [each[3] for each in batch]
        batch_labels = [each[4] for each in batch]
        batch_scores = [each[5] for each in batch]
        batch_idx = [each[6] for each in batch]
        encodings1 = tokenizer(batch_texts1, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids1 = encodings1.input_ids.to(model.device)
        attention_mask1 = encodings1.attention_mask.to(model.device)  # 处理 padding

        encodings2 = tokenizer(batch_texts2, padding=True, truncation=True, return_tensors="pt", max_length=512)
        input_ids2 = encodings2.input_ids.to(model.device)
        attention_mask2 = encodings2.attention_mask.to(model.device)  # 处理 padding

        with torch.no_grad():
            outputs = model(input_ids1, attention_mask=attention_mask1, labels=input_ids1)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            shift_logits = logits[:, :-1, :]  # 预测的 logits
            shift_labels = input_ids1[:, 1:]  # 对应的标签
            shift_attention_mask = attention_mask1[:, 1:]  # 计算 mask 以忽略 padding
            # 计算 token 级别的 log_softmax 概率
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            # 选取正确 token 位置的 log_probs (batch_size, seq_len)
            token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            # 仅计算非 padding token 的平均损失
            loss_per_token = -token_log_probs * shift_attention_mask  # 忽略 padding 部分
            loss_per_sample = loss_per_token.sum(dim=-1) / shift_attention_mask.sum(dim=-1)  # 样本级平均损失
            # 计算 PPL
            ppl_per_sample1 = torch.exp(loss_per_sample)

        with torch.no_grad():
            outputs = model(input_ids2, attention_mask=attention_mask2, labels=input_ids2)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            shift_logits = logits[:, :-1, :]  # 预测的 logits
            shift_labels = input_ids2[:, 1:]  # 对应的标签
            shift_attention_mask = attention_mask2[:, 1:]  # 计算 mask 以忽略 padding
            # 计算 token 级别的 log_softmax 概率
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            # 选取正确 token 位置的 log_probs (batch_size, seq_len)
            token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            # 仅计算非 padding token 的平均损失
            loss_per_token = -token_log_probs * shift_attention_mask  # 忽略 padding 部分
            loss_per_sample = loss_per_token.sum(dim=-1) / shift_attention_mask.sum(dim=-1)  # 样本级平均损失
            # 计算 PPL
            ppl_per_sample2 = torch.exp(loss_per_sample)

        ppls = torch.abs(ppl_per_sample1-ppl_per_sample2)
        all_conf.extend(batch_scores)
        all_pp.append(ppls)
        all_idx.extend(batch_idx)
        all_pseudo_labels.extend(batch_pseudo_labels)
        all_labels.extend(batch_labels)
    all_pp = torch.cat(all_pp)
    all_pp_norm = (all_pp / all_pp.max()) ** 0.5
    all_cpps = torch.tensor(all_conf).cuda() / (1 + all_pp_norm)  # 计算 CPP
    all_cpps = all_cpps.detach().cpu().numpy()   # 这玩意越大越好
    # ========== 归一化到 [0.5, 1] ==========
    conf_min = all_cpps.min()
    conf_max = all_cpps.max()
    if task in ['TD']:
        conf_norm = 0.5 + 0.5 * (all_cpps - conf_min) / (conf_max - conf_min)  # 归一化到 [0.5, 1]
    else:
        conf_norm = (1 / 3) + (2 / 3) * (all_cpps - conf_min) / (conf_max - conf_min)  # 线性缩放到 [1/3, 1]
    with open('iter_results/{}_{}_CPP_iter{}.txt'.format(LLM, dataset, current_iter), 'w') as f:
        for tid, pred, true_label, conf in zip(all_idx, all_pseudo_labels, all_labels, list(conf_norm)):
            f.write(f"{tid}\t{pred}\t{true_label}\t{conf:.4f}\n")

def PPL_based_sample_select(model_name, dataset, task, model, tokenizer):
    with open('filtered_samples/{}.txt'.format(dataset), 'r') as rf:
        labeled_datas = [each.strip().split('\t') for each in rf.readlines()]
    id2labels, id2label_strs, id2scores= {}, {}, {}
    for data in labeled_datas:
        if len(data) < 3:  # 确保数据格式正确
            continue
        sample_id, label, label_str, score = int(data[0]), int(data[1]), Args.label_space[task][int(data[1])], data[2]
        id2labels[sample_id] = label
        id2label_strs[sample_id] = label_str
        id2scores[sample_id] = float(score)

    if dataset in ['dynasent', 'semeval', 'sst5', 'amazon']:
        task = 'SA'
        save_model = 'models/{}'.format('amazon')
        num_labels = 3
        is_nli = False
    if dataset in ['adv_civil', 'implicit_hate', 'toxigen', 'civil_comments']:
        task = 'TD'
        save_model = 'models/{}'.format('civil_comments')
        num_labels = 2
        is_nli = False
    if dataset in ['anli', 'contract_nli', 'wanli', 'mnli']:
        task = 'NLI'
        num_labels = 3
        save_model = 'models/{}'.format('mnli')
        is_nli = True

    test_file = 'datasets/{}/{}/test.tsv'.format(task, dataset)
    datas = pd.read_csv(test_file, sep='	', nrows=5000)
    # 从datas中构建batch
    batches = []
    if task == 'SA':
        for idx, text in enumerate(list(datas['Text'].values)):
            if idx in id2label_strs.keys():
                text1 = 'Question: What is the label of text "{}", negative, positive or neutral? \n Ansawer: '.format(text)
                text2 = 'Question: What is the label of {} text "{}", negative, positive or neutral? \n Ansawer: '.format(id2label_strs[idx], text)
                # batch中封装的数据分别为：扰动的文本text1-text2，标签字符串，标签（整数），真实的标签（整数），样本对应的置信度得分，idx
                batches.append([text1, text2, id2labels[idx], id2label_strs[idx], int(datas["Label"].iloc[idx]), id2scores[idx], idx])
    elif task == 'TD':
        for idx, text in enumerate(list(datas['Text'].values)):
            if idx in id2label_strs.keys():
                text1 = 'Question: What is the label of text "{}", toxic or not? \n Ansawer: '.format(text)
                text2 = 'Question: What is the label of {} text "{}", toxic or not? \n Ansawer: '.format(id2label_strs[idx], text)
                batches.append([text1, text2, id2labels[idx], id2label_strs[idx], int(datas["Label"].iloc[idx]), id2scores[idx], idx])
    elif task == 'NLI':
        texts = ['Premise: ' + p + ' ' + 'Hypothesis: ' + h for p, h in zip(datas['Premise'].values, datas['Hypothesis'].values)]
        for idx, text in enumerate(texts):
            text = ' '.join(text.split()[:1024])
            if idx in id2label_strs.keys():
                text1 = 'Question: What is the label of text "{}", entailment, neutral or contradiction? \n Ansawer: '.format(text)
                text2 = 'Question: What is the label of {} text "{}", entailment, neutral or contradiction? \n Ansawer: '.format(id2label_strs[idx], text)
                batches.append([text1, text2, id2labels[idx], id2label_strs[idx], int(datas["Label"].iloc[idx]), id2scores[idx], idx])
    # 按照batch计算困惑度
    all_cpps, all_conf, all_pp, all_pseudo_labels, all_labels, all_idx = [], [], [], [], [], []
    batch_size = 2
    for i in tqdm(range(0, len(batches), batch_size)):
        batch = batches[i:i+batch_size]
        batch_texts1 = [each[0] for each in batch]
        batch_texts2 = [each[1] for each in batch]
        batch_pseudo_labels = [each[2] for each in batch]
        batch_label_strs = [each[3] for each in batch]
        batch_labels = [each[4] for each in batch]
        batch_scores = [each[5] for each in batch]
        batch_idx = [each[6] for each in batch]
        encodings1 = tokenizer(batch_texts1, padding=True, truncation=True, return_tensors="pt")
        input_ids1 = encodings1.input_ids.to(model.device)
        attention_mask1 = encodings1.attention_mask.to(model.device)  # 处理 padding

        encodings2 = tokenizer(batch_texts2, padding=True, truncation=True, return_tensors="pt")
        input_ids2 = encodings2.input_ids.to(model.device)
        attention_mask2 = encodings2.attention_mask.to(model.device)  # 处理 padding

        with torch.no_grad():
            outputs = model(input_ids1, attention_mask=attention_mask1, labels=input_ids1)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            shift_logits = logits[:, :-1, :]  # 预测的 logits
            shift_labels = input_ids1[:, 1:]  # 对应的标签
            shift_attention_mask = attention_mask1[:, 1:]  # 计算 mask 以忽略 padding
            # 计算 token 级别的 log_softmax 概率
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            # 选取正确 token 位置的 log_probs (batch_size, seq_len)
            token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            # 仅计算非 padding token 的平均损失
            loss_per_token = -token_log_probs * shift_attention_mask  # 忽略 padding 部分
            loss_per_sample = loss_per_token.sum(dim=-1) / shift_attention_mask.sum(dim=-1)  # 样本级平均损失
            # 计算 PPL
            ppl_per_sample1 = torch.exp(loss_per_sample)

        with torch.no_grad():
            outputs = model(input_ids2, attention_mask=attention_mask2, labels=input_ids2)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            shift_logits = logits[:, :-1, :]  # 预测的 logits
            shift_labels = input_ids2[:, 1:]  # 对应的标签
            shift_attention_mask = attention_mask2[:, 1:]  # 计算 mask 以忽略 padding
            # 计算 token 级别的 log_softmax 概率
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            # 选取正确 token 位置的 log_probs (batch_size, seq_len)
            token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            # 仅计算非 padding token 的平均损失
            loss_per_token = -token_log_probs * shift_attention_mask  # 忽略 padding 部分
            loss_per_sample = loss_per_token.sum(dim=-1) / shift_attention_mask.sum(dim=-1)  # 样本级平均损失
            # 计算 PPL
            ppl_per_sample2 = torch.exp(loss_per_sample)

        ppls = torch.abs(ppl_per_sample1-ppl_per_sample2)
        all_conf.extend(batch_scores)
        all_pp.append(ppls)
        all_idx.extend(batch_idx)
        all_pseudo_labels.extend(batch_pseudo_labels)
        all_labels.extend(batch_labels)
    all_pp = torch.cat(all_pp)
    all_pp_norm = (all_pp / all_pp.max()) ** 0.5
    all_cpps = torch.tensor(all_conf).cuda() / (1 + all_pp_norm)  # 计算 CPP
    all_cpps = list(all_cpps.detach().cpu().numpy())   # 这玩意越大越好
    with open('filtered_samples/{}_{}_CPP.txt'.format(model_name, dataset), 'w') as f:
        for tid, pred, true_label, conf in zip(all_idx, all_pseudo_labels, all_labels, all_cpps):
            f.write(f"{tid}\t{pred}\t{true_label}\t{conf:.4f}\n")
    print(f"样本CPP已保存到文件")


def cal_CPP_acc(model_name, dataset):
    # 计算 CPP 的准确率
    # 读取数据
    if dataset in ['dynasent', 'semeval', 'sst5', 'amazon']:
        task = 'SA'
        save_model = 'models/{}'.format('amazon')
        num_labels = 3
        is_nli = False
    if dataset in ['adv_civil', 'implicit_hate', 'toxigen', 'civil_comments']:
        task = 'TD'
        save_model = 'models/{}'.format('civil_comments')
        num_labels = 2
        is_nli = False
    if dataset in ['anli', 'contract_nli', 'wanli', 'mnli']:
        task = 'NLI'
        num_labels = 3
        save_model = 'models/{}'.format('mnli')
        is_nli = True

    test_file = 'datasets/{}/{}/test.tsv'.format(task, dataset)
    datas = pd.read_csv(test_file, sep='	', nrows=5000)

    file_path = 'filtered_samples/{}_{}_CPP.txt'.format(model_name, dataset)
    tid_list, pred_list, true_list, conf_list = [], [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            tid, pred, _, conf = line.strip().split("\t")
            tid_list.append(tid)
            pred_list.append(int(pred))
            true_label = datas.iloc[int(tid)]['Label']
            true_list.append(int(true_label))
            conf_list.append(float(conf))
    # 转换为 NumPy 数组
    pred_array = np.array(pred_list)
    true_array = np.array(true_list)
    conf_array = np.array(conf_list)
    # ========== 归一化到 [0.5, 1] ==========
    conf_min = conf_array.min()
    conf_max = conf_array.max()
    if task == 'TD':
        conf_norm = 0.5 + 0.5 * (conf_array - conf_min) / (conf_max - conf_min)  # 归一化到 [0.5, 1]
    else:
        conf_norm = (1 / 3) + (2 / 3) * (conf_array - conf_min) / (conf_max - conf_min)  # 线性缩放到 [1/3, 1]

    # 计算 0-1 区间的准确率
    bins = np.linspace(0, 1, 11)  # 10个区间边界
    bin_indices = np.digitize(conf_norm, bins) - 1  # 使索引从 0 开始

    acc_per_bin = {}
    for i in range(10):
        mask = bin_indices == i
        sample_count = np.sum(mask)  # 该区间的样本数量
        if sample_count > 0:
            acc = np.mean(pred_array[mask] == true_array[mask])
            acc_per_bin[f"{bins[i]:.1f}-{bins[i + 1]:.1f}"] = (acc, sample_count)
        else:
            acc_per_bin[f"{bins[i]:.1f}-{bins[i + 1]:.1f}"] = (None, 0)  # 该区间无样本

    # ========== 输出结果 ==========
    print("\n按归一化置信度划分 (0-1 区间)：")
    print(f"{'区间':<12}{'准确率':<10}{'样本数量':<10}")
    for bin_range, (acc, count) in acc_per_bin.items():
        acc_str = f"{acc:.2f}" if acc is not None else "无样本"
        print(f"{acc_str:<10}")

    with open('filtered_samples/{}_{}_CPP_norm.txt'.format(model_name, dataset), 'w') as f:
        for tid, pred, label, conf in zip(tid_list, pred_list, true_list, conf_norm.tolist()):
            f.write(f"{tid}\t{pred}\t{label}\t{conf:.4f}\n")
    print(f"样本CPP已保存到文件")


def cal_DeeBERTa_acc(dataset):
    # 读取数据
    if dataset in ['dynasent', 'semeval', 'sst5', 'amazon']:
        task = 'SA'
        save_model = 'models/{}'.format('amazon')
        num_labels = 3
        is_nli = False
    if dataset in ['adv_civil', 'implicit_hate', 'toxigen', 'civil_comments']:
        task = 'TD'
        save_model = 'models/{}'.format('civil_comments')
        num_labels = 2
        is_nli = False
    if dataset in ['anli', 'contract_nli', 'wanli', 'mnli']:
        task = 'NLI'
        num_labels = 3
        save_model = 'models/{}'.format('mnli')
        is_nli = True

    test_file = 'datasets/{}/{}/test.tsv'.format(task, dataset)
    datas = pd.read_csv(test_file, sep='	', nrows=5000)

    file_path = 'filtered_samples/{}_{}_CPP.txt'.format(model_name, dataset)
    tid_list, pred_list, true_list, conf_list = [], [], [], []

    with open(file_path, 'r') as f:
        for line in f:
            tid, pred, _, conf = line.strip().split("\t")
            tid_list.append(tid)
            pred_list.append(int(pred))
            true_label = datas.iloc[int(tid)]['Label']
            true_list.append(int(true_label))
            conf_list.append(float(conf))

    print(accuracy_score(pred_list, true_list))



if __name__ == '__main__':
    cfg = Config()
    for model_name in ['Qwen2.5-3b']:
        cfg.batch = 20
        cfg.LLM = model_name
        cfg.confidence = cfg.model2confidence
        cfg.LLM_path = '/root/autodl-tmp/models/{}'.format(model_name)
        # model, tokenizer = load_base_model_tokenizer(cfg)
        for task in ['TD', 'SA', 'NLI']:
            cfg.task = task
            if cfg.task == 'TD':
                for dataset in ['adv_civil', 'implicit_hate', 'toxigen']:
                    print('==========={}-{}========'.format(model_name, dataset))
                    # PPL_based_sample_select(model_name, dataset, task, model, tokenizer)
                    # cal_CPP_acc(model_name, dataset)
                    cal_DeeBERTa_acc(dataset)
            if cfg.task == 'SA':
                for dataset in ['dynasent', 'semeval', 'sst5']:
                    print('==========={}-{}========'.format(model_name, dataset))
                    # PPL_based_sample_select(model_name, dataset, task, model, tokenizer)
                    # cal_CPP_acc(model_name, dataset)
                    cal_DeeBERTa_acc(dataset)
            if cfg.task == 'NLI':
                for dataset in ['anli', 'contract_nli', 'wanli']:
                    print('==========={}-{}========'.format(model_name, dataset))
                    try:
                        # PPL_based_sample_select(model_name, dataset, task, model, tokenizer)
                        # cal_CPP_acc(model_name, dataset)
                        cal_DeeBERTa_acc(dataset)
                    except:
                        continue