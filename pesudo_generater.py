# 基于T5的NER标记器，同pesudo_generater具有相同的功能，但是面向不同的任务
import torch
from click.core import batch
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaV2Tokenizer, DebertaV2Model, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score
# import LLM_tester
import numpy as np
import cfg
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, T5ForConditionalGeneration
from LLM_tester import *
Args = cfg.Config()

# 函数：从句子和标签中提取实体及其类型（NER专用）
def extract_entities(sentence, labels):
    words = sentence.split()  # 将句子按空格分割成单词
    entities = []  # 用于存储实体及其类型

    current_entity = []  # 当前实体
    current_label = None  # 当前实体的标签

    for word, label in zip(words, labels):
        if label.startswith('B-'):  # 新实体的开始
            if current_entity:
                entities.append((' '.join(current_entity), current_label))  # 添加之前的实体
            current_entity = [word]  # 开始一个新的实体
            current_label = label[2:]  # 记录实体类型，去掉'B-'前缀
        elif label.startswith('I-'):  # 实体的延续
            if current_entity:
                current_entity.append(word)  # 将词添加到当前实体
            else:
                current_entity = [word]  # 处理错误的I-标签，防止没有B-标签就有I-
                current_label = label[2:]
        else:
            if current_entity:
                entities.append((' '.join(current_entity), current_label))  # 添加之前的实体
                current_entity = []  # 清空当前实体
                current_label = None  # 清空当前标签

    if current_entity:  # 如果句子末尾有未添加的实体
        entities.append((' '.join(current_entity), current_label))

    return entities

def get_Dataset(datas, tags, tokenizer, test=False):
    inputs = []
    attention_masks = []
    str_labels, labels = [], []
    for data, tag in tqdm(zip(datas, tags)):
        prompt, str_label = create_prompt(data, tag)
        encoded_text = tokenizer(prompt, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        label_text = tokenizer(str_label, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        inputs.append(encoded_text['input_ids'])
        attention_masks.append(encoded_text['attention_mask'])
        labels.append(label_text['input_ids'])
        str_labels.append(str_label)
    inputs = torch.cat(inputs, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.cat(labels, dim=0)
    dataset = NERDataset(inputs, labels, str_labels, attention_masks)
    return dataset

def load_ner_by_space(path):
    ''' 根据空格加载NER所需的实验数据
    :param path:
    :return:
    '''
    stand_labels = ["O", "B-organization", "I-organization", "B-location", "I-location", "B-person", "I-person"]
    datas = []
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read().replace('\r\n', '\n').replace('\r', '\n')
    lines = content.split('\n')
    for line in lines:
        line = line.split('\t')
        if len(line) <= 2:
            datas.append(line)
    sentences = []
    ner_tags = []
    current_sentence = []
    current_tags = []
    is_append = True
    for each in tqdm(datas):
        if len(each) == 2:
            word, tag = each[0], each[1]
            current_sentence.append(word)
            current_tags.append(tag)
        else:
            for each in current_tags:
                if each not in stand_labels:  # 排除不想计算的tag
                    is_append = False
            if is_append:
                sentences.append(current_sentence)
                ner_tags.append(current_tags)
                assert len(current_sentence) == len(current_tags)
            current_sentence = []
            current_tags = []
            is_append = True
    # 处理最后一行（如果没有以空格结尾的分隔符）
    if current_sentence:
        sentences.append(current_sentence)
        ner_tags.append(current_tags)

    sentences = [' '.join(s) for s in sentences]
    ner_tags = [extract_entities(sentence, labels) for sentence, labels in zip(sentences, ner_tags)]
    sen_results, label_results = [], []
    for s, n in zip(sentences, ner_tags):
        if len(n) > 0:
            sen_results.append(s)
            e2l = {"organization": [], "person": [], "location": []}
            for each in n:
                e2l[each[1]].append(each[0])
            label_results.append(e2l)
    return sen_results, label_results

class NERDataset(Dataset):
    def __init__(self, inputs, labels, str_labels, attention_masks):
        self.inputs = inputs
        self.str_labels = str_labels
        self.labels = labels
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            'labels': self.labels[idx],
            'str_labels': self.str_labels[idx],
            'attention_masks': self.attention_masks[idx]
        }

def create_prompt(sentence, entities):
    prompt = "NER: " + sentence + ' Entity: '
    str_label = ""
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            str_label += f"{entity_type}: {entity}; "
    return prompt, str_label

def compute_entity_accuracy(labels, preds):
    correct_count, total_count = 0, 0
    for label, pred in zip(labels, preds):
        # 将列表转换为集合
        set_pred = set(pred)
        set_label = set(label)
        # 计算交集和并集
        intersection = set_pred.intersection(set_label)
        union = set_pred.union(set_label)
        correct_count += len(intersection)
        total_count += len(union)
    return correct_count / total_count

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
                label.append(each[0] + ': ' + each[1])
        label = '; '.join(label) + ';'
    else:
        label = ';'
    return label

def ner_acc(preds, labels):
    # organization:person: Grete Stroem; location: Norway
    '''
    :param preds:
    :param labels:
    :return:
    '''
    # 解析预测标签和真实标签
    parsed_preds = [pred.split('; ') for pred in preds]
    parsed_labels = [label.split('; ') for label in labels]
    correct_count, total_count = 0, 1e-8
    for pred, label in zip(parsed_preds,parsed_labels):
        # 将列表转换为集合
        label = [each for each in label if '' != each]
        pred = [each for each in pred if '' != each]
        set_pred = set(pred)
        set_label = set(label)
        # 计算交集和并集
        intersection = set_pred.intersection(set_label)
        # print(set_pred, set_label)
        union = set_pred.union(set_label)
        correct_count += len(intersection)
        total_count += len(union)
    return correct_count/total_count

def T5_based_sample_init(args, target='conll'):
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/t5-base/', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    test_datas, test_tags = load_ner_by_space('datasets/NER/{}/sample.tsv'.format(target))
    # 将数据构建成NER的专用模板
    test_dataset = get_Dataset(test_datas, test_tags, tokenizer, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    model = T5ForConditionalGeneration.from_pretrained('/root/code/RAOOD/retriever/t5/').cuda()
    model.eval()  # 设定模型为评估模式
    with torch.no_grad():  # 在评估阶段不需要计算梯度
        generate_texts, confidences = [], []
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_masks'].cuda()
            labels = batch['labels'].cuda()
            # 前向传播
            generate_ids = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_new_tokens=30,
                                          attention_mask=attention_mask, output_scores=True,
                                          return_dict_in_generate=True)
            generated_ids = generate_ids.sequences  # 取整个 batch 的生成结果
            for i in range(input_ids.shape[0]):
                gen_ids = generated_ids[i]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                scores = generate_ids.scores  # list of tensors [batch_size, vocab_size]
                token_probs = []
                for step, score in enumerate(scores):
                    prob = torch.softmax(score[i], dim=-1)  # 第i条数据第step步的softmax
                    token_prob = prob[gen_ids[step + 1]]  # 第step+1个生成token的概率
                    token_probs.append(token_prob)
                confidence = torch.stack(token_probs).prod().item()

                generate_texts.append(gen_text)
                confidences.append(confidence)

    all_input_texts = test_datas
    csv_labels = []
    csv_confidences = []
    csv_texts = []

    for i in range(len(generate_texts)):
        text = generate_texts[i]
        label = extract_entity_from_output(text)
        confidence = confidences[i]
        input_text = all_input_texts[i]
        if label != ';':
            csv_labels.append(label)
            csv_confidences.append(confidence)
            csv_texts.append(input_text)

    # 归一化confidence
    min_confidence = min(csv_confidences)  # 所有样本的 log_likelihood
    max_confidence = max(csv_confidences)
    # 最小-最大归一化到 [0, 1]
    normalized_confidences = [(each - min_confidence) / (
            max_confidence - min_confidence) if max_confidence != min_confidence else 0 for each in csv_confidences]
    df = pd.DataFrame({
        'Label': csv_labels,
        'Confidence': normalized_confidences,
        'Text': csv_texts
    })
    print('==================一共标记{}条高置信度样本=================='.format(len(csv_texts)))
    df = pd.DataFrame(df, columns=["Text", "Label", "Confidence"])
    df.to_csv("filtered_samples/{}.csv".format(target), index=False)

def PPL_based_ner_select(model_name, dataset, model, tokenizer):
    test_datas, test_tags = load_ner_by_space('datasets/NER/{}/sample.tsv'.format(dataset))
    real_label_strs = []
    for tag in test_tags:
        real_label_str = []
        for key in tag.keys():
            if len(tag[key]) > 0:
                for each in tag[key]:
                    real_label_str.append(key + ':' + each)
        real_label_str = '; '.join(real_label_str)
        real_label_strs.append(real_label_str)

    frame = pd.read_csv("filtered_samples/{}.csv".format(dataset))
    labeled_datas = frame['Text'].values
    label_strs = frame['Label'].values
    scores = frame['Confidence'].values
    batches = []
    for idx, (text, label, score) in enumerate(zip(labeled_datas, label_strs, scores)):
        text1 = "Question: What entities are there in the text {}?\n Ansawer: ".format(text)
        text2 = "Question: What entities are there in the text {} {}?\n Ansawer: ".format(text, label)
        batches.append([text1, text2, label_strs[idx], real_label_strs[idx], scores[idx], idx])
    # 按照batch计算困惑度
    all_cpps, all_conf, all_pp, all_pseudo_labels, all_labels, all_idx = [], [], [], [], [], []
    batch_size = 2
    for i in tqdm(range(0, len(batches), batch_size)):
        batch = batches[i:i+batch_size]
        batch_texts1 = [each[0] for each in batch]
        batch_texts2 = [each[1] for each in batch]
        batch_pseudo_labels = [each[2] for each in batch]
        batch_labels = [each[3] for each in batch]
        batch_scores = [each[4] for each in batch]
        batch_idx = [each[5] for each in batch]
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
    file_path = 'filtered_samples/{}_{}_CPP.txt'.format(model_name, dataset)
    # frame = pd.read_csv('filtered_samples/{}.csv'.format(dataset))
    tid_list, pred_list, true_list, conf_list = [], [], [], []
    # tid_list, pred_list, true_list, conf_list = [], [], [], frame['Confidence']
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            tid, pred, true, conf = line
            tid_list.append(tid)
            if true[-1] == ';':
                true = true[:-1]
            if pred[-1] == ';':
                pred = pred[:-1]
            if ': ' not in pred:
                pred = pred.replace(':', ': ')
            if ': ' not in true:
                true = true.replace(':', ': ')
            pred_list.append(pred)
            true_list.append(true)
            conf_list.append(float(conf))
    # 转换为 NumPy 数组
    conf_array = np.array(conf_list)
    pred_list = np.array(pred_list)
    true_list = np.array(true_list)
    # ========== 归一化到 [0.5, 1] ==========
    conf_min = conf_array.min()
    conf_max = conf_array.max()
    conf_norm = (conf_array - conf_min) / (conf_max - conf_min)  # 归一化到 [0.5, 1]

    # 计算 0-1 区间的准确率
    bins = np.linspace(0, 1, 11)  # 10个区间边界
    bin_indices = np.digitize(conf_norm, bins) - 1  # 使索引从 0 开始
    print(ner_acc(pred_list, true_list))
    acc_per_bin = {}
    for i in range(10):
        mask = bin_indices == i
        sample_count = np.sum(mask)  # 该区间的样本数量
        if sample_count > 0:
            acc = ner_acc(pred_list[mask], true_list[mask])
            acc_per_bin[f"{bins[i]:.1f}-{bins[i + 1]:.1f}"] = (acc, sample_count)
        else:
            acc_per_bin[f"{bins[i]:.1f}-{bins[i + 1]:.1f}"] = (None, 0)  # 该区间无样本

    # ========== 输出结果 ==========
    print("\n按归一化置信度划分 (0-1 区间)：")
    print(f"{'区间':<12}{'准确率':<10}{'样本数量':<10}")
    for bin_range, (acc, count) in acc_per_bin.items():
        acc_str = f"{acc:.4f}" if acc is not None else "无样本"
        print(f"{acc_str:<10}")

    with open('filtered_samples/{}_{}_CPP_norm.txt'.format(model_name, dataset), 'w') as f:
        for tid, pred, label, conf in zip(tid_list, pred_list, true_list, conf_norm.tolist()):
            f.write(f"{tid}\t{pred}\t{label}\t{conf:.4f}\n")
    print(f"样本CPP已保存到文件")

if __name__ == '__main__':
    args = cfg.Config()
    args.task = 'NER'
    args.batch = 32
    model_name = 'llama3.1-8b'
    args.LLM_path = '/root/autodl-tmp/models/{}'.format(model_name)
    # model, tokenizer = load_base_model_tokenizer(args)
    # T5_based_sample_init(args, 'conll')
    # T5_based_sample_init(args, 'ener')
    # T5_based_sample_init(args, 'wnut')

    # PPL_based_ner_select(model_name, 'conll', model, tokenizer)
    # PPL_based_ner_select(model_name, "ener", model, tokenizer)
    # PPL_based_ner_select(model_name, "wnut", model, tokenizer)
    #
    # # 随后对_CPP.txt样本进行norm
    cal_CPP_acc(model_name, 'conll')
    cal_CPP_acc(model_name, 'ener')
    cal_CPP_acc(model_name, 'wnut')
