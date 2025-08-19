
class Config:
    task = 'NLI'
    assert task in ['NLI', 'SA', 'TD']
    LLM = 'gpt2'
    LLM_path = '/root/autodl-tmp/models/{}'.format(LLM)
    shots = 3
    batch = 20
    score_func = 'bert'
    if_sim_first = True     # 是不是相似度优先，如果True，则先测试语义相似度高的样本；如果不是则测试相似度低的样本
    select_from_filter = False   # 从测试样本的过滤数据中进行数据选取
    test_datasets = {
        'SA': ['dynasent', 'semeval', 'sst5', 'amazon'],
        'TD': ['adv_civil', 'implicit_hate', 'toxigen', 'civil_comments'],
        'NLI': ['anli', 'contract_nli', 'wanli', 'mnli'],
        'NER': ['conll', 'ener', 'wnut']
    }
    source_datasets = {
        'SA': 'amazon',
        'TD': 'civil_comments',
        'NLI': 'mnli',
        'NER': 'fewnerd'
    }
    label_space = {
        'SA': {0: 'negative', 1: 'positive', 2: 'neutral'},
        'TD': {0: 'benign', 1: 'toxic'},
        'NLI': {0: 'entailment', 1:'neutral', 2:'contradiction'},
        'NER': {0: "O", 1: "B-organization", 2: "I-organization", 3: "B-location", 4: "I-location",
                5: "B-person", 6: "I-person", 7: "B-product", 8: "I-product", 9: "B-art", 10: "I-art"}
    }
    iter_times = 3     # 迭代次数设置为3
    model2confidence = 0.8
    gamma = 0.8
    max_sen_len = 100