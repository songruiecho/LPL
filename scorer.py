# 根据文本相似性查找与test样本最相似（最不像）的训练样本，注意要保证样本数量的平衡【也就是不同类别的样本数量应该一致为K-shots】  BOSS论文里面为5
import math
from collections import Counter
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KernelDensity
# import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import torch
from matplotlib.lines import Line2D

def random_scorer(train_datas, test_datas):
    '''
    :param train_datas:
    :param test_datas:
    :return:  随机生成一个相似度矩阵
    '''
    n, m = len(train_datas), len(test_datas)
    similarity_matrix = np.zeros((m, n), dtype=int)
    # 对每一行，生成0到m-1的随机排列并填充到矩阵中
    for i in range(m):
        # np.random.permutation(m) 生成一个长度为m的数组，包含0到m-1的随机排列
        row = np.random.permutation(n)
        # 将生成的随机排列赋值给矩阵的当前行
        similarity_matrix[i, :] = row
    return similarity_matrix

def BM25_scorer(train_datas, test_datas):
    texts = train_datas + test_datas
    # 计算文本中的总词数
    total_words = sum(len(text.split()) for text in texts)
    # 计算文本中每个词汇项的文档频率（DF）
    df = Counter()
    for text in texts:
        words = set(text.split())
        for word in words:
            df[word] += 1
    # 计算文档总数
    N1, N2 = len(train_datas), len(test_datas)
    # 设置BM25参数
    k1 = 1.5  # 调节因子，控制文档长度对相似性的影响
    b = 0.75  # 调节因子，控制文档长度对相似性的影响
    # 初始化相似性矩阵
    similarity_matrix = np.zeros((N1, N2))
    # 计算每对文本之间的BM25相似性
    for i in tqdm(range(N1)):
        for j in range(N2):
            text1 = train_datas[i].split()
            text2 = test_datas[j].split()
            intersection = set(text1) & set(text2)  # 交集词汇项
            similarity = 0
            for word in intersection:
                idf = math.log((N1 - df[word] + 0.5) / (df[word] + 0.5) + 1.0)  # 计算逆文档频率
                tf1 = text1.count(word) / len(text1)  # 计算词频
                tf2 = text2.count(word) / len(text2)
                similarity += idf * ((tf1 * (k1 + 1)) / (tf1 + k1 * (1 - b + b * (len(text1) / total_words))) *
                                     (tf2 * (k1 + 1)) / (tf2 + k1 * (1 - b + b * (len(text2) / total_words))))
            similarity_matrix[i][j] = similarity
    return similarity_matrix

def tfidf_scorer(train_datas, test_datas):
    # 将文本转换为TF-IDF向量
    train_datas = [each.strip().replace('Premise: ', '').replace(' Hypothesis: ', '') for each in train_datas]
    test_datas = [each.strip().replace('Premise: ', '').replace(' Hypothesis: ', '') for each in test_datas]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(train_datas + test_datas)  # 合并两个数组以共享词汇表

    # 分割回原始数组
    X1 = X[:len(train_datas)]
    X2 = X[len(train_datas):]

    # 计算余弦相似度矩阵
    # 注意：这里计算的是text1中每个元素与text2中每个元素的相似度
    similarity_matrix = cosine_similarity(X1, X2)
    return similarity_matrix


def bert_encoder(texts, tokenizer, model, batch_size, padding=True, truncation=True, max_length=128):
    """ 使用 BERT 模型编码文本列表，支持批处理和CUDA加速 """
    # 确保模型在CUDA设备上
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=padding, truncation=truncation, max_length=max_length,
                           return_tensors='pt')

        # 将输入张量移动到CUDA设备上
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.cuda()
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 标记的嵌入，并可选地将其移回CPU以节省GPU内存
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)

def gpt_encoder(texts, tokenizer, model, batch_size, padding=True, truncation=True, max_length=100):
    """ 使用 BERT 模型编码文本列表，支持批处理和CUDA加速 """
    # 确保模型在CUDA设备上
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=padding, truncation=truncation, max_length=max_length,
                           return_tensors='pt')

        # 将输入张量移动到CUDA设备上
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.cuda()
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 标记的嵌入，并可选地将其移回CPU以节省GPU内存
        all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)

def bert_scorer(train_datas, test_datas, bert_model, bert_tokenizer):
    train_embeddings = bert_encoder(train_datas, bert_tokenizer, bert_model, batch_size=256).cpu().numpy()
    # 编码 test_datas
    test_embeddings = bert_encoder(test_datas, bert_tokenizer, bert_model, batch_size=256).cpu().numpy()
    # train_reshaped = train_embeddings.unsqueeze(1).repeat(1, test_embeddings.size(0), 1)
    # test_reshaped = test_embeddings.unsqueeze(0).repeat(train_embeddings.size(0), 1, 1)
    similarity_matrix = cosine_similarity(train_embeddings, test_embeddings)

    return similarity_matrix

def gpt_scorer(train_datas, test_datas, model, gpt_tokenizer):
    train_embeddings = gpt_encoder(train_datas, gpt_tokenizer, model, batch_size=32).cpu().numpy()
    # 编码 test_datas
    test_embeddings = gpt_encoder(test_datas, gpt_tokenizer, model, batch_size=32).cpu().numpy()
    # train_reshaped = train_embeddings.unsqueeze(1).repeat(1, test_embeddings.size(0), 1)
    # test_reshaped = test_embeddings.unsqueeze(0).repeat(train_embeddings.size(0), 1, 1)
    similarity_matrix = cosine_similarity(train_embeddings, test_embeddings)

    return similarity_matrix

def draw_LDE(train_datas, test_datas, testset):
    # 将文本转换为TF-IDF向量
    vectorizer = TfidfVectorizer()
    datas = vectorizer.fit_transform(train_datas + test_datas)  # 合并两个数组以共享词汇表
    # 分割回原始数组
    train = datas[:len(train_datas)]
    test = datas[len(train_datas):]

    print(train.shape, test.shape)

    # 假设train和test是两组一维数据
    np.random.seed(0)  # 为了可重复性设置随机种子

    # 为了降低计算成本，先降维
    print('降维打击~biu biu biu')
    svd = TruncatedSVD(n_components=2, random_state=42)    # 二维才可以估计
    train = svd.fit_transform(train)
    test = svd.fit_transform(test)

    print(train.shape, test.shape)


    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(1.5, 1.5), dpi=600)
    # 计算 KDE
    # kde_train = gaussian_kde(train.T)
    # kde_test = gaussian_kde(test.T)

    # 定义绘图网格
    # x = np.linspace(train[:, 0].min(), train[:, 0].max(), 100)
    # y = np.linspace(train[:, 1].min(), train[:, 1].max(), 100)
    # X, Y = np.meshgrid(x, y)
    # Z_train = kde_train(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    # Z_test = kde_test(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    #
    # # 绘制 KDE
    # fig, ax = plt.subplots()
    # contour_train = ax.contourf(X, Y, Z_train, levels=30, cmap='Greens', alpha=0.8, label='Train')
    # contour_test = ax.contourf(X, Y, Z_test, levels=30, cmap='Reds', alpha=0.8, label='Test')

    sns.kdeplot(x=train[:,0], y=train[:, 1], cmap="Blues", fill=True, alpha=0.99, label="amazon")
    sns.kdeplot(x=test[:,0], y=test[:, 1], cmap="Reds", fill=True, alpha=0.8, label=testset)

    # 添加图例和颜色条
    # 自定义图例项
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='amazon'),
        Line2D([0], [0], color='red', lw=2, label=testset)
    ]
    plt.legend(handles=legend_elements, loc='upper center',
               fontsize='small',  # 设置字体大小
               handlelength=1,  # 设置图例标记的长度
               handletextpad=0.1)  # 设置图例标记和文本之间的距离
    plt.gca().patch.set_edgecolor('none')  # 移除边框
    plt.gca().axis('off')
    # 添加图例
    # plt.legend(handles=legend_elements)
    # plt.legend(['Amazon', testset])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False,
                    labelleft=False)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    # 显示图形
    plt.show()