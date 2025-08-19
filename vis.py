# 基于KDE与二维表征的可视化，用于探索随机采样样本、bert采样样本与测试样本之间的分布差异
import os

import numpy

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from cfg import Config
from dataloader import *
from scorer import *
from transformers import AutoModel, AutoTokenizer

def draw_scatter():
    dataset = 'civil_comments'
    top = 100

    # 生成测试样本
    train_datas, test_datas = load_datasets(args)
    train_texts = []
    for train_data in train_datas:
        train_texts.append(train_data[0].replace('###', ' '))

    test_texts = []
    print('load {}..............'.format(dataset))
    for test_data in test_datas[dataset][:top]:    # 取top 500测试样本
        test_texts.append(test_data[0].replace('###', ' '))    # ### for NLI tasks

    print(len(test_texts))

    # 生成随机样本
    print('load random samples.........')
    random_texts = []
    file = open('ICL_samples/{}/{}/{}_sim.json'.format('random', args.task, dataset), 'r', encoding='utf-8')
    context_samples = json.load(file)
    for test_sample, context in zip(test_texts, context_samples[:top]):
        for i in range(args.shots):
            for label in context.keys():
                text = train_texts[context[label][i]]
                if text not in random_texts:
                    random_texts.append(text)
    print(len(random_texts))

    # 生成相似样本
    print('load bert sim samples.........')
    bert_texts = []
    file = open('ICL_samples/{}/{}/{}_sim.json'.format('bert', args.task, dataset), 'r', encoding='utf-8')
    context_samples = json.load(file)
    for test_sample, context in zip(test_texts, context_samples[:top]):
        for i in range(args.shots):
            for label in context.keys():
                text = train_texts[context[label][i]]
                if text not in bert_texts:
                    bert_texts.append(text)
    print(len(bert_texts))

    print('load bert unsim samples.........')
    un_bert_texts = []
    file = open('ICL_samples/{}/{}/{}_unsim.json'.format('bert', args.task, dataset), 'r', encoding='utf-8')
    context_samples = json.load(file)
    for test_sample, context in zip(test_texts, context_samples[:top]):
        for i in range(args.shots):
            for label in context.keys():
                text = train_texts[context[label][i]]
                if text not in un_bert_texts:
                    un_bert_texts.append(text)
    print(len(un_bert_texts))

    # 加载bert模型并获取相应的文档嵌入
    # bert_model = BertModel.from_pretrained('/home/songrui/data/bert-base-uncased/').cuda()
    # bert_tokenizer = BertTokenizer.from_pretrained('/home/songrui/data/bert-base-uncased/')
    #
    # # random_embeds = bert_encoder(random_texts, bert_tokenizer, bert_model, 128).cpu().numpy()
    # bert_embeds = bert_encoder(bert_texts, bert_tokenizer, bert_model, 128).cpu().numpy()
    # test_embeds = bert_encoder(test_texts, bert_tokenizer, bert_model, 128).cpu().numpy()
    # un_bert_embeds = bert_encoder(un_bert_texts, bert_tokenizer, bert_model, 128).cpu().numpy()

    bert_model = AutoModel.from_pretrained('/home/songrui/data/GPT-J-6B/').cuda()
    bert_tokenizer = AutoTokenizer.from_pretrained('/home/songrui/data/GPT-J-6B/', trust_remote_code=True, padding_side='left')
    bert_tokenizer.pad_token = bert_tokenizer.eos_token
    random_embeds = bert_encoder(random_texts, bert_tokenizer, bert_model, 4, max_length=64).cpu().numpy()
    bert_embeds = bert_encoder(bert_texts, bert_tokenizer, bert_model, 4, max_length=64).cpu().numpy()
    test_embeds = bert_encoder(test_texts, bert_tokenizer, bert_model, 4, max_length=64).cpu().numpy()
    un_bert_embeds = bert_encoder(un_bert_texts, bert_tokenizer, bert_model, 4, max_length=64).cpu().numpy()

    print(bert_embeds.shape, test_embeds.shape, un_bert_embeds.shape)

    print(test_embeds)
    print(test_texts)

    # 构建数据以及相应的标签
    _labels = ['test_sample']*test_embeds.shape[0] + ['random_sample']*random_embeds.shape[0] + ['bert_sim_sample']*bert_embeds.shape[0]

    _embeds = np.concatenate([test_embeds, random_embeds, bert_embeds], axis=0)

    # 对高维bert输出数据进行降维
    svd = TruncatedSVD(n_components=2, random_state=42)    # 二维才可以估计
    _embeds = svd.fit_transform(_embeds)

    random_df = pd.DataFrame({
        'X': _embeds[:, 0],
        'Y': _embeds[:, 1],
        'Label': _labels
    })
    random_df['Label'] = random_df['Label'].astype('category')

    # 使用类别代码作为颜色索引，但这里我们直接使用类别代码的颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(random_df['Label'].cat.categories)))
    random_df['Color'] = random_df['Label'].cat.codes

    # 绘制散点图，使用颜色代码
    plt.scatter(random_df['X'], random_df['Y'], c=random_df['Color'], s=6)

    # 为图例手动创建标签
    handles = [plt.Line2D([], [], color=color, marker='o', linestyle='None', markersize=4, label=cat)
               for cat, color in zip(random_df['Label'].cat.categories, colors)]

    # 添加图例
    plt.legend(handles=handles)
    # 添加标题
    plt.title('sample distribution for {}'.format(dataset))
    # 显示图表
    plt.show()

def draw_KDE(args, dataset, idx):
    train_datas, test_datas = load_datasets(args)
    test_texts = []
    print('load {}..............'.format(dataset))
    train_texts = [data[0].replace('###', ' ') for data in train_datas[dataset]]
    # train_texts = train_texts[:2000]
    for test_data in test_datas[dataset]:  # 取top 500测试样本
        test_texts.append(test_data[0].replace('###', ' '))  # ### for NLI tasks

    all_texts = test_texts
    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(all_texts)

    # 降维
    print('降维打击~biu biu biu')
    svd = TruncatedSVD(n_components=2, random_state=42)
    reduced = svd.fit_transform(tfidf)
    # df1 = pd.DataFrame(reduced[:len(train_texts), :], columns=['PC1', 'PC2'])
    df2 = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
    df3 = pd.DataFrame(reduced[np.array(idx), :], columns=['PC1', 'PC2'])
    print(df2.shape, df3.shape)
    # 开始绘图
    plt.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(1.5, 1.5), dpi=600)

    # 绘制 df2 的 KDE 曲线
    # sns.kdeplot(x='PC1', y='PC2', data=df1, color='b', label='div_texts', fill=True, thresh=0.05)
    # 绘制 df1 的 KDE 曲线
    sns.kdeplot(x='PC1', y='PC2', data=df2, color='g', label='sim_texts', fill=True, thresh=0.05)
    # 绘制 df3 的 KDE 曲线
    sns.kdeplot(x='PC1', y='PC2', data=df3, color='r', label='current_texts', fill=True, thresh=0.05)
    legend_elements = [
        # plt.Line2D([0], [0], color='blue', lw=3, label='train datas'),
        plt.Line2D([0], [0], color='green', lw=3, label='test datas'),
        plt.Line2D([0], [0], color='red', lw=3, label='selected datas')
    ]
    plt.legend(handles=legend_elements, loc='upper right',
               fontsize='5',  # 设置字体大小
               handlelength=0.1,  # 设置图例标记的长度
               handletextpad=0.3)  # 设置图例标记和文本之间的距离
    plt.gca().patch.set_edgecolor('none')  # 移除边框
    # plt.title(dataset)
    plt.gca().axis('off')
    plt.show()


def scatter_three_subplots(args, dataset, iter_ids_list, save_path=None):
    # 加载数据
    train_datas, test_datas = load_datasets(args)
    print('load {}..............'.format(dataset))

    train_texts = [data[0].replace('###', ' ') for data in train_datas[dataset]][:5000]
    test_texts = [data[0].replace('###', ' ') for data in test_datas[dataset]]

    all_texts = train_texts + test_texts
    # 加载SimCSE模型
    model_path = '/home/tianmingjie/songrui/models/SimCSE/'
    bert_model = AutoModel.from_pretrained(model_path).cuda()
    bert_tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 获取所有句子的嵌入
    all_embeddings = []
    batch_size = 256
    for i in tqdm(range(0, len(all_texts), batch_size)):
        batch = all_texts[i:i + batch_size]
        inputs = bert_tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        all_embeddings.append(embeddings)
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # SVD降维
    print('降维打击~biu biu biu')
    svd = TruncatedSVD(n_components=2, random_state=42)
    reduced = svd.fit_transform(all_embeddings)

    # 可视化准备
    palette = {'source': '#c7e9c0', 'target': '#2171b5', 'selected samples': '#f768a1'}
    sns.set(style='whitegrid', font_scale=0.8)
    # plt.rcParams['font.family'] = 'Times New Roman'

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), dpi=600)
    plt.subplots_adjust(
        left=0.03,  # 左边距
        right=0.97,  # 右边距
        top=0.9,  # 上边距
        bottom=0.1,  # 下边距（留给图例）
        wspace=-0.2  # 子图之间的水平间距，默认是0.2，设小点更紧凑
    )
    for idx, iter_ids in enumerate(iter_ids_list):
        df1 = pd.DataFrame(reduced[:len(train_texts), :], columns=['PC1', 'PC2'])
        df2 = pd.DataFrame(reduced[len(train_texts):, :], columns=['PC1', 'PC2'])
        df3 = pd.DataFrame(reduced[len(train_texts) + np.array(iter_ids), :], columns=['PC1', 'PC2'])
        df1['Type'] = 'source'
        df2['Type'] = 'target'
        df3['Type'] = 'selected samples'

        df_all = pd.concat([df1, df2, df3], ignore_index=True)
        ax = axes[idx]
        if idx == 0:
            sns.scatterplot(data=df_all, x='PC1', y='PC2', hue='Type', palette=palette,
                            s=15, alpha=0.8, ax=ax, legend=True, edgecolor='lightgray', linewidth=0.5)
            leg = ax.get_legend()
            if leg:
                leg.remove()  # 删除子图的图例（隐藏它）
        else:
            sns.scatterplot(data=df_all, x='PC1', y='PC2', hue='Type', palette=palette,
                            s=15, alpha=0.8, ax=ax, legend=False, edgecolor='lightgray', linewidth=0.5)
        # ax.set_title(f'j={idx}', fontsize=12)
        ax.set_xlabel(f'j={idx}', fontsize=18)
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    # 添加统一图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1),
        fontsize=18,
        frameon=True,
        ncol=3,  # ✅ 水平排列（每行放3个）
        columnspacing=0.5,  # ✅ 控制列间距
        handletextpad=0.2,  # ✅ 控制图例点与文字的间距
        markerscale=2,  # ✅ 放大图例中“点”的大小
        title=None  # ✅ 去掉图例标题
    )

    plt.savefig("vis-sst.pdf", bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 首先需要获取相应迭代过程中的idx
    args = Config()
    args.task = 'SA'
    dataset = "sst5"
    iter_ids_list = []

    thresholds = [0.95, 0.85, 0.75]
    for i in range(3):
        ids = []
        with open(f'iter_results/llama3.2-3b_{dataset}_iter{i}.txt', 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if float(parts[-1]) > thresholds[i]:
                    ids.append(int(parts[0]))
        iter_ids_list.append(ids)

    # 绘制合并图
    scatter_three_subplots(args, dataset, iter_ids_list)