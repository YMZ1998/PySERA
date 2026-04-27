# -- coding: utf-8 --
import os

import matplotlib.pyplot as plt
import numpy as np
# 导入常用库
import pandas as pd
from sklearn.linear_model import LassoCV  # 导入Lasso工具包LassoCV
from sklearn.preprocessing import StandardScaler  # 标准化工具包StandardScaler


def split_df(df, ratio):
    # 用来分割数据集，保留一定比例的数据集当做最终的测试集
    cut_idx = int(round(ratio * df.shape[0]))
    print(cut_idx)
    data_test, data_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
    return (data_train, data_test)


if __name__ == '__main__':
    data_path = r"E:\AI-data2\nnUNet_raw\Dataset0978_test"
    class_dic = ["A", "B"]

    test_ratio = 0.15
    random_state = 114  # 固定随机种子
    A_data = pd.read_excel(os.path.join(data_path, "A", "results", 'extracted_radiomics_features.xlsx'))
    B_data = pd.read_excel(os.path.join(data_path, "B", "results", 'extracted_radiomics_features.xlsx'))

    A_data.insert(0, 'label', 1)  # 插入标签
    B_data.insert(0, 'label', 0)  # 插入标签

    A_data = A_data.sample(frac=1.0, random_state=random_state)  # 全部打乱
    B_data = B_data.sample(frac=1.0, random_state=random_state)  # 全部打乱

    # # 因为有些特征是字符串，直接删掉
    # cols = [x for i, x in enumerate(A_data.columns) if type(A_data.iat[1, i]) == str]
    # A_data = A_data.drop(cols, axis=1)
    # cols = [x for i, x in enumerate(B_data.columns) if type(B_data.iat[1, i]) == str]
    # B_data = B_data.drop(cols, axis=1)
    head = A_data.columns
    cols_to_remove = []
    for h in head:
        if "dim" in h:
            cols_to_remove.append(h)
    print("cols_to_remove: ", cols_to_remove)
    # 如果要删除多列
    # cols_to_remove = ['vox_dim_z_init_img', 'img_dim_z_init_img', 'img_dim_z_interp_img']
    A_data = A_data.drop(columns=cols_to_remove)
    B_data = B_data.drop(columns=cols_to_remove)

    # 去掉字符串列
    cols_str_A = [col for col in A_data.columns if isinstance(A_data[col].iloc[1], str)]
    A_data = A_data.drop(cols_str_A, axis=1)

    cols_str_B = [col for col in B_data.columns if isinstance(B_data[col].iloc[1], str)]
    B_data = B_data.drop(cols_str_B, axis=1)

    # 去掉全是 NaN 的列
    A_data = A_data.dropna(axis=1, how='all')
    B_data = B_data.dropna(axis=1, how='all')

    # 去掉包含 NaN 的列
    A_data = A_data.dropna(axis=1, how='any')
    B_data = B_data.dropna(axis=1, how='any')

    # 现在 A_data 和 B_data 只剩下完整的数值特征列
    print("A_data shape:", A_data.shape)
    print("B_data shape:", B_data.shape)

    A_data_train, A_data_test = split_df(A_data, test_ratio)  # 返回train 和test数据集
    B_data_train, B_data_test = split_df(B_data, test_ratio)  # 返回train 和test数据集

    # 保存测试集为cvs 后面最终验证使用
    A_data_test.to_csv('A_test.csv', index=False)
    B_data_test.to_csv('B_test.csv', index=False)
    # 查看总数据类别是否平衡
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sns.set()
    total_data = pd.concat([A_data, B_data])
    sns.countplot(x='label',hue='label',data=total_data)
    plt.show()
    print(total_data['label'].value_counts())
    '''
    # 把hgg_data_train 和lgg_data_train 并在一起并且打乱。
    # 查看总体数据情况
    data = pd.concat([A_data_train, B_data_train])
    data = data.sample(frac=1.0, random_state=random_state)  # 全部打乱
    print("一共有{}行特征数据".format(len(data)))
    print("一共有{}列不同特征".format(data.shape[1]))
    # 再把特征值数据和标签数据分开
    x = data[data.columns[1:]]
    y = data['label']
    # 取X的5行看看数据
    # print(x.head())
    # 通过T检验从106个特征筛选

    counts = 0
    columns_index = []
    print("开始进行T检验")
    use_tt = 1
    if use_tt:
        from scipy.stats import levene, ttest_ind

        for column_name in A_data_train.columns[1:]:
            if levene(A_data_train[column_name], B_data_train[column_name])[1] > 0.05:
                if ttest_ind(A_data_train[column_name], B_data_train[column_name], equal_var=True)[1] < 0.05:
                    columns_index.append(column_name)
            else:
                if ttest_ind(A_data_train[column_name], B_data_train[column_name], equal_var=False)[1] < 0.05:
                    columns_index.append(column_name)
    else:

        for column_name in A_data_train.columns[1:]:
            columns_index.append(column_name)
    print("columns_index: ", columns_index)
    print("筛选后剩下的特征数：{}个".format(len(columns_index)))

    from kydavra import MUSESelector

    # 数据只保留从T检验筛选出的特征数据，重新组合成data
    if not 'label' in columns_index:
        columns_index = ['label'] + columns_index
    A_train = A_data_train[columns_index]
    B_train = B_data_train[columns_index]

    data = pd.concat([A_train, B_train])
    data = data.sample(frac=1.0, random_state=random_state)  # 全部打乱
    # 缪斯选择器筛选特征
    # 主要思想是在一个特征下，不同 类别的分布是有明显差异的，如果各个类别都是均匀分布，那这个特征就没有用。
    print("开始进行MUSE选择器")
    max_columns_num = 60  # 这个值是人工定义值
    selector = MUSESelector(num_features=max_columns_num)
    # selector = LassoSelector()
    # columns_index = selector.select(data, 'label')

    print("筛选后剩下的特征数：{}个".format(len(columns_index)))
    # Lasso
    # 数据只保留从T检验筛选出的特征数据，重新组合成data
    if not 'label' in columns_index:
        columns_index = ['label'] + columns_index
    A_train = A_data_train[columns_index]
    B_train = B_data_train[columns_index]

    data = pd.concat([A_train, B_train])
    data = data.sample(frac=1.0, random_state=random_state)  # 全部打乱

    # 再把特征值数据和标签数据分开
    x = data[data.columns[1:]]
    y = data['label']
    # 先保存X的列名
    columnNames = x.columns

    lassoCV_x = x.astype(np.float32)  # 把x数据转换成np.float格式
    lassoCV_y = y

    standardscaler = StandardScaler()
    lassoCV_x = standardscaler.fit_transform(lassoCV_x)  # 对x进行均值-标准差归一化
    lassoCV_x = pd.DataFrame(lassoCV_x, columns=columnNames)  # 转 DataFrame 格式

    # 形成5为底的指数函数
    # 5**（-3） ~  5**（-2）
    # alpha_range = np.logspace(-3,-2,50,base=5)
    alpha_range = np.logspace(-3, 1, 50)
    # alpha_range在这个参数范围里挑出aplpha进行训练，cv是把数据集分5分，进行交叉验证，max_iter是训练1000轮
    lassoCV_model = LassoCV(alphas=alpha_range, cv=5, max_iter=100000)
    # 进行训练
    lassoCV_model.fit(lassoCV_x, lassoCV_y)
    # 打印训练找出来的入值
    print(lassoCV_model.alpha_)
    # print("Coefficient of the model:{}".format(lassoCV_model.coef_) )
    # print("intercept of the model:{}".format(lassoCV_model.intercept_))

    coef = pd.Series(lassoCV_model.coef_, index=columnNames)
    print("从原来{}个特征，筛选剩下{}个".format(len(columnNames), sum(coef != 0)))
    print("分别是以下特征")
    print(coef[coef != 0])
    index = coef[coef != 0].index
    lassoCV_x = lassoCV_x[index]
    # lassoCV_x.head()

    # 绘制特征相关系数热力图
    import seaborn as sns

    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(lassoCV_x.corr(), annot=True, cmap='coolwarm', annot_kws={'size': 10, 'weight': 'bold', },
                ax=ax)  # 绘制混淆矩阵
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, va="top", ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    # plt.show()
    # 画一个特征系数的柱状图
    weight = coef[coef != 0].to_dict()
    # 根据值大小排列一下
    weight = dict(sorted(weight.items(), key=lambda x: x[1], reverse=False))
    plt.figure(figsize=(8, 6))  # 设置画布的尺寸
    plt.title('characters classification weight', fontsize=15)  # 标题，并设定字号大小
    plt.xlabel(u'weighted value', fontsize=14)  # 设置x轴，并设定字号大小
    plt.ylabel(u'feature')
    plt.barh(range(len(weight.values())), list(weight.values()), tick_label=list(weight.keys()), alpha=0.6,
             facecolor='blue', edgecolor='black', label='feature weight')
    plt.legend(loc=4)  # 图例展示位置，数字代表第几象限
    # plt.show()
    # 绘制误差棒图
    MSEs = lassoCV_model.mse_path_
    mse = list()
    std = list()
    for m in MSEs:
        mse.append(np.mean(m))
        std.append(np.std(m))

    plt.figure(figsize=(8, 6))
    plt.errorbar(lassoCV_model.alphas_, mse, std, fmt='o:', ecolor='lightblue',
                 elinewidth=3, ms=5, mfc='wheat', mec='salmon', capsize=3)
    plt.axvline(lassoCV_model.alpha_, color='red', ls='--')
    plt.title('Errorbar')
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    # plt.show()
    # 这个图显示随着lambda 的变化，系数的变化走势
    x = data[data.columns[1:]]
    y = data['label']
    # 先保存X的列名
    columnNames = x.columns
    lassoCV_x = x.astype(np.float32)  # 把x数据转换成np.float格式
    lassoCV_y = y
    lassoCV_x = standardscaler.transform(lassoCV_x)  # 对x进行均值-标准差归一化
    lassoCV_x = pd.DataFrame(lassoCV_x, columns=columnNames)  # 转 DataFrame 格式
    coefs = lassoCV_model.path(lassoCV_x, lassoCV_y, alphas=alpha_range, max_iter=1000)[1].T
    plt.figure(figsize=(8, 6))
    plt.plot(lassoCV_model.alphas, coefs, '-')
    plt.axvline(lassoCV_model.alpha_, color='red', ls='--')
    plt.xlabel('Lambda')
    plt.ylabel('coef')
    # plt.show()
    # 随机森林分类
    model_name = 'svm'
    print('Model:{}', model_name)
    from sklearn.model_selection import train_test_split  # 分割训练集和验证集
    from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
    import joblib  # 用来保存 sklearn 训练好的模型

    # 把数据分成训练集和验证集，7：3比例
    index_ = coef[coef != 0].index
    rforest_x = x[index_]
    rforest_y = y
    standardscaler = StandardScaler()
    rforest_x = standardscaler.fit_transform(rforest_x)  # 对x进行均值-标准差归一化
    x_train, x_test, y_train, y_test = train_test_split(rforest_x, rforest_y, test_size=0.1)
    if model_name == 'forest':
        model = RandomForestClassifier(n_estimators=30, random_state=random_state).fit(x_train, y_train)
    elif model_name == 'svm':
        from sklearn import svm

        model = svm.SVC(kernel='rbf', gamma='auto', probability=True).fit(x_train, y_train)
    elif model_name == 'adaboost':
        from sklearn.ensemble import AdaBoostClassifier

        # n_estimators表示要组合的弱分类器个数；
        # algorithm可选{‘SAMME’, ‘SAMME.R’}，默认为‘SAMME.R’，表示使用的是real boosting算法，‘SAMME’表示使用的是discrete boosting算法
        model = AdaBoostClassifier(n_estimators=100, algorithm='SAMME.R')
        model.fit(x_train, y_train)
    elif model_name == 'decisiontree':
        from sklearn.tree import DecisionTreeClassifier

        # criterion可选‘gini’, ‘entropy’，默认为gini(对应CART算法)，entropy为信息增益（对应ID3算法）
        model = DecisionTreeClassifier(criterion='gini')
        model.fit(x_train, y_train)
    elif model_name == 'bayes':
        from sklearn.naive_bayes import GaussianNB

        model = GaussianNB()
        model.fit(x_train, y_train)
    elif model_name == 'MLP':
        from sklearn.neural_network import MLPClassifier

        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 2), random_state=1)
        model.fit(x_train, y_train)
    else:
        print('No this method')
    score = model.score(x_test, y_test)
    print("在验证集上的准确率：{}".format(score))
    # 模型保存
    model_path = os.path.join(data_path, 'model_' + model_name + '.model')
    joblib.dump(model, model_path)
    # 在测试集验证
    import joblib

    A_test = pd.read_csv('./A_test.csv')
    B_test = pd.read_csv('./B_test.csv')
    # 再把特征值数据和标签数据分开
    data_test = pd.concat([A_test, B_test], axis=0)

    x_test_data = data_test[data_test.columns[1:]]
    # 只提取之前Lasso 筛选后的
    index = coef[coef != 0].index
    x_test_data = x_test_data[index]

    columnNames = x_test_data.columns
    x_test_data = x_test_data.astype(np.float32)

    x_test_data = standardscaler.transform(x_test_data)  # 均值-标准差归一化
    x_test_data = pd.DataFrame(x_test_data, columns=columnNames)
    y_test_data = data_test['label']

    print("测试集一共有{}行特征数据，{}列不同特征,包含A:{}例，B:{}例".format(len(x_test_data), x_test_data.shape[1],
                                                                           len(A_data_test),
                                                                           len(B_data_test)))
    # 加载保存后的模型，然后进行预测
    model = joblib.load(model_path)  # 这是自己训练模型，记得替换自己的。
    score = model.score(x_test_data, y_test_data)
    print("在测试集上的准确率：{}".format(score))

    # 绘制混淆矩阵
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # 绘制混淆矩阵图方法1
    import seaborn as sns

    predict_label = model.predict(x_test_data)  # 预测的标签
    label = y_test_data.to_list()  # 真实标签
    confusion = confusion_matrix(label, predict_label)  # 计算混淆矩阵

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion, cmap='Blues_r', annot=True, annot_kws={'size': 20, 'weight': 'bold', })  # 绘制混淆矩阵
    plt.xlabel('Predict')
    plt.ylabel('True')
    # plt.show()

    # 绘制混淆图方法2,一行代码
    # plot_confusion_matrix(model_forest, x_test_data, y_test_data,values_format='d',cmap='GnBu_r')

    print("混淆矩阵为：\n{}".format(confusion))
    print("\n计算各项指标：")
    print(classification_report(label, predict_label))

    # # 绘制ROC曲线,方法1
    # from sklearn.metrics import roc_curve, auc
    #
    # kind = {'A': 1, "B": 0}
    # # model = joblib.load(model_path)  # 这是自己训练模型，记得替换自己的
    # label = y_test_data.to_list()  # 真实标签
    # y_predict = model.predict_proba(x_test_data)  # 得到标签0和1对应的概率
    # fpr, tpr, threshold = roc_curve(label, y_predict[:, kind['B']], pos_label=kind['B'])
    # roc_auc = auc(fpr, tpr)  # 计算auc的
    # fpr1, tpr1, threshold = roc_curve(label, y_predict[:, kind['A']], pos_label=kind['A'])
    # roc_auc1 = auc(fpr1, tpr1)  # 计算auc的
    # plt.figure(figsize=(6, 5))
    # plt.plot(fpr, tpr, marker='o', markersize=5, label='B')
    # plt.plot(fpr1, tpr1, marker='*', markersize=5, label='A')
    # plt.title("B AUC:{:.2f}, A AUC:{:.2f}".format(roc_auc, roc_auc1))
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.legend(loc=4)
    # # plt.show()
    # # 绘制ROC方法2,两行代码
    # from sklearn.metrics import RocCurveDisplay
    #
    # ax1 = RocCurveDisplay.from_estimator(model, x_test_data, y_test_data, name='B', pos_label=0)
    # RocCurveDisplay.from_estimator(model, x_test_data, y_test_data, ax=ax1.ax_, name='ACC', pos_label=1)
    #
    # # 绘制PR曲线，一行代码
    # from sklearn.metrics import PrecisionRecallDisplay
    #
    # PrecisionRecallDisplay.from_estimator(model, x_test_data, y_test_data, name='PR', pos_label=0)
    # plt.show()
