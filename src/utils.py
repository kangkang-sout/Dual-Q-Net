#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：qnn4cdm -> utils.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/9/21
@Desc   ：
==================================================
"""
import time
import warnings
from collections import Counter
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from scipy.stats import norm
from rpy2 import robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr

# R GDINA 2 PY
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import Literal

from IPython.core.interactiveshell import InteractiveShell

from rmodels import RCDM2py

InteractiveShell.ast_node_interactivity = 'all'
warnings.filterwarnings("ignore")

gdina = importr('GDINA')
cdm = importr('CDM')
edm = importr('edmdata')


def simGDINA4R(N: int, q: ndarray, g=0.1, s=0.1,
              model: Literal["DINA", "GDINA"]='DINA',
              attr: Literal["uniform", "higher_order", "mvnorm"]='uniform', **kwargs) -> (ndarray, ndarray):
    """
    获取模拟数据
    :param N: 样本量
    :param q: Q矩阵
    :param g: 猜测率，可以是固定值，也可以是区间值，区间值会进行均匀采样
    :param s: 失误率
    :param model: DINA、GDINA
    :param attr: 属性的分布情况，包括三种类型的分布，也可以直接传入已有的属性矩阵
    :param kwargs: seed，可以设置随机种子
    :return: resp2py, attr2py  模拟的作答数据与模拟的技能掌握模式
    """
    model = robjects.StrVector([model])

    if 'seed' in kwargs.keys():
        robjects.r('set.seed')(kwargs.get('seed'))

    num_rows_q = q.shape[0]
    num_col_q = q.shape[1]

    N = numpy2ri.py2rpy(N)
    q = numpy2ri.py2rpy(q)

    if isinstance(g, float):
        g = [g] * num_rows_q
    elif isinstance(g, (tuple, list)):
        if len(g) == 2:
            g = np.linspace(g[0]+g[0]*0.1, g[1]-g[1]*0.1, num_rows_q)
    if isinstance(s, float):
        s = [s] * num_rows_q
    elif isinstance(s, (tuple, list)):
        if len(s) == 2:
            s = np.linspace(s[0]+s[0]*0.1, s[1]-s[1]*0.1, num_rows_q)
    gs = pd.DataFrame({'guessing': g, 'slip': s})
    gs = numpy2ri.py2rpy(gs.values)

    if isinstance(attr, str):
        if attr == 'uniform':
            attr = numpy2ri.py2rpy(attr)
            sim = gdina.simGDINA(N, q, gs_parm=gs, model=model)
        elif attr == 'higher_order':
            theta = np.random.randn(N)
            theta = numpy2ri.py2rpy(theta)
            a = [1] * num_col_q
            b = np.linspace(-2,2, num_col_q)
            df = pd.DataFrame({'a': a, 'b': b})
            lamb = numpy2ri.py2rpy(df.values)
            sim = gdina.simGDINA(N, q, gs_parm=gs, model=model, att_dist='higher.order',
                                higher_order_parm=robjects.ListVector({'theta': theta, 'lambda': lamb}))
        elif attr == 'mvnorm':
            cutoffs = robjects.FloatVector(norm.ppf(np.array(range(1, num_col_q+1))/(num_col_q+1)))
            mean = robjects.FloatVector(np.zeros(num_col_q).astype(int))
            matrix = np.eye(num_col_q) * 0.5 + np.ones((num_col_q, num_col_q)) * 0.5
            vcov = numpy2ri.py2rpy(matrix)
            sim = gdina.simGDINA(N, q, gs_parm=gs, model=model, att_dist="mvnorm",
                                mvnorm_parm=robjects.ListVector({'mean': mean, 'sigma': vcov, 'cutoffs': cutoffs}))
    elif isinstance(attr, (ndarray, DataFrame)):
        N = numpy2ri.py2rpy(attr.shape[0])
        attr2py = numpy2ri.py2rpy(attr)
        sim = gdina.simGDINA(N, q, gs_parm=gs, model=model, attribute=attr2py)

    attr = gdina.extract(sim, what='attribute')
    resp = gdina.extract(sim, what='dat')
    attr2py = numpy2ri.rpy2py(attr)
    resp2py = numpy2ri.rpy2py(resp)
    return resp2py, attr2py


def get_data4cdm(data_name: str):
    data = robjects.r(data_name)
    dat = pandas2ri.rpy2py(data.rx2('data')).values
    q = numpy2ri.rpy2py(data.rx2('q.matrix'))
    return q, dat


def get_data4edm(q='qmatrix_probability_part_one', data_name='items_probability_part_one'):
    # items_probability_part_one
    # qmatrix_probability_part_one
    q = robjects.r(q)
    dat = robjects.r(data_name)
    dat2numpy = numpy2ri.rpy2py(dat)
    q2numpy = numpy2ri.rpy2py(q)
    return q, dat


def get_data4gdina(data_name: str, sim=True) -> ndarray:
    """
    从R中的GDINA包获取数据集
    :param data_name: 数据集名称
    :return: Q矩阵q和作答反应数据dat
    """
    data_name_robj = robjects.r(data_name)  # 数据集的名称需要先转换为R对象
    get_data4gdina_code = """
            function(data_name, sim=TRUE){
                if (sim == TRUE){
                    q = data_name$simQ
                    dat = data_name$simdat
                }else{
                    q = data_name$Q
                    dat = data_name$dat
                }
                return(list(q=q, dat=dat))
            }
        """
    get_data4gdina = robjects.r(get_data4gdina_code)
    data = get_data4gdina(data_name_robj, sim)  # data为R List对象中存放了Q矩阵和作答反应数据dat
    q = np.array(data.rx('q')).squeeze()  # 通过.rx方法获取data对象中的数据，.squeeze()减少不必要的数据维度
    dat = np.array(data.rx('dat')).squeeze()
    return q, dat


def class_rate4gdina(y: ndarray, y_hat: ndarray, par_digit=-1, decimals=3) -> [float, float]:
    """
    计算分类准去率AAR和PAR
    :param y: 原始属性掌握模式
    :param y_hat: 模型预测的属性掌握模式
    :return: (AAR, PAR)
    """
    y = y.cpu().detach().numpy() if isinstance(y, (torch.Tensor)) else y
    y_hat = y_hat.cpu().detach().numpy() if isinstance(y_hat, (torch.Tensor)) else y_hat
    attr = numpy2ri.py2rpy(np.atleast_2d(y))  # gdina.ClassRate函数要求参数必须是二维矩阵
    attr_hat = numpy2ri.py2rpy(np.atleast_2d(y_hat))
    PCA_PCV = gdina.ClassRate(attr, attr_hat)
    AAR = np.array(PCA_PCV.rx('PCA')).squeeze()  # .squeeze()降维只取矩阵中的数字
    # PAR = np.array(PCA_PCV.rx('PCV')).squeeze()[par:]  # 只取向量相似度即只有所有属性都一样的两个向量才认为是相等
    PAR = np.array(PCA_PCV.rx('PCV')).squeeze()[int(par_digit):]

    return AAR.round(decimals), PAR.round(decimals)


def oversample_data(features: ndarray, labels: ndarray, target_ratio=0.6):
    """
    对数据集进行过采样，使得少数类别的样本数量接近指定的目标比例。

    参数:
    features (ndarray): 特征数据数组，每行代表一个样本的特征。
    labels (ndarray): 标签数据数组，每行代表一个样本的标签。
    target_ratio (float): 目标过采样比例，即少数类别样本数量与多数类别样本数量的比例。

    返回:
    oversampled_features (ndarray): 过采样后的特征数据数组。
    oversampled_labels (ndarray): 过采样后的标签数据数组。
    """
    unique_labels, label_counts = np.unique(labels, axis=0, return_counts=True)
    max_label_count = np.max(label_counts)

    oversampled_features = features.copy()
    oversampled_labels = labels.copy()

    for label in unique_labels:
        label_indices = np.where((labels == label).all(axis=1))[0]
        num_to_add = int(max_label_count * target_ratio) - label_indices.shape[0]

        if num_to_add > 0:
            random_indices = np.random.choice(label_indices, size=num_to_add, replace=True)
            oversampled_features = np.vstack((oversampled_features, features[random_indices]))
            oversampled_labels = np.vstack((oversampled_labels, labels[random_indices]))

    return oversampled_features, oversampled_labels


def count_unique_rows(matrix: ndarray):
    """
    统计矩阵中不重复的行个数
    :param matrix:
    :return:
    """
    unique_rows, counts = np.unique(matrix, axis=0, return_counts=True)
    unique_row_counts = dict(zip(map(tuple, unique_rows), counts))
    return unique_row_counts


def attr_dis_count(attr: ndarray):
    """
    统计矩阵中不重复的每一行的个数，并打印柱状图
    :param attr:
    :return:
    """
    data_dict = Counter(tuple(row) for row in attr.astype(int))
    labels = [str(key) for key in data_dict.keys()]
    values = list(data_dict.values())
    plt.bar(labels, values)
    for i, value in enumerate(values):
        plt.text(i, value, str(value), ha='center', va='bottom')

    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Chart')
    plt.xticks(rotation=35)

    plt.show()


def get_train_data(feature_data, label_data, test_size=0.25, random_state=42, shuffle=True):
    """
    随机划分测试集和训练集
    :param feature_data:
    :param label_data:
    :param test_size:
    :param random_state:
    :param shuffle:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=test_size,
                                                        random_state=random_state, shuffle=shuffle)
    return x_train, x_test, y_train, y_test


def sampling4impbyrow(matrix: ndarray, rows: int, alpha=0, verbose=True):
    """
    从原矩阵中按行采样形成新的矩阵，原矩阵第一行在新矩阵中的占比最大，最后一行占比最小，alpha=0表示每行的数量一样大
    :param matrix: 从原矩阵中按行采样形成新的矩阵
    :param rows: 新矩阵的行数
    :param alpha: 调整原矩阵中不同行的比例超参数，alpha越大原矩阵的第一行在新矩阵中的比例越大,alpha取负数则表示最后一行的比例大，第一行的比例最小
    :param verbose: 是否显示新采样矩阵中不同行的比例柱状图，和不同行的比例
    :return: 返回新采样的矩阵
    """
    num_rows4matrix = matrix.shape[0]
    weights = np.arange(num_rows4matrix+1, 1, -1)
    weights = np.exp(alpha * weights)
    weights /= np.sum(weights)
    sampled_rows = np.random.choice(np.arange(num_rows4matrix), size=rows, p=weights)
    new_matrix = matrix[sampled_rows]
    if verbose:
        print(weights)
        attr_dis_count(new_matrix)
    return new_matrix


def stratified_sample(X, y, test_size=0.2, random_state=None, n_splits=1):
    random_state = random_state if random_state else int(time.time())

    if isinstance(test_size, int):
        x_count = X.shape[0]
        test_size = test_size / x_count if 0 < (test_size / x_count) < 1 else 1

    stratified_splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    # 使用分层采样器来划分训练集和测试集
    for train_index, test_index in stratified_splitter.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test


def get_train_data4Stratified(data, labels, K, random_state=None):

    random_state = random_state if random_state else int(time.time())

    unique_labels, label_counts = np.unique(labels, axis=0, return_counts=True)
    unique_labels_with_single_sample = unique_labels[label_counts == 1]

    # 如果有只有一个样本的类别
    if len(unique_labels_with_single_sample) > 0:
        # 首先将单个样本的类别添加到采样结果中
        sampled_data = []
        sampled_labels = []
        for label in unique_labels_with_single_sample:
            mask = np.all(labels == label, axis=1)
            sampled_data.append(data[mask])
            sampled_labels.append(labels[mask])

        # 使用StratifiedShuffleSplit对其余样本进行分层抽样
        # remaining_labels = labels[~np.isin(labels, unique_labels_with_single_sample).all(axis=1)]
        remaining_index = ~(labels[:, None] == unique_labels_with_single_sample).all(axis=(2)).any(axis=1)
        remaining_labels = labels[remaining_index]
        remaining_data = data[remaining_index]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=K - len(unique_labels_with_single_sample),
                                     random_state=random_state)

        for train_index, test_index in sss.split(remaining_data, remaining_labels):
            sampled_data.append(remaining_data[test_index])
            sampled_labels.append(remaining_labels[test_index])

        return np.vstack(sampled_data), np.vstack(sampled_labels)
    else:
        # 没有只有一个样本的类别，直接进行分层抽样
        sss = StratifiedShuffleSplit(n_splits=1, test_size=K, random_state=random_state)
        for train_index, test_index in sss.split(data, labels):
            sampled_data = data[test_index]
            sampled_labels = labels[test_index]
        return sampled_data, sampled_labels


def get_q_strs(q: ndarray) -> ndarray:
    """
    计算交互式Q矩阵
    :param q:
    :return:
    """
    num_col = q.shape[1]
    poly = PolynomialFeatures(degree=num_col, include_bias=False, interaction_only=True)
    q_strs = poly.fit_transform(q)[:, num_col:]
    return q_strs


def get_imp_irp4ideal(q: ndarray) -> [ndarray, ndarray]:
    """
    获取理想的反应模式和理想的技能掌握模式
    :param q:
    :return: imp2py, irp2py  理想的技能掌握模式与理想的技能反应模式
    """
    q2r = numpy2ri.py2rpy(q)
    data = cdm.ideal_response_pattern(q2r)
    imp = data.rx2('skillspace')
    irp = data.rx2('idealresp')
    imp2py = numpy2ri.rpy2py(imp).squeeze().astype(int)
    irp2py = numpy2ri.rpy2py(irp).squeeze().astype(int).T
    return imp2py, irp2py


def get_all_attr4q(q) -> ndarray:
    """
    根据Q矩阵获取所有可能得技能掌握模式，也可以直接传入技能数量得到所有可能得技能掌握模式
    当模型为二分模型时可以直接传入技能熟练，当模型为多级模型时需要传入q矩阵
    :param q:
    :return:
    """
    if isinstance(q, ndarray):
        k2r = numpy2ri.py2rpy(q.shape[1])
        q2r = numpy2ri.py2rpy(q)
        attrs = gdina.attributepattern(k2r, q2r)
    elif isinstance(q, int):
        q2r = numpy2ri.py2rpy(q)
        attrs = gdina.attributepattern(K=q2r)
    attrs2py = numpy2ri.rpy2py(attrs)
    return attrs2py


def get_q_s(q: ndarray, q_star: ndarray) -> ndarray:
    q_s = q_star.T @ q
    q_s = np.where(q_s > 0, 1, 0)
    q_s[-1] = np.ones(q_s.shape[1])
    return q_s


def get_coordinates(N, centre, r):
    """
    获取半径为r的BUM周围的邻域
    :param N:
    :param centre:
    :param r:
    :return:
    """
    coordinates = list()
    for index in range(len(centre)):
        coordinates.append(centre[index])
        i = centre[index][0]
        j = centre[index][1]
        for x in range(i - r, i + r + 1):
            for y in range(j - r, j + r + 1):
                if 0 <= x < N and 0 <= y < N:
                    coordinates.append(np.array([x, y], dtype=centre[index].dtype))
    return coordinates


def get_net_init_dataset(q: ndarray, attr='mvnorm', isoversamping=True, **kwargs):
    """
    获取模型初始化数据
    :param q:
    :param model:
    :param attr:
    :param kwargs:
    :return: dat, attr
    """
    N = kwargs.get('N') if 'N' in kwargs else 1000
    g = kwargs.get('g') if 'g' in kwargs else (0, 0.3)
    s = kwargs.get('s') if 's' in kwargs else g
    if 'model' in kwargs:
        model = kwargs.get('model')
    else:
        model = 'GDINA' if q.shape[0] > 14 else 'DINA'

    imp, irp = get_imp_irp4ideal(q)
    imp, irp = np.tile(imp, (100, 1)), np.tile(irp, (100, 1))
    dat, attr = simGDINA4R(N, q, g, s, model, attr)
    dat, attr = np.vstack((dat, irp)), np.vstack((attr, imp))
    if isoversamping:
        dat, attr = oversample_data(dat, attr)

    return dat, attr

def get_init_net_data(q):
    N = 2 ** q.shape[1] * 100
    n_dina = 2 ** (q.shape[1] - 1) * 50
    n_gdina = 2 ** (q.shape[1] - 1) * 100
    g_h = (0.05, 0.15)
    s_h = (0.05, 0.15)
    g_l = (0.15, 0.30)
    s_l = (0.15, 0.30)
    attr_dst = 'mvnorm'

    imp, irp = get_imp_irp4ideal(q)
    imp = np.tile(imp, (N//10, 1))
    irp = np.tile(irp, (N//10, 1))

    dat_dina_h, attr_dina_h = simGDINA4R(n_dina, q, g_h, s_h, 'DINA', attr_dst)
    dat_dina_l, attr_dian_l = simGDINA4R(n_dina, q, g_l, s_l, 'DINA', attr_dst)

    dat_gdina_h, attr_gdina_h = simGDINA4R(n_gdina, q, g_h, s_h, 'GDINA', attr_dst)
    dat_gdina_l, attr_gdian_l = simGDINA4R(n_gdina, q, g_l, s_l, 'GDINA', attr_dst)

    dat = np.vstack((dat_dina_h, dat_dina_l, dat_gdina_h, dat_gdina_l, irp))
    attr = np.vstack((attr_dina_h, attr_dian_l, attr_gdina_h, attr_gdian_l, imp))
    dat, attr = oversample_data(dat, attr, 0.5)
    dat, attr = get_train_data4Stratified(dat, attr, N)

    return dat, attr


def create_directory(folder_path):
    try:
        # 检查文件夹是否存在，如果不存在则创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path
        else:
            return folder_path
    except Exception as e:
        print(f"创建文件夹 '{folder_path}' 时发生错误：{str(e)}")


def get_attr4candidate(q, dat, models=[RCDM2py.DINA, RCDM2py.GDINA, RCDM2py.RDINA, RCDM2py.NPC, RCDM2py.GNPC]):
    attrs = []
    for model in models:
        attrs.append((model(q, dat)))

    attrs = np.vstack(attrs)

    unique_rows, counts = np.unique(attrs, axis=0, return_counts=True)
    return unique_rows


def get_dl_train(x: Tensor, y: Tensor, batch_size=32, shuffle=True, num_workers=0):
    dl = DataLoader(TensorDataset(x, y), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    return dl


def np2tensor(device='cuda', *args):
    args = [torch.from_numpy(arg).to(device).float() for arg in args]
    if len(args) < 2:
        return args[0]
    else:
        return args

def numpy2tensor(device, *args):
    res = (torch.from_numpy(arg).to(device).float() for arg in args)
    return res


def getgs4gdina(dat, q, model='DINA'):
    dat2r = numpy2ri.py2rpy(dat)
    q2r = numpy2ri.py2rpy(q)
    mod = gdina.GDINA(dat2r, q2r, model)
    cat = gdina.extract(mod, 'catprob.parm')
    cat2py = [list(item) for item in cat]
    p0 = [min(item) for item in cat2py]
    p1 = [1-max(item) for item in cat2py]
    return p0, p1


def get_Qstar(q, q_apos):
    # 检查输入矩阵是否满足条件
    assert q.shape[1] == q_apos.shape[1], "矩阵A和矩阵B的列数必须相同"
    assert np.all(np.isin(q, [0, 1])), "矩阵A中的元素只能为0或1"
    assert np.all(np.isin(q_apos, [0, 1])), "矩阵B中的元素只能为0或1"

    # 初始化输出矩阵C
    q_star = np.zeros((q.shape[0], q_apos.shape[0]), dtype=int)

    # 遍历矩阵A和矩阵B的每一行
    for i in range(q.shape[0]):
        for j in range(q_apos.shape[0]):
            # 计算矩阵A的第i行和矩阵B的第j行的逻辑与
            and_result = np.logical_and(q[i],q_apos[j])
            # 如果矩阵B的第j行中为1的元素的位置矩阵A的第i行中也为1，则矩阵C的第i行第j列的元素为1
            if np.all(and_result[q_apos[j] == 1]):
                q_star[i, j] = 1

    # 返回矩阵C
    return q_star


def get_q_apos(x, y):
    # 检查输入是否合法
    assert x > 0 and y > 1, "矩阵的维度必须是正整数，且列数必须大于1"
    assert x <= 2**y - y - 1, "矩阵的行数不能超过2的列数次方减去列数再减1，否则无法保证没有重复的行且每行至少有两个1"

    # 初始化一个空的矩阵
    matrix = np.empty((x, y), dtype=int)

    # 用一个集合来存储已经生成的行，用于检查重复
    seen = set()

    # 遍历每一行
    for i in range(x):
        # 生成一个随机的二值向量
        row = np.random.randint(0, 2, y)
        # 将向量转换为二进制数
        num = int("".join(map(str, row)), 2)
        # 检查是否为0或者只有一个1或者已经存在
        while num == 0 or num & (num - 1) == 0 or num in seen:
            # 重新生成一个随机的二值向量
            row = np.random.randint(0, 2, y)
            # 重新转换为二进制数
            num = int("".join(map(str, row)), 2)
        # 将生成的行赋值给矩阵
        matrix[i] = row
        # 将生成的数添加到集合中
        seen.add(num)

    # 返回矩阵
    return matrix


def plot4df(df: pd.DataFrame, kind='line'):
    df.plot(kind=kind)
    plt.show()


def plot4dfs(*args, kind='line', **kwargs: Literal['path']):
    for d in args:
        d.plot(kind=kind)
        if 'path' in kwargs:
            plt.savefig(os.path.join(kwargs.get("path"), f"imgs/{d.columns[0][4:]}.png"), dpi=300)
        plt.show()


def pd_settings():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)



if __name__ == "__main__":
    # from data import EDM
    # path = create_directory(f'../data_local/re/{EDM.data_name}')
    # print(path)

    from data import FS
    dat, attr = get_init_net_data(FS.q)
    attr_dis_count(attr)
    print(attr.shape)

    dat_, attr_  = get_net_init_dataset(FS.q)
    attr_dis_count(attr_)
    print(attr_.shape)


