#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：qnn4cdm -> data.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/9/21
@Desc   ：
==================================================
"""
import abc
import math
import statistics
import time
from varname import nameof

import numpy as np
from numpy import ndarray
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from sklearn.model_selection import train_test_split
import utils as utils

from models import RCDM2py

cdm = importr('CDM')
gdina = importr('GDINA')
edm = importr('edmdata')


q1 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 1, 1]])

q2 = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 1, 0],
    [1, 0, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 1, 1, 0, 1],
    [1, 1, 1, 0, 1],
    [0, 0, 0, 1, 1],
    [1, 0, 0, 1, 1],
    [0, 1, 0, 1, 1],
    [1, 1, 0, 1, 1],
    [0, 0, 1, 1, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]])

q3 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]])

q4 = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1],
    [1, 1, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 1, 0, 1, 1],
    [0, 0, 1, 1, 1],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 0, 1],
    [1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1], ])

q5 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]])


class sim30DINA:
    q, dat = utils.get_data4gdina('sim30DINA')


class sim10GDINA:
    q, dat = utils.get_data4gdina('sim10GDINA')


class EdmData:
    q = numpy2ri.rpy2py(robjects.r('qmatrix_probability_part_one'))
    dat = numpy2ri.rpy2py(robjects.r('items_probability_part_one'))


class FSData:
    q, dat = utils.get_data4gdina('frac20', False)
    q = q.T
    dat = dat.T


class SFSData:
    q, dat = utils.get_data4cdm('data.fraction1')


class BaseDate(abc.ABC):
    def __init__(self, data_name: str, q: ndarray, model: str):
        self.data_name = data_name
        self.q = q
        self.model = model

    @abc.abstractmethod
    def get_init_data4train(self, isverbose=False):
        pass
    
    @abc.abstractmethod
    def _get_sub_data(self, N, random_state=None):
        pass

    def get_sub_data4train_test(self, N, isoversampling=0.5, isverbose=False):
        x_test, y_test = self._get_sub_data(N)
        unique_rows, counts = np.unique(y_test, axis=0, return_counts=True)

        is_addii = math.ceil(N /10/ unique_rows.shape[0])

        index = []
        for i in range((len(self.imp))):
            if any(np.array_equal(self.imp[i], item) for item in unique_rows):
                index.append(i)

        x_train = np.vstack((x_test, np.tile(self.irp[index], (is_addii, 1))))
        y_train = np.vstack((y_test, np.tile(self.imp[index], (is_addii, 1))))

        if 0 < isoversampling < 1:
            x_train, y_train = utils.oversample_data(x_train, y_train, isoversampling)

        if isverbose:
            utils.attr_dis_count(y_test)
            utils.attr_dis_count(y_train)
            from varname import nameof
            print(f'{nameof(y_test)}.shape:{y_test.shape}')
            print(f'{nameof(y_train)}.shape:{y_train.shape}')

        return x_train, x_test, y_train, y_test


class RealData(BaseDate):
    def __init__(self, data_name, q, dat, model='DINA'):
        super(RealData, self).__init__(data_name, q, model)
        self.dat = dat
        self.q = q
        self.q_star = utils.get_q_strs(q)
        self.q_s = utils.get_q_s(q, self.q_star)
        self.imp, self.irp = utils.get_imp_irp4ideal(q)
        self.attr = RCDM2py.DINA(q, dat) if model == 'DINA' else RCDM2py.GDINA(q, dat)
        self.attr4candidate = utils.get_attr4candidate(self.q, self.dat)
        self.guess, self.slip = self._get_gs4data()

    def _get_gs4data(self):
        self.p0, self.p1 = utils.getgs4gdina(self.dat, self.q, self.model)
        p0_mean = round(statistics.mean(self.p0), 2)
        p1_mean = round(statistics.mean(self.p1), 2)
        return p0_mean, p1_mean

    def get_init_data4train(self, isverbose=False):
        sample_size =  2 ** self.q.shape[1] * 100
        g_h = [self.guess] * self.q.shape[0]
        s_h = [self.slip] * self.q.shape[0]
        g_l = self.p0
        s_l = self.p1

        dat_h, attr_h = utils.simGDINA4R(sample_size, self.q, g_h, s_h, self.model, attr='mvnorm')
        dat_l, attr_l = utils.simGDINA4R(sample_size, self.q, g_l, s_l, self.model, attr='mvnorm')

        index_h = []
        index_l = []

        for i in range(sample_size):
            if (self.attr4candidate == attr_h[i]).all(axis=1).any():
                index_h.append(i)

            if (self.attr4candidate == attr_l[i]).all(axis=1).any():
                index_l.append(i)

        dat_h, attr_h = dat_h[index_h], attr_h[index_h]
        dat_l, attr_l = dat_l[index_l], attr_l[index_l]

        dat = np.vstack((dat_h, dat_l))
        attr = np.vstack((attr_h, attr_l))

        is_addii = math.ceil(dat.shape[0] / 10 / self.attr4candidate.shape[0])

        index_i = []
        for i in range(self.imp.shape[0]):
            if any(np.array_equal(self.imp[i], item) for item in self.attr4candidate):
                index_i.append(i)

        irp, imp = self.irp[index_i], self.imp[index_i]

        dat = np.vstack((dat, np.tile(irp, (is_addii, 1))))
        attr = np.vstack((attr, np.tile(imp, (is_addii, 1))))

        dat, attr = utils.oversample_data(dat, attr, 0.3)
        x_train, y_train = utils.get_train_data4Stratified(dat, attr, self.attr4candidate.shape[0]*100)

        if isverbose:
            utils.attr_dis_count(y_train)
            from varname import nameof
            print(f'{self.data_name}--ini--{nameof(y_train)}.shape:{y_train.shape}')

        return x_train.astype(float), y_train.astype(float)

    def _get_sub_data(self, N, random_state=None):
        random_state = random_state if random_state else int(time.time())
        if N < 301:
            try:
                x_train, y_train = utils.get_train_data4Stratified(self.dat, self.attr, N, random_state=random_state)
            except Exception as e:
                dat, attr = utils.oversample_data(self.dat, self.attr, 0.2)
                x_train, y_train = utils.get_train_data4Stratified(dat, attr, N, random_state=random_state)
            return x_train, y_train
        else:
            return self.dat, self.attr

    def get_sub_data4train_test(self, N, is_addii=None, isoversampling=0.3, isverbose=False):
        x_test, y_test = self._get_sub_data(N)
        unique_rows, counts = np.unique(y_test, axis=0, return_counts=True)
        is_addii = math.ceil(N / 10 / unique_rows.shape[0])
        x_sim, _ = utils.simGDINA4R(self.imp.shape[0], self.q, 0.15, 0.15, self.model, self.imp)

        index = []
        for i in range((len(self.imp))):
            if any(np.array_equal(self.imp[i], item) for item in unique_rows):
                index.append(i)

        x_train = np.vstack((x_test, np.tile(self.irp[index], (is_addii, 1))))
        y_train = np.vstack((y_test, np.tile(self.imp[index], (is_addii, 1))))

        x_train = np.vstack((x_train, np.tile(x_sim[index], (is_addii, 1))))
        y_train = np.vstack((y_train, np.tile(self.imp[index], (is_addii, 1))))

        if 0 < isoversampling < 1:
            x_train, y_train = utils.oversample_data(x_train, y_train, isoversampling)

        # x_train, y_train = utils.get_train_data4Stratified(x_train, y_train, N*10)

        if isverbose:
            utils.attr_dis_count(y_test)
            utils.attr_dis_count(y_train)
            from varname import nameof
            print(f'{nameof(y_test)}.shape:{y_test.shape}')
            print(f'{nameof(y_train)}.shape:{y_train.shape}')

        return x_train.astype(float), x_test.astype(float), y_train.astype(float), y_test.astype(float)


class SimDataSet(BaseDate):
    def __init__(self, data_name, q=q1, sample_size=None, g=(0.0, 0.15), s=(0.0, 0.15), model='DINA', attr='mvnorm'):
        super(SimDataSet, self).__init__(data_name, q, model)
        self.g = g
        self.s = s
        self.attr = attr
        self.q = q
        self.q_star = utils.get_q_strs(q)
        self.q_s = utils.get_q_s(q, self.q_star)
        self.imp, self.irp = utils.get_imp_irp4ideal(q)
        self.sample_size = sample_size if sample_size else 2**q.shape[1]*100

        self.dat, self.attr = utils.simGDINA4R(self.sample_size, q, g, s, model, attr)

    def get_init_data4train(self, isverbose=False):
        is_addii = math.ceil(self.sample_size/10/(2**self.q.shape[1]))

        x_sim, _ = utils.simGDINA4R(self.imp.shape[0], self.q, self.g, self.s, self.model, self.imp)

        dat = np.vstack((self.dat, np.tile(self.irp, (is_addii, 1))))
        attr = np.vstack((self.attr, np.tile(self.imp, (is_addii, 1))))

        dat = np.vstack((dat, np.tile(x_sim, (is_addii, 1))))
        attr = np.vstack((attr, np.tile(self.imp, (is_addii, 1))))

        dat, attr = utils.oversample_data(dat, attr, 0.3)
        x_train, y_train = utils.get_train_data4Stratified(dat, attr, self.sample_size)

        if isverbose:
            utils.attr_dis_count(self.attr)
            utils.attr_dis_count(attr)
            utils.attr_dis_count(y_train)
            print(f'{nameof(self.attr)}.shape:{self.attr.shape}')
            print(f'{nameof(attr)}.shape:{attr.shape}')
            print(f'{nameof(y_train)}.shape:{y_train.shape}')
        return x_train.astype(float), y_train.astype(float)

    def _get_sub_data(self, N, random_state=None):
        random_state = random_state if random_state else int(time.time())
        dat, attr = self.dat, self.attr
        # if isresim_size > 0:
        #     dat, attr = utils.simGDINA4R(isresim_size, self.q, self.g, self.s, self.model, self.attr)

        x_train, y_train = utils.get_train_data4Stratified(dat, attr, N, random_state=random_state)

        return x_train.astype(float), y_train.astype(float)

    def get_sub_data4train_test(self, N, isoversampling=0.3, isverbose=False):
        x_test, y_test = self._get_sub_data(N)
        unique_rows, counts = np.unique(y_test, axis=0, return_counts=True)
        is_addii = math.ceil(N / 10 / unique_rows.shape[0])
        x_sim, _ = utils.simGDINA4R(self.imp.shape[0], self.q, self.g, self.s, self.model, self.imp)

        index = []
        for i in range((len(self.imp))):
            if any(np.array_equal(self.imp[i], item) for item in unique_rows):
                index.append(i)

        x_train = np.vstack((x_test, np.tile(self.irp[index], (is_addii, 1))))
        y_train = np.vstack((y_test, np.tile(self.imp[index], (is_addii, 1))))

        x_train = np.vstack((x_train, np.tile(x_sim[index], (is_addii, 1))))
        y_train = np.vstack((y_train, np.tile(self.imp[index], (is_addii, 1))))

        if 0 < isoversampling < 1:
            x_train, y_train = utils.oversample_data(x_train, y_train, isoversampling)

        if isverbose:
            utils.attr_dis_count(y_test)
            utils.attr_dis_count(y_train)
            from varname import nameof
            print(f'{nameof(y_test)}.shape:{y_test.shape}')
            print(f'{nameof(y_train)}.shape:{y_train.shape}')

        return x_train.astype(float), x_test.astype(float), y_train.astype(float), y_test.astype(float)


# sim data
HSD1 = SimDataSet('HSD1')  # 10x3
LSD1 = SimDataSet('LSD1', g=(0.15, 0.30), s=(0.15, 0.30))  # 10x3

HSD2 = SimDataSet('HSD2', q=q2, g=(0.0, 0.15), s=(0.0, 0.15), model='GDINA')  # 31x5
LSD2 = SimDataSet('LSD2', q=q2, g=(0.15, 0.30), s=(0.15, 0.30), model='GDINA')  # 31x5

# real data

FS = RealData('FRAC', FSData.q, FSData.dat)  # 20x8x536
SFS = RealData('SFS', SFSData.q, SFSData.dat)  # 15x5x536
EDM = RealData('EPTT', EdmData.q, EdmData.dat, 'GDINA')  # 12x4x504


simData = [HSD1, LSD1, HSD2, LSD2]
realData = [EDM, SFS, FS]
simData_H = [HSD1, HSD2]
simData_L = [LSD1, LSD2]
simData4one = [HSD1, LSD1]
simData4two = [HSD2, LSD2]


if __name__ == '__main__':
    # q, dat = SFSData.q, SFSData.dat
    # HSD1_50 = HSD1.get_dataset4train(500, verbose=True)
    # LSD1_50 = LSD1.get_dataset4train(500, verbose=True)
    # HSD2_50 = HSD2.get_dataset4train(500, verbose=True)
    # LSD2_50 = LSD2.get_dataset4train(500, verbose=True)
    # N = [50, 100, 300, 500]
    N = [500]
    # data = [HSD1, LSD1, HSD2, LSD2, EDM, SFS, FS]
    data = [HSD1]

    for d in data:
        print(f'{d.data_name}-------------------')
        d.get_init_data4train(isverbose=True)
        for n in N:
            print(f'{d.data_name}-----{n}--------------')
            d.get_sub_data4train_test(n, isverbose=True)
            pass

