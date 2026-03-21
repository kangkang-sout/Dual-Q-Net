#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：qnn4cdm -> model.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/9/21
@Desc   ：
==================================================
"""
import abc
import copy
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
from typing_extensions import Literal

import torch.optim
from numpy import ndarray
from rpy2 import robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from varname import nameof

# import src.utils as utils
import utils


class RCDM2py:
    # The classic CDMs implemented based on R language
    gdina = importr('GDINA')
    npcd = importr('NPCD')
    cdt = importr('cdmTools')

    @classmethod
    def DINA(cls, q: ndarray, dat: ndarray, verbose=False) -> ndarray:
        """
        DINA
        :param q:
        :param dat:
        :param verbose:
        :return: attr
        """
        q = numpy2ri.py2rpy(q)
        dat = numpy2ri.py2rpy(dat)
        dina = cls.gdina.GDINA(dat=dat, Q=q, model='DINA', verbose=verbose)
        attr = cls.gdina.personparm(dina)
        attr2py = numpy2ri.rpy2py(attr)
        return attr2py

    @classmethod
    def GDINA(cls, q: ndarray, dat: ndarray, verbose=False):
        q = numpy2ri.py2rpy(q)
        dat = numpy2ri.py2rpy(dat)
        gina = cls.gdina.GDINA(dat=dat, Q=q, model='GDINA', verbose=verbose)
        attr = cls.gdina.personparm(gina)
        attr2py = numpy2ri.rpy2py(attr)
        return attr2py

    @classmethod
    def NPC(cls, q: ndarray, dat: ndarray):
        q = numpy2ri.py2rpy(q)
        dat = numpy2ri.py2rpy(dat)
        npc = cls.npcd.AlphaNP(dat, q)
        attr = npc.rx2('alpha.est')
        attr2py = numpy2ri.rpy2py(attr)
        return attr2py

    @classmethod
    def GNPC(cls, q: ndarray, dat: ndarray, verbose=False):
        q = numpy2ri.py2rpy(q)
        dat = numpy2ri.py2rpy(dat)
        gnpc = cls.cdt.GNPC(dat, q, verbose=verbose)
        attr = gnpc.rx2('alpha.est')
        attr2py = numpy2ri.rpy2py(attr)
        return attr2py

    @classmethod
    def RDINA(cls, q: ndarray, dat: ndarray, verbose=False):
        q = numpy2ri.py2rpy(q)
        dat = numpy2ri.py2rpy(dat)
        em_args = robjects.ListVector({'maxitr': 1000, 'conv_crit': 1e-04, 'init_phi': 0.2, 'verbose': verbose})
        rdina = cls.cdt.RDINA(dat, q, EM_args=em_args)
        attr = rdina.rx2('MLE')
        attr2py = pandas2ri.rpy2py(attr)
        return attr2py.values[:, 0:-1]

    @staticmethod
    def RRUM(cls, q: ndarray, dat: ndarray, verbose=False):
        q = numpy2ri.py2rpy(q)
        dat = numpy2ri.py2rpy(dat)
        gina = cls.gdina.GDINA(dat=dat, Q=q, model='RRUM', verbose=verbose)
        attr = cls.gdina.personparm(gina)
        attr2py = numpy2ri.rpy2py(attr)
        return attr2py

    @staticmethod
    def ACDM(cls, q: ndarray, dat: ndarray, verbose=False):
        q = numpy2ri.py2rpy(q)
        dat = numpy2ri.py2rpy(dat)
        gina = cls.gdina.GDINA(dat=dat, Q=q, model='ACDM', verbose=verbose)
        attr = cls.gdina.personparm(gina)
        attr2py = numpy2ri.rpy2py(attr)
        return attr2py

    @staticmethod
    def LCDM(cls, q: ndarray, dat: ndarray, verbose=False):
        q = numpy2ri.py2rpy(q)
        dat = numpy2ri.py2rpy(dat)
        gina = cls.gdina.GDINA(dat=dat, Q=q, model='logitGDINA', verbose=verbose)
        attr = cls.gdina.personparm(gina)
        attr2py = numpy2ri.rpy2py(attr)
        return attr2py


class BaseNet(nn.Module):
    # Cognitive diagnosis method based on neural network
    @abc.abstractmethod
    def forward(self, x: Tensor):
        pass

    @staticmethod
    def metric_par(y_pred, y_true):
        y_pred = torch.where(y_pred > 0.5,
                             torch.ones_like(y_pred, dtype=torch.float),
                             torch.zeros_like(y_pred, dtype=torch.float))

        par = torch.mean(torch.prod(torch.where((y_pred == y_true) == True,
                                                torch.ones_like(y_pred, dtype=torch.float),
                                                torch.zeros_like(y_pred, dtype=torch.float)), axis=1))  # par
        return par.item()

    @staticmethod
    def metric_acc(y_pred, y_true):
        y_pred = torch.where(y_pred > 0.5,
                             torch.ones_like(y_pred, dtype=torch.float),
                             torch.zeros_like(y_pred, dtype=torch.float))
        acc = torch.mean(1 - torch.abs(y_true - y_pred))  # aar
        return acc.item()

    @staticmethod
    def loss_func(y_true: Tensor, y_pred: Tensor):
        loss_f = nn.MSELoss()
        return loss_f(y_true, y_pred)

    # @property
    def get_optimizer(self, lr=0.001):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def predictive(self, features, is2binary=False):
        y_hat = self.forward(features)
        if is2binary:
            y_hat = torch.where(y_hat > 0.5,
                                 torch.ones_like(y_hat, dtype=torch.float),
                                 torch.zeros_like(y_hat, dtype=torch.float))
        return y_hat

    @staticmethod
    def plot_log(dfhistory: pd.DataFrame, x_point=50):
        dfhistory.plot(kind='line')
        plt.show()

    @staticmethod
    def _get_data_loader(x: Tensor, y: Tensor, batch_size=32, shuffle=True, num_workers=0):
        dl = DataLoader(TensorDataset(x, y), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
        return dl

    def _train_net_step(self, features, labels):
        # Zero gradient
        self.optimizer.zero_grad()

        # forward
        pre = self.forward(features)
        loss = self.loss_func(pre, labels)

        # backward
        loss.backward()

        # updated parameter
        self.optimizer.step()
        return loss.item()

    def train_net(self, x_train, y_train, lr=0.001, epochs=500, batch_size=64, shuffle=True, num_workers=0, verbose=True,**kwargs: Literal['x_test, y_test', 'pt_path', 'log_path']):
        self.optimizer = self.get_optimizer(lr=lr)
        dl = self._get_data_loader(x_train, y_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        if 'x_test' in kwargs and 'y_test' in kwargs:
            dfhistory = pd.DataFrame(columns=["Epochs", f"{self._name}_Tran_loss", f'{self._name}_Train_AAR', f'{self._name}_Train_PAR', f"{self._name}_Val_loss", f"{self._name}_Val_AAR", f"{self._name}_Val_PAR"])
        else:
            dfhistory = pd.DataFrame(columns=["Epochs", f"{self._name}_Tran_loss", f'{self._name}_Train_AAR', f'{self._name}_Train_PAR'])

        for epoch in range(0, epochs):
            self.train()
            tran_loss, step = 0, 0
            for features, labels in dl:
                tran_loss += self._train_net_step(features, labels)
                step += 1

            # 记录 loss，aar, par
            tran_loss = tran_loss / step
            y_hat_train = self.predictive(x_train)
            train_AAR = self.metric_acc(y_hat_train, y_train)
            train_PAR = self.metric_par(y_hat_train, y_train)

            if 'x_test' in kwargs and 'y_test' in kwargs:
                self.eval()
                with torch.no_grad():
                    y_hat_test = self.predictive(kwargs.get('x_test'))
                    val_loss = self.loss_func(y_hat_test, kwargs.get('y_test')).item()
                    val_AAR = self.metric_acc(y_hat_test, kwargs.get('y_test'))
                    val_PAR = self.metric_par(y_hat_test, kwargs.get('y_test'))
                info = (epoch, tran_loss, train_AAR, train_PAR, val_loss, val_AAR, val_PAR)
            else:
                info = (epoch, tran_loss, train_AAR, train_PAR)
            dfhistory.loc[epoch] = tuple(map(lambda x: round(x, 3), info))
            if verbose and (epoch % (epochs/20)==0):
                print(f'{self._name}{"*"*20}{dfhistory.loc[epoch]}')

        if verbose:
            self.plot_log(dfhistory.iloc[:, 1:])
        if 'pt_path' in kwargs:
            torch.save(self.state_dict(), os.path.join(kwargs.get('pt_path'), f'{self._name}_parms.pt'))
        if 'log_path' in kwargs:
            dfhistory.to_csv(os.path.join(kwargs.get('log_path'), f'dfhistory.csv'))
        return dfhistory


class MLP(BaseNet):
    def __init__(self, q: Tensor):
        super().__init__()
        self._name = 'MLP'
        self.mc = nn.Linear(q.shape[0], q.shape[1])

    def forward(self, x: Tensor):
        y = torch.sigmoid(self.mc(x))
        return y


class ANN(BaseNet):
    def __init__(self, q: Tensor):
        super().__init__()
        self._name = 'ANN'
        self.mc = nn.Linear(q.shape[0], q.shape[1])
        self.mh = nn.Linear(q.shape[1], q.shape[1])

    def forward(self, x):  # sourcery skip: inline-immediately-returned-variable
        x = x if type(x) == torch.Tensor else self._to_Tensor(x)
        h = torch.sigmoid(self.mc(x))
        y = torch.sigmoid(self.mh(h))
        return y


class QNet(BaseNet):
    def __init__(self, q: Tensor, q_strs: Tensor):
        super().__init__()
        self._name = 'Dual Q-Net'
        self._cons_mc = torch.t(q)
        self._cons_lc = torch.t(q_strs)
        self.mc = nn.Linear(q.shape[0], q.shape[1])
        self.lc = nn.Linear(q.shape[0], q_strs.shape[1])
        self.cc = nn.Linear(q.shape[1]+q_strs.shape[1], q.shape[1])

    def forward(self, x):  # sourcery skip: inline-immediately-returned-variable
        m = torch.relu(self.mc(x))
        l = torch.tanh(self.lc(x))
        y = torch.sigmoid(self.cc(torch.cat((m, l), dim=1)))
        return y

    def _train_net_step(self, features, labels):
        # forward
        pre = self.forward(features)
        loss = self.loss_func(pre, labels)

        # Zero gradient
        self.optimizer.zero_grad()

        # backward
        loss.backward()

        # updated paramenter
        self.optimizer.step()
        # self.mc.weight.data = self.mc.weight.data * self._cons_mc
        self.lc.weight.data = self.lc.weight.data * self._cons_lc

        return loss.item()


class QNet1(BaseNet):
    def __init__(self, q: Tensor, q_strs: Tensor):
        super().__init__()
        self._name = 'Dual Q-Net'
        self._cons_mc = torch.t(q)
        self._cons_lc = torch.t(q_strs)
        
        # 初始化线性层
        self.mc = nn.Linear(q.shape[0], q.shape[1])
        self.lc = nn.Linear(q.shape[0], q_strs.shape[1])
        self.cc = nn.Linear(q.shape[1]+q_strs.shape[1], q.shape[1])
        
        # 初始化权重
        nn.init.xavier_uniform_(self.mc.weight)
        nn.init.xavier_uniform_(self.lc.weight)
        nn.init.xavier_uniform_(self.cc.weight)
        
        # 批量归一化层
        self.bn1 = nn.BatchNorm1d(q.shape[1])
        self.bn2 = nn.BatchNorm1d(q_strs.shape[1])
        
        # Dropout层
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 主路径
        m = self.mc(x)
        m = self.bn1(m)
        m = torch.relu(m)
        m = self.dropout(m)
        
        # 侧路径
        l = self.lc(x)
        l = self.bn2(l)
        l = torch.tanh(l)
        l = self.dropout(l)
        
        # 组合输出
        y = torch.sigmoid(self.cc(torch.cat((m, l), dim=1)))
        return y

    def _train_net_step(self, features, labels):
        # forward
        pre = self.forward(features)
        loss = self.loss_func(pre, labels)

        # Zero gradient
        self.optimizer.zero_grad()

        # backward
        loss.backward()

        # 更新参数
        self.optimizer.step()
        
        # 应用Q矩阵约束
        self.mc.weight.data = self.mc.weight.data * self._cons_mc
        self.lc.weight.data = self.lc.weight.data * self._cons_lc

        return loss.item()





if __name__ == '__main__':
    from data import HSD1,HSD2, LSD1, LSD2, EDM, FS
    from configs import device
    from utils import numpy2tensor, get_Qstar, get_q_apos, plot4dfs
    x_train, y_train = LSD1.get_init_data4train()
    x_test, _, y_test, _ = LSD1.get_sub_data4train_test(100)
    q = LSD1.q
    q_apos = get_q_apos(4, 3)
    q_star = get_Qstar(q, q_apos)
    x_train, y_train, x_test, y_test, q, q_star = numpy2tensor(device, x_train, y_train, x_test, y_test, q, q_star)

    mlp = MLP(q).to(device)
    ANN = ANN(q).to(device)
    QNet = QNet(q, q_star).to(device)

    dfhistory_mlp = mlp.train_net(x_train, y_train, x_test=x_test, y_test=y_test, lr=0.01, epochs=100, batch_size=64)
    dfhistory_ann = ANN.train_net(x_train, y_train, x_test=x_test, y_test=y_test, lr=0.01, epochs=100, batch_size=64)
    dfhistory_qnet = QNet.train_net(x_train, y_train, x_test=x_test, y_test=y_test, lr=0.01, epochs=100, batch_size=64)

    columns = ["Epochs", "Tran_loss", 'Train_AAR', 'Train_PAR', "Val_loss", "Val_AAR", "Val_PAR"]
    df_Tran_loss = pd.concat([dfhistory_mlp.iloc[:, 1:2], dfhistory_ann.iloc[:, 1:2], dfhistory_qnet.iloc[:, 1:2]], axis=1)
    df_Train_AAR = pd.concat([dfhistory_mlp.iloc[:, 2:3], dfhistory_ann.iloc[:, 2:3], dfhistory_qnet.iloc[:, 2:3]], axis=1)
    df_Train_PAR = pd.concat([dfhistory_mlp.iloc[:, 3:4], dfhistory_ann.iloc[:, 3:4], dfhistory_qnet.iloc[:, 3:4]], axis=1)
    df_Val_loss = pd.concat([dfhistory_mlp.iloc[:, 4:5], dfhistory_ann.iloc[:, 4:5], dfhistory_qnet.iloc[:, 4:5]], axis=1)
    df_Val_AAR = pd.concat([dfhistory_mlp.iloc[:, 5:6], dfhistory_ann.iloc[:, 5:6], dfhistory_qnet.iloc[:, 5:6]], axis=1)
    df_Val_PAR = pd.concat([dfhistory_mlp.iloc[:, 6:], dfhistory_ann.iloc[:, 6:], dfhistory_qnet.iloc[:, 6:]], axis=1)

    plot4dfs(df_Tran_loss, df_Train_AAR, df_Train_PAR, df_Val_loss, df_Val_AAR, df_Val_PAR)

    print('end')
















