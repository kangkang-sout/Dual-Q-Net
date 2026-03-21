#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：DualQNet -> main.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/12/7
@Desc   ：Examples of using QNet
==================================================
"""
from data import HSD1
from train import train_data, train_net
from models import QNet
from utils import class_rate4gdina


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, q, q_star = train_data(HSD1, q_apos=4, verbose=True)
    qnet, log = train_net(QNet, x_train, y_train, q, q_star, x_test=x_test, y_test=y_test, verbose=True)
    y_hat = qnet.predictive(x_test, is2binary=True)
    acc = class_rate4gdina(y_test, y_hat, -2, 3)

    # AAR,PAR(k),PAR(k-1)
    print(f'{"*"*50}\nAAR:{acc[0]}\nPAR({q.shape[1]}):{acc[1][1]}\nPAR({q.shape[1]-1}):{acc[1][0]}')
