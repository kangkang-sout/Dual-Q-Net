#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：DualQNet -> eptt_pre_train.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/12/22
@Desc   ：
==================================================
"""
from src.configs import edm_parms
from src.train import mult_train_net, save_mods_logs
from src.utils import plot4dfs

if __name__ == '__main__':
    mods, logs = mult_train_net(**edm_parms)

    path = '../../../data/output/edm/pre_train'

    plot4dfs(*logs, path=path)
    save_mods_logs(mods, logs, path)