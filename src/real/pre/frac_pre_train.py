#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：DualQNet -> frac_pre_train.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/12/22
@Desc   ：
==================================================
"""
from src.configs import fs_parms
from src.train import mult_train_net, save_mods_logs
from src.utils import plot4dfs

if __name__ == '__main__':
    mods, logs = mult_train_net(**fs_parms)

    path = '../../../data/output/fs/pre_train'

    plot4dfs(*logs, path=path)
    save_mods_logs(mods, logs, path)