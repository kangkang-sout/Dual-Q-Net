#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：DualQNet -> train.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/12/7
@Desc   ：
==================================================
"""
from configs import hsd1_pre_paarms
from train import mult_train_net, save_mods_logs


if __name__ == '__main__':
    mods, logs = mult_train_net(**hsd1_pre_paarms)

    path = '../data/output/hsd1/pre_train'
    save_mods_logs(mods, logs, path)