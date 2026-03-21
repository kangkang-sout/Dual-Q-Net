#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：DualQNet -> frac_ex.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/12/23
@Desc   ：
==================================================
"""
from src.train import ex
from src.configs import frac_ex_parms

if __name__ == '__main__':
    df_aar, df_aar_mean, df_par, df_par_mean = ex(**frac_ex_parms)