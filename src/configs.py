#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：DualQNet -> configs.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/12/8
@Desc   ：
==================================================
"""
import numpy as np
import torch

from models import MLP, ANN, QNet
from data import HSD1,HSD2, LSD1, LSD2, EDM, FS, SimDataSet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nets = [MLP, ANN, QNet]
samples4sim = [100, 200, 300, 500]
samples4real = [50, 100, 200, 300, 500]

# pre
hsd1_pre_paarms = {'net_list': nets,
                'data': HSD1,
                'q_apos': 4,
                'is_init': True,
                'is_ideal': False,
                'samp_size_train': 500,
                'samp_size_test': 100,
                'lr': 0.008,
                'epochs': 100,
                'batch_size': 64,
                'shuffle': True,
                'num_workers': 0,
                'verbose': False}

lsd1_pre_paarms = {'net_list': nets,
                'data': LSD1,
                'q_apos': 4,
                'is_init': True,
                'is_ideal': False,
                'samp_size_train': 500,
                'samp_size_test': 100,
                'lr': 0.005,
                'epochs': 200,
                'batch_size': 128,
                'shuffle': True,
                'num_workers': 0,
                'verbose': False}

hsd2_pre_paarms = {'net_list': nets,
                'data': HSD2,
                'q_apos': 10,
                'is_init': True,
                'is_ideal': False,
                'samp_size_train': 500,
                'samp_size_test': 100,
                'lr': 0.01,
                'epochs': 100,
                'batch_size': 128,
                'shuffle': True,
                'num_workers': 0,
                'verbose': False}

lsd2_pre_paarms = {'net_list': nets,
                'data': LSD2,
                'q_apos': 10,
                'is_init': True,
                'is_ideal': False,
                'samp_size_train': 500,
                'samp_size_test': 100,
                'lr': 0.005,
                'epochs': 200,
                'batch_size': 64,
                'shuffle': True,
                'num_workers': 0,
                'verbose': False}

hsd2_pre_paarms4 = {'net_list': nets,
                'data': HSD2,
                'q_apos': 4,
                'is_init': True,
                'is_ideal': False,
                'samp_size_train': 500,
                'samp_size_test': 100,
                'lr': 0.005,
                'epochs': 100,
                'batch_size': 64,
                'shuffle': True,
                'num_workers': 0,
                'verbose': False}

lsd2_pre_paarms4 = {'net_list': nets,
                'data': LSD2,
                'q_apos': 4,
                'is_init': True,
                'is_ideal': False,
                'samp_size_train': 500,
                'samp_size_test': 100,
                'lr': 0.004,
                'epochs': 200,
                'batch_size': 64,
                'shuffle': True,
                'num_workers': 0,
                'verbose': False}

# ideal
hsd1_ideal_paarms = {'net_list': nets,
                'data': HSD1,
                'q_apos': 4,
                'is_init': False,
                'is_ideal': True,
                'samp_size_train': 500,
                'samp_size_test': 100,
                'lr': 0.01,
                'epochs': 500,
                'batch_size': 128,
                'shuffle': True,
                'num_workers': 0,
                'verbose': False}

lsd1_ideal_paarms = {'net_list': nets,
                'data': LSD1,
                'q_apos': 4,
                'is_init': False,
                'is_ideal': True,
                'samp_size_train': 500,
                'samp_size_test': 100,
                'lr': 0.001,
                'epochs': 500,
                'batch_size': 128,
                'shuffle': True,
                'num_workers': 0,
                'verbose': False}

hsd2_ideal_paarms = {'net_list': nets,
                'data': HSD2,
                'q_apos': 10,
                'is_init': False,
                'is_ideal': True,
                'samp_size_train': 500,
                'samp_size_test': 100,
                'lr': 0.01,
                'epochs': 100,
                'batch_size': 64,
                'shuffle': True,
                'num_workers': 0,
                'verbose': False}

lsd2_ideal_paarms = {'net_list': nets,
                'data': LSD2,
                'q_apos': 10,
                'is_init': False,
                'is_ideal': True,
                'samp_size_train': 500,
                'samp_size_test': 100,
                'lr': 0.01,
                'epochs': 100,
                'batch_size': 64,
                'shuffle': True,
                'num_workers': 0,
                'verbose': False}

# sim_ex
hsd1_sim_ex_parms = {
                    'data': HSD1,
                    'q_apos': 4,
                    'samples': samples4sim,
                    'rsd': 20,
                    'save_path': "../../data/output/hsd1/ex/",
                    'mlp_parms_path': "../../data/output/hsd1/pre_train/pth/MLP_pre.pt",
                    'ann_parms_path': "../../data/output/hsd1/pre_train/pth/ANN_pre.pt",
                    'qnet_parms_path': "../../data/output/hsd1/pre_train/pth/Dual Q-Net_pre.pt",
                    'net_parms': hsd1_pre_paarms,
                    'par_digit': -2,
                    'verbose': True}

lsd1_sim_ex_parms = {
                    'data': LSD1,
                    'q_apos': 4,
                    'samples': samples4sim,
                    'rsd': 20,
                    'save_path': "../../data/output/lsd1/ex/",
                    'mlp_parms_path': "../../data/output/lsd1/pre_train/pth/MLP_pre.pt",
                    'ann_parms_path': "../../data/output/lsd1/pre_train/pth/ANN_pre.pt",
                    'qnet_parms_path': "../../data/output/lsd1/pre_train/pth/Dual Q-Net_pre.pt",
                    'net_parms': lsd1_pre_paarms,
                    'par_digit': -2,
                    'verbose': True}

hsd2_sim_ex_parms = {
                    'data': HSD2,
                    'q_apos': 10,
                    'samples': samples4sim,
                    'rsd': 20,
                    'save_path': "../../data/output/hsd2/ex/",
                    'mlp_parms_path': "../../data/output/hsd2/pre_train/pth/MLP_pre.pt",
                    'ann_parms_path': "../../data/output/hsd2/pre_train/pth/ANN_pre.pt",
                    'qnet_parms_path': "../../data/output/hsd2/pre_train/pth/Dual Q-Net_pre.pt",
                    'net_parms': hsd2_pre_paarms,
                    'par_digit': -2,
                    'verbose': True}

lsd2_sim_ex_parms = {
                    'data': LSD2,
                    'q_apos': 10,
                    'samples': samples4sim,
                    'rsd': 20,
                    'save_path': "../../data/output/lsd2/ex/",
                    'mlp_parms_path': "../../data/output/lsd2/pre_train/pth/MLP_pre.pt",
                    'ann_parms_path': "../../data/output/lsd2/pre_train/pth/ANN_pre.pt",
                    'qnet_parms_path': "../../data/output/lsd2/pre_train/pth/Dual Q-Net_pre.pt",
                    'net_parms': lsd2_pre_paarms,
                    'par_digit': -2,
                    'verbose': True}

# real
q_apos4edm=np.array([[1, 0, 1, 0],
                     [0, 1, 1, 0],
                     [0, 0, 1, 1],
                     [0, 1, 0, 1],
                     [0, 1, 1, 1]])

q_apos4fs = np.array([
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 1, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 1, 0, 0, 0],
                    [0, 1, 0, 1, 1, 0, 1, 0],
                    [0, 1, 1, 0, 1, 0, 0, 0],
                    [0, 1, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0, 0, 1, 0]])


edm_parms = {'net_list': nets,
                'data': EDM,
                'q_apos': q_apos4edm,  # q_apos4edm
                'is_init': True,
                'is_ideal': False,
                'samp_size_train': 500,
                'samp_size_test': 100,
                'lr': 0.004,
                'epochs': 100,
                'batch_size': 64,
                'shuffle': True,
                'num_workers': 0,
                'verbose': False}


fs_parms = {'net_list': nets,
                'data': FS,
                'q_apos': q_apos4fs,  # q_apos4fs
                'is_init': True,
                'is_ideal': False,
                'samp_size_train': 500,
                'samp_size_test': 100,
                'lr': 0.001,
                'epochs': 100,
                'batch_size': 64,
                'shuffle': True,
                'num_workers': 0,
                'verbose': False}


eptt_ex_parms = {
                    'data': EDM,
                    'q_apos': q_apos4edm,  # q_apos4edm  5
                    'samples': samples4sim,
                    'rsd': 10,
                    'save_path': "../../../data/output/edm/ex/",
                    'mlp_parms_path': "../../../data/output/edm/pre_train/pth/MLP_pre.pt",
                    'ann_parms_path': "../../../data/output/edm/pre_train/pth/ANN_pre.pt",
                    'qnet_parms_path': "../../../data/output/edm/pre_train/pth/Dual Q-Net_pre.pt",
                    'net_parms': edm_parms,
                    'par_digit': -2,
                    'verbose': True}


frac_ex_parms = {
                    'data': FS,
                    'q_apos': q_apos4fs,  # q_apos4fs  17
                    'samples': samples4sim,
                    'rsd': 10,
                    'save_path': "../../../data/output/fs/ex/",
                    'mlp_parms_path': "../../../data/output/fs/pre_train/pth/MLP_pre.pt",
                    'ann_parms_path': "../../../data/output/fs/pre_train/pth/ANN_pre.pt",
                    'qnet_parms_path': "../../../data/output/fs/pre_train/pth/Dual Q-Net_pre.pt",
                    'net_parms': fs_parms,
                    'par_digit': -2,
                    'verbose': True}


if __name__ == '__main__':
    print(hsd1_pre_paarms)