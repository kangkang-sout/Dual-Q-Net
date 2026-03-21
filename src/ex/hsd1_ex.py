#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：DualQNet -> hsd1_ex.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/12/15
@Desc   ：
==================================================
"""
import os.path

import pandas as pd
import torch

from src.configs import device, samples4sim, hsd1_pre_paarms, hsd1_sim_ex_parms
from src.models import MLP, ANN, QNet
from src.train import loadNet, get_subdata, ex
from src.data import HSD1, LSD1, HSD2, LSD2
from src.rmodels import RCDM2py
from src.utils import class_rate4gdina, pd_settings
from typing_extensions import Literal

mlp_parms_path = "../../data/output/hsd1/pre_train/pth/MLP_pre.pt"
ann_parms_path = "../../data/output/hsd1/pre_train/pth/ANN_pre.pt"
qnet_parms_path = "../../data/output/hsd1/pre_train/pth/Dual Q-Net_pre.pt"

if __name__ == '__main__':
    # x_train, x_test, y_train, y_test, q, q_star = get_subdata(HSD1, 4, 100)
    # mlp = loadNet(MLP, mlp_parms_path, q, q_star)
    # y_hat = mlp.predictive(x_test, is2binary=True)
    # aar = mlp.metric_acc(y_hat, y_test)
    # par = mlp.metric_par(y_hat, y_test)
    # print(aar)
    # y_hat_dina = RCDM2py.DINA(HSD1.q, x_test.cpu().numpy().astype(float))
    #
    # print('end')

    # def net_estimate(net, q, q_star, x_test, y_test, **kwargs: Literal['x_train', 'y_train', 'x_init', 'y_init', 'net_parms_path', 'net_parms']):
    #     if net in (MLP, ANN):
    #         mod = net(q).to(device)
    #     else:
    #         mod = net(q, q_star).to(device)
    #     if 'net_parms_path' in kwargs:
    #         mod.load_state_dict(torch.load(kwargs.get('net_parms_path')))  # pre
    #         if 'x_train' in kwargs and 'y_train' in kwargs:  #sft
    #             if 'net_parms' in kwargs:
    #                 mod.train_net(kwargs.get('x_train'), kwargs.get('y_train'), **kwargs.get('net_parms'))
    #             else:
    #                 mod.train_net(kwargs.get('x_train'), kwargs.get('y_train'))
    #     else:
    #         if 'x_init' in kwargs and 'y_init' in kwargs:  # pre
    #             if 'net_parms' in kwargs:
    #                 mod.train_net(kwargs.get('x_init'), kwargs.get('y_init'), **kwargs.get('net_parms'))
    #             else:
    #                 mod.train_net(kwargs.get('x_init'), kwargs.get('y_init'))
    #             if 'x_train' in kwargs and 'y_train' in kwargs:  # sft
    #                 if 'net_parms' in kwargs:
    #                     mod.train_net(kwargs.get('x_train'), kwargs.get('y_train'), **kwargs.get('net_parms'))
    #                 else:
    #                     mod.train_net(kwargs.get('x_train'), kwargs.get('y_train'))
    #         else:  # train
    #             if 'x_train' in kwargs and 'y_train' in kwargs:
    #                 if 'net_parms' in kwargs:
    #                     mod.train_net(kwargs.get('x_train'), kwargs.get('y_train'), **kwargs.get('net_parms'))
    #                 else:
    #                     mod.train_net(kwargs.get('x_train'), kwargs.get('y_train'))
    #             else:  # train4test
    #                 if 'net_parms' in kwargs:
    #                     mod.train_net(x_test, y_test, **kwargs.get('net_parms'))
    #                 else:
    #                     mod.train_net(x_test, y_test)
    #     y_hat = mod.predictive(x_test, is2binary=True).cpu().numpy()
    #     return y_hat
    #
    #
    # def compare4models(data, q_apos, sample, mlp_parms_path=None, ann_parms_path=None, qnet_parms_path=None, net_parms=None, par=-2, verbose=False):
    #     q, q2tensor, q_star2tensor, x_train, y_train, x_test, y_test, irp, imp, x_init, y_init, x_train2tensor, y_train2tensor, x_test2tensor, y_test2tensor, irp2tensor, imp2tensor, x_init2tensor, y_init2tensor = get_subdata(data, q_apos, sample)
    #
    #     if mlp_parms_path:
    #         if net_parms:
    #             #init+sft
    #             y_hat_mlp = net_estimate(MLP, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms_path=mlp_parms_path, net_parms=net_parms, x_train=x_train2tensor, y_train=y_train2tensor)
    #         else:
    #             # init
    #             y_hat_mlp = net_estimate(MLP, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms_path=mlp_parms_path)
    #     elif net_parms:
    #         # tain+init+sft
    #         y_hat_mlp = net_estimate(MLP, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms=net_parms, x_train=x_train2tensor, y_train=y_train2tensor, x_init=x_init2tensor, y_init=y_init2tensor)
    #
    #     if ann_parms_path:
    #         if net_parms:
    #             y_hat_ann = net_estimate(ANN, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms_path=ann_parms_path, net_parms=net_parms, x_train=x_train2tensor, y_train=y_train2tensor)
    #         else:
    #             y_hat_ann = net_estimate(ANN, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms_path=ann_parms_path)
    #     elif net_parms:
    #         y_hat_ann = net_estimate(ANN, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms=net_parms, x_train=x_train2tensor, y_train=y_train2tensor, x_init=x_init2tensor, y_init=y_init2tensor)
    #
    #     if qnet_parms_path:
    #         if net_parms:
    #             y_hat_qnet = net_estimate(QNet, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms_path=qnet_parms_path, net_parms=net_parms, x_train=x_train2tensor, y_train=y_train2tensor)
    #         else:
    #             y_hat_qnet = net_estimate(QNet, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms_path=qnet_parms_path)
    #     elif net_parms:
    #         y_hat_qnet = net_estimate(QNet, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms=net_parms, x_train=x_train2tensor, y_train=y_train2tensor, x_init=x_init2tensor, y_init=y_init2tensor)
    #
    #     y_hat_dina = RCDM2py.DINA(q, x_test)
    #     y_hat_gdina = RCDM2py.GDINA(q, x_test)
    #     y_hat_npc = RCDM2py.NPC(q, x_test)
    #     y_hat_gnpc = RCDM2py.GNPC(q, x_test)
    #
    #     mlp_aar, mlp_par = class_rate4gdina(y_test, y_hat_mlp, par)
    #     ann_aar, ann_par = class_rate4gdina(y_test, y_hat_ann, par)
    #     qnet_aar, qnet_par = class_rate4gdina(y_test, y_hat_qnet, par)
    #     dina_aar, dina_par = class_rate4gdina(y_test, y_hat_dina, par)
    #     gdina_aar, gdina_par = class_rate4gdina(y_test, y_hat_gdina, par)
    #     npc_aar, npc_par = class_rate4gdina(y_test, y_hat_npc, par)
    #     gnpc_aar, gnpc_par = class_rate4gdina(y_test, y_hat_gnpc, par)
    #
    #     if verbose:
    #         print()
    #         print(f'mlp_aar:{mlp_aar}--ann_aar:{ann_aar}--QNet_aar:{qnet_aar}--dina_aar:{dina_aar}--gdina_aar:{gdina_aar}--npc_aar:{npc_aar}--gnpc_aar:{gnpc_aar}')
    #         print(f'mlp_par:{mlp_par}--ann_par:{ann_par}--QNet_par:{qnet_par}--dina_par:{dina_par}--gdina_par:{gdina_par}--npc_par:{npc_par}--gnpc_par:{gnpc_par}')
    #
    #     return (dina_aar, gdina_aar, npc_aar, gnpc_aar, mlp_aar, ann_aar, qnet_aar), (dina_par, gdina_par, npc_par, gnpc_par, mlp_par, ann_par, qnet_par)
    #
    #
    # def ex(data, q_apos, samples, rsd=20, save_path=None, mlp_parms_path=None, ann_parms_path=None, qnet_parms_path=None, net_parms=None, par=-2, verbose=True):
    #     df_aar = pd.DataFrame(columns=['sample', 'dina_aar', 'gdina_aar', 'npc_aar', 'gnpc_aar', 'mlp_aar', 'ann_aar', 'qnet_aar'])
    #     df_par = pd.DataFrame(columns=['sample', 'dina_par', 'gdina_par', 'npc_par', 'gnpc_par', 'mlp_par', 'ann_par', 'qnet_par'])
    #
    #     for sample in samples:
    #         for _ in range(rsd):
    #             aar, par = compare4models(HSD1, q_apos, sample, mlp_parms_path, ann_parms_path, qnet_parms_path, net_parms, par)
    #             df_aar.loc[len(df_aar)] = (sample, *aar)
    #             df_par.loc[len(df_par)] = (sample, *par)
    #
    #     df_par = pd.concat([df_par['sample']] + [df_par[col].apply(lambda x: pd.Series(x, index=[f"{col}_{i}" for i in range(len(x))])) for col in df_par.columns[1:]], axis=1)
    #
    #     df_aar_main = df_aar.groupby('sample').mean()
    #     df_par_main = df_par.groupby('sample').mean()
    #
    #     if verbose:
    #         pd_settings()
    #         print(df_aar_main)
    #         print(df_par_main)
    #
    #     if save_path:
    #         df_aar.to_csv(os.path.join(save_path, f'{data.data_name}_aar.csv'))
    #         df_aar_main.to_csv(os.path.join(save_path, f'{data.data_name}_aar_main.csv'))
    #
    #         df_par.to_csv(os.path.join(save_path, f'{data.data_name}_par.csv'))
    #         df_par_main.to_csv(os.path.join(save_path, f'{data.data_name}_par_main.csv'))
    #
    #     return df_aar, df_par


    # aar, par = ex(HSD1, 4, samples4sim, 5, None, mlp_parms_path, ann_parms_path, qnet_parms_path, hsd1_pre_paarms, -2, True)

    df_aar, df_aar_mean, df_par, df_par_mean = ex(**hsd1_sim_ex_parms)

    print('end')



