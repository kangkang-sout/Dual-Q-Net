#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：DualQNet -> train.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/12/9
@Desc   ：
==================================================
"""
import os.path

import numpy as np
import torch
from typing_extensions import Literal

import pandas as pd

from data import HSD1,HSD2, LSD1, LSD2, EDM, FS, SimDataSet
from configs import device
from rmodels import RCDM2py
from utils import numpy2tensor, get_Qstar, get_q_apos, plot4dfs, class_rate4gdina, pd_settings
from models import MLP, ANN, QNet


def train_data(data: SimDataSet, q_apos, is_init=True, is_ideal=False, samp_size_train=500, samp_size_test=100, verbose=False):
    q = data.q
    if is_ideal:
        x_train, y_train = data.irp, data.imp
    else:
        if is_init:
            x_train, y_train = data.get_init_data4train()
        else:
            x_train, _, y_train, _ = data.get_sub_data4train_test(samp_size_train)

    samp_size_test = samp_size_test if isinstance(samp_size_test, (int)) else round(samp_size_train * samp_size_test)
    _, x_test, _, y_test = data.get_sub_data4train_test(samp_size_test)

    if isinstance(q_apos, (int, float)):
        q_apos = get_q_apos(q_apos, q.shape[1])
        q_star = get_Qstar(q, q_apos)
    elif isinstance(q_apos, (np.ndarray, pd.DataFrame)):
        q_star = get_Qstar(q, q_apos)
    x_train, y_train, x_test, y_test, q, q_star = numpy2tensor(device, x_train, y_train, x_test, y_test, q, q_star)
    if verbose:
        print(f'\n y_train_shape:{y_train.shape}--y_test_shape:{y_test.shape}--q_shape:{q.shape}--s_star_shape:{q_star.shape}')
    return x_train, y_train, x_test, y_test, q, q_star


def get_subdata(data, q_apos, samp_size=500, is2tensor=True):
    q = data.q
    x_train, x_test, y_train, y_test = data.get_sub_data4train_test(samp_size)
    irp, imp = data.irp, data.imp
    x_init, y_init = data.get_init_data4train()
    if isinstance(q_apos, (int, float)):
        q_apos = get_q_apos(q_apos, q.shape[1])
        q_star = get_Qstar(q, q_apos)
    elif isinstance(q_apos, (np.ndarray, pd.DataFrame)):
        q_star = get_Qstar(q, q_apos)
    if is2tensor:
        x_train2tensor, x_test2tensor, y_train2tensor, y_test2tensor, irp2tensor, imp2tensor, x_init2tensor, y_init2tensor, q2tensor, q_star2tensor = numpy2tensor(device, x_train, x_test, y_train, y_test, irp, imp, x_init, y_init, q, q_star)
    return q, q2tensor, q_star2tensor, x_train, y_train, x_test, y_test, irp, imp, x_init, y_init, x_train2tensor, y_train2tensor, x_test2tensor, y_test2tensor, irp2tensor, imp2tensor, x_init2tensor, y_init2tensor

def train_net(net, x_train, y_train, q, q_star, lr=0.01, epochs=100, batch_size=64, shuffle=True, num_workers=0,
              verbose=False, **kwargs: Literal['x_test, y_test', 'pt_path', 'log_path']):
    if net in (MLP, ANN):
        mod = net(q).to(device)
    else:
        mod = net(q, q_star).to(device)

    if 'x_test' in kwargs:
        log = mod.train_net(x_train, y_train, x_test=kwargs.get('x_test'), y_test=kwargs.get('y_test'), lr=lr, epochs=epochs, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, verbose=verbose)
    else:
        log = mod.train_net(x_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, verbose=verbose)
    return mod, log


def mult_train_net(net_list, data, q_apos, is_init=True, is_ideal=False, samp_size_train=500, samp_size_test=100,
                   lr=0.01, epochs=100, batch_size=64, shuffle=True, num_workers=0, verbose=True):
    x_train, y_train, x_test, y_test, q, q_star = train_data(data, q_apos, is_init, is_ideal, samp_size_train, samp_size_test, verbose)
    mod_list = []
    df_Tran_loss, df_Train_AAR, df_Train_PAR = pd.DataFrame(), pd.DataFrame(),pd.DataFrame()
    df_Val_loss, df_Val_AAR, df_Val_PAR = pd.DataFrame(), pd.DataFrame(),pd.DataFrame()

    for net in net_list:
        mod, df_log = train_net(net, x_train, y_train, q, q_star, lr, epochs, batch_size, shuffle, num_workers, verbose, x_test=x_test, y_test=y_test)
        mod_list.append(mod)
        df_Tran_loss[df_log.columns[1]] = df_log[df_log.columns[1]]
        df_Train_AAR[df_log.columns[2]] = df_log[df_log.columns[2]]
        df_Train_PAR[df_log.columns[3]] = df_log[df_log.columns[3]]
        df_Val_loss[df_log.columns[4]] = df_log[df_log.columns[4]]
        df_Val_AAR[df_log.columns[5]] = df_log[df_log.columns[5]]
        df_Val_PAR[df_log.columns[6]] = df_log[df_log.columns[6]]

    if verbose:
        plot4dfs(df_Tran_loss, df_Train_AAR, df_Train_PAR, df_Val_loss, df_Val_AAR, df_Val_PAR)
    return mod_list, (df_Tran_loss, df_Train_AAR, df_Train_PAR, df_Val_loss, df_Val_AAR, df_Val_PAR)


def loadNet(net, net_parms_path, q, q_star):
    if net in (MLP, ANN):
        mod = net(q).to(device)
    else:
        mod = net(q, q_star).to(device)
    mod.load_state_dict(torch.load(net_parms_path))
    return mod


def loadNets(net_parms: dict, q, q_star):
    mods = []
    for net, parm in net_parms.items():
        mods.append(loadNet(net, parm, q, q_star))
    return mods


def save_mods_logs(mods, logs, path):
    for mod in mods:
        parms = mod.state_dict()
        torch.save(parms, os.path.join(path, f'pth/{mod._name}_pre.pt'))

    df_logs = pd.concat(logs, axis=1)
    df_logs.to_csv(os.path.join(path, f'acc/{mod._name}_pre.csv'))


def net_estimate(net, q, q_star, x_test, y_test, **kwargs: Literal['x_train', 'y_train', 'x_init', 'y_init', 'net_parms_path', 'net_parms']):
    if net in (MLP, ANN):
        mod = net(q).to(device)
    else:
        mod = net(q, q_star).to(device)
    if 'net_parms_path' in kwargs:
        mod.load_state_dict(torch.load(kwargs.get('net_parms_path')))  # pre
        if 'x_train' in kwargs and 'y_train' in kwargs:  #sft
            if 'net_parms' in kwargs:
                mod.train_net(kwargs.get('x_train'), kwargs.get('y_train'), **kwargs.get('net_parms'))
            else:
                mod.train_net(kwargs.get('x_train'), kwargs.get('y_train'))
    else:
        if 'x_init' in kwargs and 'y_init' in kwargs:  # pre
            if 'net_parms' in kwargs:
                mod.train_net(kwargs.get('x_init'), kwargs.get('y_init'), **kwargs.get('net_parms'))
            else:
                mod.train_net(kwargs.get('x_init'), kwargs.get('y_init'))
            if 'x_train' in kwargs and 'y_train' in kwargs:  # sft
                if 'net_parms' in kwargs:
                    mod.train_net(kwargs.get('x_train'), kwargs.get('y_train'), **kwargs.get('net_parms'))
                else:
                    mod.train_net(kwargs.get('x_train'), kwargs.get('y_train'))
        else:  # train
            if 'x_train' in kwargs and 'y_train' in kwargs:
                if 'net_parms' in kwargs:
                    mod.train_net(kwargs.get('x_train'), kwargs.get('y_train'), **kwargs.get('net_parms'))
                else:
                    mod.train_net(kwargs.get('x_train'), kwargs.get('y_train'))
            else:  # train4test
                if 'net_parms' in kwargs:
                    mod.train_net(x_test, y_test, **kwargs.get('net_parms'))
                else:
                    mod.train_net(x_test, y_test)
    y_hat = mod.predictive(x_test, is2binary=True).cpu().numpy()
    return y_hat


def compare4models(data, q_apos, sample, mlp_parms_path=None, ann_parms_path=None, qnet_parms_path=None, net_parms=None, par_digit=-2, verbose=False):
    q, q2tensor, q_star2tensor, x_train, y_train, x_test, y_test, irp, imp, x_init, y_init, x_train2tensor, y_train2tensor, x_test2tensor, y_test2tensor, irp2tensor, imp2tensor, x_init2tensor, y_init2tensor = get_subdata(data, q_apos, sample)

    if mlp_parms_path:
        if net_parms:
            #init+sft
            y_hat_mlp = net_estimate(MLP, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms_path=mlp_parms_path, net_parms=net_parms, x_train=x_train2tensor, y_train=y_train2tensor)
        else:
            # init
            y_hat_mlp = net_estimate(MLP, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms_path=mlp_parms_path)
    elif net_parms:
        # tain+init+sft
        y_hat_mlp = net_estimate(MLP, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms=net_parms, x_train=x_train2tensor, y_train=y_train2tensor, x_init=x_init2tensor, y_init=y_init2tensor)

    if ann_parms_path:
        if net_parms:
            y_hat_ann = net_estimate(ANN, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms_path=ann_parms_path, net_parms=net_parms, x_train=x_train2tensor, y_train=y_train2tensor)
        else:
            y_hat_ann = net_estimate(ANN, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms_path=ann_parms_path)
    elif net_parms:
        y_hat_ann = net_estimate(ANN, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms=net_parms, x_train=x_train2tensor, y_train=y_train2tensor, x_init=x_init2tensor, y_init=y_init2tensor)

    if qnet_parms_path:
        if net_parms:
            y_hat_qnet = net_estimate(QNet, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms_path=qnet_parms_path, net_parms=net_parms, x_train=x_train2tensor, y_train=y_train2tensor)
        else:
            y_hat_qnet = net_estimate(QNet, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms_path=qnet_parms_path)
    elif net_parms:
        y_hat_qnet = net_estimate(QNet, q2tensor, q_star2tensor, x_test2tensor, y_test2tensor, net_parms=net_parms, x_train=x_train2tensor, y_train=y_train2tensor, x_init=x_init2tensor, y_init=y_init2tensor)

    y_hat_dina = RCDM2py.DINA(q, x_test)
    y_hat_gdina = RCDM2py.GDINA(q, x_test)
    y_hat_npc = RCDM2py.NPC(q, x_test)
    y_hat_gnpc = RCDM2py.GNPC(q, x_test)

    mlp_aar, mlp_par = class_rate4gdina(y_test, y_hat_mlp, par_digit)
    ann_aar, ann_par = class_rate4gdina(y_test, y_hat_ann, par_digit)
    qnet_aar, qnet_par = class_rate4gdina(y_test, y_hat_qnet, par_digit)
    dina_aar, dina_par = class_rate4gdina(y_test, y_hat_dina, par_digit)
    gdina_aar, gdina_par = class_rate4gdina(y_test, y_hat_gdina, par_digit)
    npc_aar, npc_par = class_rate4gdina(y_test, y_hat_npc, par_digit)
    gnpc_aar, gnpc_par = class_rate4gdina(y_test, y_hat_gnpc, par_digit)

    if verbose:
        print()
        print(f'mlp_aar:{mlp_aar}--ann_aar:{ann_aar}--QNet_aar:{qnet_aar}--dina_aar:{dina_aar}--gdina_aar:{gdina_aar}--npc_aar:{npc_aar}--gnpc_aar:{gnpc_aar}')
        print(f'mlp_par:{mlp_par}--ann_par:{ann_par}--QNet_par:{qnet_par}--dina_par:{dina_par}--gdina_par:{gdina_par}--npc_par:{npc_par}--gnpc_par:{gnpc_par}')

    return (dina_aar, gdina_aar, npc_aar, gnpc_aar, mlp_aar, ann_aar, qnet_aar), (dina_par, gdina_par, npc_par, gnpc_par, mlp_par, ann_par, qnet_par)


def ex(data, q_apos, samples, rsd=20, save_path=None, mlp_parms_path=None, ann_parms_path=None, qnet_parms_path=None, net_parms=None, par_digit=-2, verbose=True):
    df_aar = pd.DataFrame(columns=['sample', 'dina_aar', 'gdina_aar', 'npc_aar', 'gnpc_aar', 'mlp_aar', 'ann_aar', 'qnet_aar'])
    df_par = pd.DataFrame(columns=['sample', 'dina_par', 'gdina_par', 'npc_par', 'gnpc_par', 'mlp_par', 'ann_par', 'qnet_par'])

    for sample in samples:
        if verbose:
            print(f'{data.data_name}--sample:{sample}')
        for _ in range(rsd):
            aar, par = compare4models(data, q_apos, sample, mlp_parms_path, ann_parms_path, qnet_parms_path, net_parms, par_digit)
            df_aar.loc[len(df_aar)] = (sample, *aar)
            df_par.loc[len(df_par)] = (sample, *par)

    df_par = pd.concat([df_par['sample']] + [df_par[col].apply(lambda x: pd.Series(x, index=[f"{col}_{i}" for i in range(len(x))])) for col in df_par.columns[1:]], axis=1)

    df_aar_main = df_aar.groupby('sample').mean()
    df_par_main = df_par.groupby('sample').mean()

    if verbose:
        pd_settings()
        print(df_aar_main)
        print(df_par_main)

    if save_path:
        df_aar.to_csv(os.path.join(save_path, f'{data.data_name}_aar.csv'))
        df_aar_main.to_csv(os.path.join(save_path, f'{data.data_name}_aar_main.csv'))

        df_par.to_csv(os.path.join(save_path, f'{data.data_name}_par.csv'))
        df_par_main.to_csv(os.path.join(save_path, f'{data.data_name}_par_main.csv'))

    return df_aar, df_aar_main, df_par, df_par_main







if __name__ == '__main__':
    nets = [MLP, ANN, QNet]
    # mods, logs = mult_train_net(nets, HSD1, 4, is_ideal=True, lr=0.01, epochs=500)
    mods, logs = mult_train_net(nets, HSD1, 4, lr=0.01, epochs=100)
    print('end')

    # path = '../data/output/hsd1/pre_train'
    # save_mods_logs(mods, logs, path)

