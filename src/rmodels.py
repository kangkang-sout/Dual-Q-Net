#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================
@Project -> File   ：DualQNet -> rmodels.py
@IDE    ：PyCharm
@Author ：jhong.tao
@Date   ：2023/12/10
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
# from src import utils
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