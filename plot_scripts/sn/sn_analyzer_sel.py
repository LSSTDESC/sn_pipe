#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:50:54 2023

@author: philippe.gris@clermont.in2p3.fr
"""


import glob
from optparse import OptionParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sn_analysis.sn_calc_plot import bin_it


def load_OS(dbDir, dbName, runType, fieldType='DDF'):

    fullDir = '{}/{}/{}_{}'.format(dbDir, dbName, fieldType, runType)

    fis = glob.glob('{}/SN_{}*.hdf5'.format(fullDir, fieldType))

    df = pd.DataFrame()

    for fi in fis:
        aa = pd.read_hdf(fi)
        print('loading', fieldType, len(aa))
        df = pd.concat((df, aa))
        # break
    return df


def plot_nsn_z(df, xvar='z', xleg='z',
               bins=np.arange(0.005, 0.8, 0.01),
               norm_factor=30, ax=None, cumul=False):

    if ax is None:
        fig, ax = plt.subplots()

    res = bin_it(df, xvar=xvar, bins=bins, norm_factor=norm_factor)

    print(res)

    vv = res['NSN']
    if cumul:
        vv = np.cumsum(res['NSN'])
    ax.plot(res[xvar], vv, 'ko')


parser = OptionParser(description='Script to analyze SN prod after selection')

parser.add_option('--dbDir_DD', type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbList_DD', type=str,
                  default='input/DESC_cohesive_strategy/config_ana.csv',
                  help='OS DD list[%default]')
parser.add_option('--norm_factor_DD', type=int,
                  default=30,
                  help='DD normalization factor [%default]')
parser.add_option('--dbDir_WFD', type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--OS_WFD', type=str,
                  default='draft_connected_v2.99_10yrs',
                  help='OS WFD [%default]')
parser.add_option('--norm_factor_WFD', type=int,
                  default=10,
                  help='WFD normalization factor [%default]')
parser.add_option('--budget_DD', type=float,
                  default=0.07,
                  help='DD budget [%default]')
parser.add_option('--runType', type=str,
                  default='spectroz',
                  help='run type  [%default]')

opts, args = parser.parse_args()

dbDir_DD = opts.dbDir_DD
dbList_DD = opts.dbList_DD
norm_factor_DD = opts.norm_factor_DD
dbDir_WFD = opts.dbDir_WFD
OS_WFD = opts.OS_WFD
norm_factor_WFD = opts.norm_factor_WFD
budget_DD = opts.budget_DD
runType = opts.runType


wfd = load_OS(dbDir_WFD, OS_WFD, runType=runType, fieldType='WFD')

print('alors', wfd.columns)
fig, ax = plt.subplots()
figb, axb = plt.subplots()
plot_nsn_z(wfd, norm_factor=norm_factor_WFD, ax=ax, cumul=True)
plot_nsn_z(wfd, xvar='season', xleg='season', bins=np.arange(
    0.5, 11.5, 1), norm_factor=norm_factor_WFD, ax=axb)
print('allo', len(wfd)/norm_factor_WFD)
idx = wfd['sigmaC'] <= 0.04
plot_nsn_z(wfd[idx], norm_factor=norm_factor_WFD, ax=ax)
plot_nsn_z(wfd[idx], xvar='season', xleg='season', bins=np.arange(
    0.5, 11.5, 1), norm_factor=norm_factor_WFD, ax=axb)
print('allo', len(wfd[idx])/norm_factor_WFD)

ax.grid()
axb.grid()
plt.show()
