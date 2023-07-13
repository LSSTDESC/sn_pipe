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
    """
    Function to load OS data

    Parameters
    ----------
    dbDir : str
        data location dir.
    dbName : str
        db name to process.
    runType : str
        run type (spectroz or photoz).
    fieldType : str, optional
        field type (DDF or WFD). The default is 'DDF'.

    Returns
    -------
    df : pandas df
        OS data.

    """

    fullDir = '{}/{}/{}_{}'.format(dbDir, dbName, fieldType, runType)

    fis = glob.glob('{}/SN_{}*.hdf5'.format(fullDir, fieldType))

    df = pd.DataFrame()

    for fi in fis:
        aa = pd.read_hdf(fi)
        print('loading', fieldType, len(aa))
        df = pd.concat((df, aa))
        # break
    return df


class Plot_nsn_vs:
    def __init__(self, data, norm_factor, bins=np.arange(0.005, 0.8, 0.01),
                 xvar='z', xleg='z', logy=False, cumul=False, xlim=[0.01, 0.8]):

        self.data = data
        self.norm_factor = norm_factor
        self.bins = bins
        self.xvar = xvar
        self.xleg = xleg
        self.logy = logy
        self.cumul = cumul
        self.xlim = xlim

        self.plot_nsn_versus_two()

    def plot_nsn_versus(self, data, label='', ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 8))

        print('bins', self.bins)
        res = bin_it(data, xvar=self.xvar, bins=self.bins,
                     norm_factor=self.norm_factor)

        print(res)

        vv = res['NSN']
        if self.cumul:
            vv = np.cumsum(res['NSN'])
        ax.plot(res[self.xvar], vv, label=label)

        ax.set_xlabel(self.xleg)
        ax.set_ylabel(r'$N_{SN}$')
        ax.set_xlim(self.xlim)

    def plot_nsn_versus_two(self):

        fig, ax = plt.subplots(figsize=(14, 8))
        self.plot_nsn_versus(self.data, ax=ax)
        idx = self.data['sigmaC'] <= 0.04
        label = '$\sigma_C \leq 0.04$'
        self.plot_nsn_versus(self.data[idx], label=label, ax=ax)
        if self.logy:
            ax.set_yscale("log")

        ax.set_xlabel(self.xleg)
        ylabel = '$N_{SN}$'
        if self.cumul:
            ylabel = '$\sum N_{SN}$'
        ax.set_ylabel(r'{}'.format(ylabel))
        ax.legend()
        ax.grid()


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

# Plot_nsn_vs(wfd, norm_factor_WFD, xvar='z', xleg='z',
#            logy=True, cumul=True, xlim=[0.01, 0.7])
Plot_nsn_vs(wfd, norm_factor_WFD, bins=np.arange(
    0.5, 11.5, 1), xvar='season', xleg='season', logy=False, xlim=[1, 10])
"""
plot_nsn_two(wfd, norm_factor_WFD, xvar='z', xleg='z',
             logy=True, cumul=True, xlim=[0.01, 0.7])
plot_nsn_two(wfd, norm_factor_WFD, bins=np.arange(
    0.5, 11.5, 1), xvar='season', xleg='season', logy=False, xlim=[1, 10])
"""
plt.show()
