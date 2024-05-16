#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 08:24:50 2024

@author: gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import glob
from sn_tools.sn_utils import n_z
import matplotlib.pyplot as plt
import numpy as np


def mean_vs(data, varx='z', vary='sigma_mu', bins=np.arange(0.005, 0.11, 0.01)):

    group = data.groupby(pd.cut(data[varx], bins))

    _centers = (bins[:-1] + bins[1:])/2
    print('kkk', list(group[vary]))
    _values = group[vary].mean()

    df = pd.DataFrame(_centers, columns=[varx])

    df[vary] = list(_values)

    return df


def load_data(dataDir, dbName, zType, fieldType):
    """
    Function to load data

    Parameters
    ----------
    dataDir : str
        Data directory.
    dbName : str
        dbName to process.
    zType : str
        ztype (spectroz, photz, ...).
    fieldType : str
        Field type to process.

    Returns
    -------
    df : pandas df
        output data.

    """

    fullPath = '{}/{}/{}_{}'.format(dataDir, dbName, fieldType, zType)

    fis = glob.glob('{}/*.hdf5'.format(fullPath))

    df = pd.DataFrame()

    for fi in fis:
        dd = pd.read_hdf(fi)
        df = pd.concat((df, dd))

    return df


def plot_nsn_z(vals, fig=None, ax=None):

    if fig is None:
        fig, ax = plt.subplots()

    nsn_tot = len(vals)
    idx = vals['remove_sat'] == 0
    sel = vals[idx]
    print(key, len(sel), nsn_tot-len(sel), len(sel)/norm_factor)
    nsn_z = n_z(sel, norm_factor=norm_factor)
    ax.plot(nsn_z['z'], np.cumsum(nsn_z['nsn']))
    nsn_zb = n_z(vals[~idx], norm_factor=norm_factor)
    ax.plot(nsn_zb['z'], np.cumsum(nsn_zb['nsn']), linestyle='dashed')


def plot_sigma_mu_z(vals, varp='sigma_mu', fig=None, ax=None):

    if fig is None:
        fig, ax = plt.subplots()

    nsn_tot = len(vals)
    idx = vals['remove_sat'] == 0
    sel = vals[idx]
    print(key, len(sel), nsn_tot-len(sel), len(sel)/norm_factor)
    nsn_z = mean_vs(sel, varx='z', vary=varp)
    ax.plot(nsn_z['z'], nsn_z[varp])
    nsn_zb = mean_vs(vals[~idx], varx='z', vary=varp)
    ax.plot(nsn_zb['z'], nsn_zb[varp], linestyle='dashed')


parser = OptionParser()

parser.add_option("--dataDir", type=str, default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_G10_JLA',
                  help="data directory [%default]")

parser.add_option("--dbList", type=str, default='list_OS_new.csv',
                  help="db to process [%default]")
parser.add_option("--fieldType", type=str, default='WFD',
                  help="field type to process [%default]")

parser.add_option("--norm_factor", type=int, default=30,
                  help="simulation normalisation factor [%default]")
opts, args = parser.parse_args()

dataDir = opts.dataDir
dbList = opts.dbList
fieldType = opts.fieldType
norm_factor = opts.norm_factor

thelist = pd.read_csv(dbList, comment='#')

data_dict = {}
for i, ll in thelist.iterrows():
    dbName = ll['dbName']
    zType = ll['zType']
    tt = zType.split('sat_')[1]
    nn = '{}_{}'.format(dbName, tt)
    data_dict[nn] = load_data(dataDir, dbName, zType, fieldType)

fig, ax = plt.subplots()
figb, axb = plt.subplots()

for key, vals in data_dict.items():
    print('processing', key)
    idx = vals['z'] <= 0.11
    vals = vals[idx]
    # plot_nsn_z(vals, fig=fig, ax=ax)
    # plot_sigma_mu_z(vals, fig=figb, ax=axb)
    ax.plot(vals['z'], vals['sigma_mu'], 'ko')

plt.show()
