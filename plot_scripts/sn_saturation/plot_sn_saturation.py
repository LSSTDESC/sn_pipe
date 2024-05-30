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
from sn_analysis.sn_tools import get_spline

zmax = 0.026


def mean_vs(data, varx='z', vary='sigma_mu', bins=np.arange(0.005, 0.11, 0.01)):

    group = data.groupby(pd.cut(data[varx], bins))

    _centers = (bins[:-1] + bins[1:])/2
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


def plot_nsn_z(vals, fig=None, ax=None, cumul=False):

    if fig is None:
        fig, ax = plt.subplots()

    nsn_tot = len(vals)
    idx = vals['z'] <= zmax
    sel = vals[idx]
    delta = 0.002
    bins = np.arange(0.01-delta/2, zmax+delta, delta)
    print(key, len(sel), nsn_tot-len(sel), len(sel)/norm_factor)
    nsn_z = n_z(sel, norm_factor=norm_factor,
                bins=bins)
    print(nsn_z[['z', 'nsn']])
    nn = nsn_z['nsn']
    if cumul:
        nn = np.cumsum(nn)
    ax.plot(nsn_z['z'], nn)
    # nsn_zb = n_z(vals[~idx], norm_factor=norm_factor)
    # ax.plot(nsn_zb['z'], np.cumsum(nsn_zb['nsn']), linestyle='dashed')
    ax.grid(visible=True)
    ax.set_xlim([0.01, 0.025])
    # ax.set_ylim([0.0, 10.])
    ax.set_xlabel('$z$')
    ax.set_ylabel('$\sum N_{SN}$')


def plot_sigma_mu_z(vals, varp='sigma_mu',
                    fig=None, ax=None,
                    type='all',
                    ratio=False,
                    label='',
                    marker='o', color='k', ls='solid'):

    if fig is None:
        fig, ax = plt.subplots()

    nsn_tot = len(vals)
    idx = vals['remove_sat'] == 0
    sel = vals[idx]
    selb = vals[~idx]
    """
    print(key, len(sel), nsn_tot-len(sel), len(sel) /
          norm_factor, len(selb)/norm_factor)
    print(sel[['z', 'sigma_c', 'remove_sat']])
    print(selb[['z', 'sigma_c', 'remove_sat']])
    """
    delta = 0.001
    bins = np.arange(0.01-delta/2, zmax+delta, delta)
    if type == 'mean':
        nsn_z = mean_vs(sel, varx='z', vary=varp, bins=bins)
        nsn_zb = mean_vs(vals[~idx], varx='z', vary=varp, bins=bins)
        if not ratio:
            ax.plot(nsn_z['z'], nsn_z[varp])
            ax.plot(nsn_zb['z'], nsn_zb[varp], linestyle='dashed')
        else:
            tt = nsn_z.merge(nsn_zb, left_on=['z'], right_on=['z'])
            tt['ratio'] = tt['sigma_mu_y']/tt['sigma_mu_x']
            print(tt[['z', 'ratio']])
            x, y = get_spline(tt, 'z', 'ratio')
            ax.plot(x, y, label=label, marker=marker,
                    color=color, linestyle=ls, mfc='None', ms=10, markevery=5)
            ax.plot(tt['z'], tt['ratio'])
    if type == 'all':
        ax.plot(sel['z'], sel[varp], 'ko')
        ax.plot(selb['z'], selb[varp], 'r*')

    ax.grid(visible=True)
    ax.set_xlim([0.01, zmax])
    ax.set_xlabel(r'$z$')
    rr = '$\frac{\sigma_{\mu}^{no~LC~sat}{\sigma_{\mu}}$'
    # ax.set_ylabel(r'{}'.format(rr))
    ax.set_ylabel(r'$\frac{\sigma_{\mu}^{no~LC~sat}}{\sigma_{\mu}}$')


parser = OptionParser()

parser.add_option("--dataDir", type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_G10_JLA',
                  help="data directory [%default]")

parser.add_option("--dbList", type=str, default='list_OS_new.csv',
                  help="db to process [%default]")
parser.add_option("--fieldType", type=str, default='WFD',
                  help="field type to process [%default]")

parser.add_option("--norm_factor", type=int, default=50,
                  help="simulation normalisation factor [%default]")
opts, args = parser.parse_args()

dataDir = opts.dataDir
dbList = opts.dbList
fieldType = opts.fieldType
norm_factor = opts.norm_factor

thelist = pd.read_csv(dbList, comment='#')

data_dict = {}
display_dict = {}
for i, ll in thelist.iterrows():
    dbName = ll['dbName']
    zType = ll['zType']
    tt = zType.split('sat_')[1]
    nn = '{}_{}'.format(dbName, tt)
    print(nn)
    data_dict[nn] = load_data(dataDir, dbName, zType, fieldType)
    dis_dict = {}
    for kk in ['marker', 'color', 'ls']:
        dis_dict[kk] = ll[kk]
    display_dict[nn] = dis_dict

print(display_dict)

fig, ax = plt.subplots(figsize=(15, 12))
figb, axb = plt.subplots(figsize=(15, 12))

dbref = 'onesnap'
dbref = 'baseline'
for key, vals in data_dict.items():
    spl = key.split('_')
    fullwell = spl[-1]
    dbName = '_'.join(spl[:3])
    pp = key.split('{}_'.format(dbName))[-1].split('_{}'.format(fullwell))[:-1]
    print('aooo', pp)
    psf = '_'.join(pp)
    print('processing', dbName, fullwell, psf)
    leg = '{} - {}kpe'.format(psf, int(int(fullwell)/1000))
    # idx = vals['z'] <= zmax+0.01
    # vals = vals[idx]
    print('llll', np.max(vals['z']))
    if key == '{}_v3.4_10yrs_moffat_90000'.format(dbref):
        plot_nsn_z(vals, fig=fig, ax=ax, cumul=True)
    dd = display_dict[key]
    plot_sigma_mu_z(vals, fig=figb, ax=axb, type='mean',
                    ratio=True, label=leg, marker=dd['marker'],
                    color=dd['color'], ls=dd['ls'])
    # ax.plot(vals['z'], vals['sigma_mu'], 'ko')

# ax.set_xlim([0.01, 0.03])
axb.set_xlim([0.01, 0.025])
axb.legend()
plt.show()
