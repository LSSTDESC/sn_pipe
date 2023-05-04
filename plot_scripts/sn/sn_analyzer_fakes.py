#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 13:30:27 2023

@author: philippe.gris@clermont.in2p3.fr
"""
from sn_analysis.sn_tools import loadData_fakeSimu
import matplotlib.pyplot as plt
import operator
import numpy as np
from sn_analysis.sn_calc_plot import Calc_zlim, histSN_params, select
from sn_analysis.sn_calc_plot import plot_effi, effi
import glob
import pandas as pd
from optparse import OptionParser


filtercolors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['font.size'] = 20
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['lines.markersize'] = 15


def gime_zlim(df, dict_sel, selvar):

    df['sigmaC'] = np.sqrt(df['Cov_colorcolor'])
    # simulation parameters
    # histSN_params(df)

    sel = select(df, dict_sel[selvar])

    """
    fig, ax = plt.subplots()
    plot_effi(df, sel, leg='test', fig='tt', ax=ax)
    plot_2D(df)
    """
    effival = effi(df, sel)
    mycl = Calc_zlim(effival)
    zlim = mycl.zlim

    # mycl.plot_zlim(zlim)

    return zlim


def plot_2D(res, varx='z', legx='$', vary='sigmaC',
            legy='$\sigma C$', fig=None, ax=None):

    if fig is None:
        fig, ax = plt.subplots()

    ax.plot(res[varx], res[vary], 'ko')

    # plt.show()


def plot_res(df, fig=None, ax=None, label='', color='b', lst='solid', mark='None', mfc='None'):

    if fig is None:
        fig, ax = plt.subplots(figsize=(11, 6))

    idx = df['moon_frac'] < 0
    zlim_ref = df[idx]['zlim'].values[0]
    df.loc[idx, 'moon_frac'] = 0.
    print(zlim_ref)
    df['delta_zlim'] = df['zlim']-zlim_ref
    print(df)

    idx = df['moon_frac'] > -0.5
    sel = df[idx]
    sel = sel.sort_values(by=['moon_frac'])
    ax.plot(sel['moon_frac'], sel['delta_zlim'],
            label=label, color=color, linestyle=lst, marker=mark, mfc=mfc)
    # ax.grid()


def get_data(theDir, fis):

    df_dict = {}
    for fi in fis:
        print('processing', fi)
        ffi = fi.split('/')[-1]
        hname = ffi.split('_')[4]
        bb = ffi.split('_')[2]
        df = loadData_fakeSimu(theDir, ffi)
        key = '{}_{}'.format(bb, hname)
        if key not in df_dict.keys():
            df_dict[key] = df
        else:
            df_dict[key] = pd.concat((df, df_dict[key]))

    return df_dict


def plot_delta_zlim(df_dict, dict_sel, selvar):

    r = []
    for key, vals in df_dict.items():
        zlim = gime_zlim(vals, dict_sel, selvar)
        tt = key.split('_')
        bb = tt[0]
        hname = tt[1]
        print(hname, zlim)
        r.append((bb, int(hname), zlim))

    df = pd.DataFrame.from_records(r, columns=['band', 'moon_frac', 'zlim'])

    print(df)

    fig, ax = plt.subplots(figsize=(12, 7))
    leg = {}
    bands = 'rizy'
    for b in bands:
        leg[b] = 'u <-> '+b

    lst = dict(zip(bands, ['solid', 'dashed', 'dotted', 'dashdot']))
    marks = dict(zip(bands, ['o', 's', '^', 'h']))
    for b in bands:
        idx = df['band'] == b
        plot_res(df[idx], fig=fig, ax=ax, label=leg[b],
                 color=filtercolors[b], lst=lst[b], mark=marks[b])

    ax.set_xlabel('Moon Phase [%]', fontweight='bold')
    zcomp = '$z_{complete}$'
    ax.set_ylabel('$\Delta $ {}'.format(zcomp))

    ax.grid()
    ax.legend()


def plot_delta_nsn(df_dict, dict_sel, selvar, zmin=0.8):

    r = []

    for key, df in df_dict.items():
        tt = key.split('_')
        bb = tt[0]
        hname = tt[1]

        # print(hname, zlim)
        # r.append((bb, int(hname), zlim))
        df['sigmaC'] = np.sqrt(df['Cov_colorcolor'])
        # plt.plot(df['z'], df['sigmaC'], 'ko')
        # check_simuparams(df)
        seldf = select(df, dict_sel[selvar])
        idx = seldf['z'] >= zmin
        print(key, len(seldf), len(seldf[idx]))
        nsn_tot = len(seldf)
        nsn_07 = len(seldf[idx])
        r.append((bb, int(hname), nsn_tot, nsn_07))

    df = pd.DataFrame.from_records(
        r, columns=['band', 'moon_frac', 'nSN', 'nSN_0{}'.format(int(10*zmin))])

    fig, ax = plt.subplots(figsize=(12, 7))
    leg = {}
    bands = 'rizy'
    for b in bands:
        leg[b] = 'u <-> '+b

    lst = dict(zip(bands, ['solid', 'dashed', 'dotted', 'dashdot']))
    marks = dict(zip(bands, ['o', 's', '^', 'h']))

    vartoplot = 'nSN_0{}'.format(int(10*zmin))
    for b in bands:
        idx = df['band'] == b
        plot_resb(df[idx], fig=fig, ax=ax, var=vartoplot, label=leg[b],
                  color=filtercolors[b], lst=lst[b], mark=marks[b])

    if vartoplot != 'nSN':
        ttit = '$z\geq$'
        ttit += '{}'.format(np.round(zmin, 1))
        fig.suptitle(ttit)
    ax.set_xlabel('Moon Phase [%]', fontweight='bold')
    zcomp = '$N_{SN}$'
    ax.set_ylabel('$\Delta $ {}'.format(zcomp))
    ax.set_ylabel(
        r'$\frac{\mathrm{N_{SN}}}{\mathrm{N_{SN}}^{\mathrm{Moon~Phase=0}}}$')
    # ax.set_ylabel('{}'.format(legb))

    ax.legend()

    ax.grid()


def plot_resb(df, fig=None, ax=None, var='nSN', label='', color='b',
              lst='solid', mark='None', mfc='None'):

    if fig is None:
        fig, ax = plt.subplots(figsize=(11, 6))

    idx = df['moon_frac'] < 0
    nSN_ref = df[idx][var].values[0]
    df.loc[idx, 'moon_frac'] = 0.
    print(nSN_ref)
    p = df[var]/nSN_ref
    df['delta_nSN'] = p
    df['var_nSN'] = np.sqrt(nSN_ref*p*(1.-p))
    df['err_nSN'] = df['var_nSN']/nSN_ref
    print(df)

    idx = df['moon_frac'] > -0.5
    sel = df[idx]
    sel = sel.sort_values(by=['moon_frac'])
    ax.plot(sel['moon_frac'], sel['delta_nSN'], label=label,
            color=color, linestyle=lst, marker=mark, mfc=mfc)
    # ax.errorbar(sel['moon_frac'], sel['delta_nSN'], yerr=sel['err_nSN'],
    #            label=label, color=color, linestyle=lst, marker=mark, mfc=mfc)
    # ax.grid()


def check_simuparams(df):

    fig, ax = plt.subplots(ncols=2, nrows=2)

    vvars = ['x1', 'color', 'daymax', 'z']
    ppos = [(0, 0), (0, 1), (1, 0), (1, 1)]

    dpos = dict(zip(vvars, ppos))

    for key, vals in dpos.items():
        i = vals[0]
        j = vals[1]
        ax[i, j].hist(df[key], histtype='step')


parser = OptionParser()

parser.add_option("--inputDir", type="str",
                  default='Output_SN/Fakes', help="input dir [%default]")
parser.add_option("--plot_delta_zlim", type=int, default=0,
                  help="to plot delta_zlim [%default]")
parser.add_option("--plot_delta_nsn", type=int, default=1,
                  help="to plot delta_nsn [%default]")
parser.add_option("--zmin", type=float, default=0.8,
                  help="zmin to plot delta_nsn [%default]")

opts, args = parser.parse_args()

theDir = opts.inputDir
zmin = opts.zmin

fis = glob.glob('{}/SN_*.hdf5'.format(theDir))
# theFile = 'SN_conf_z_moon_-1_full_salt3.hdf5'


df_dict = get_data(theDir, fis)
print(df_dict.keys())

dict_sel = {}

dict_sel['G10'] = [('n_epochs_m10_p35', operator.ge, 4),
                   ('n_epochs_m10_p5', operator.ge, 1),
                   ('n_epochs_p5_p20', operator.ge, 1),
                   ('n_bands_m8_p10', operator.ge, 2),
                   ('sigmaC', operator.le, 0.04),
                   ]

dict_sel['metric'] = [('n_epochs_bef', operator.ge, 4),
                      ('n_epochs_aft', operator.ge, 10),
                      ('n_epochs_phase_minus_10', operator.ge, 1),
                      ('n_epochs_phase_plus_20', operator.ge, 1),
                      ('sigmaC', operator.le, 0.04),
                      ]


if opts.plot_delta_zlim:
    plot_delta_zlim(df_dict, dict_sel, 'G10')

if opts.plot_delta_nsn:
    plot_delta_nsn(df_dict, dict_sel, 'G10', zmin=zmin)

plt.show()
