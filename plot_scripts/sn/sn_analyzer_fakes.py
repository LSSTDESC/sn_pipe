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
from sn_analysis.sn_calc_plot import Calc_zlim, histSN_params, select, plot_effi, effi
import glob
import pandas as pd


def gime_zlim(df):

    df['sigmaC'] = np.sqrt(df['Cov_colorcolor'])
    # simulation parameters
    # histSN_params(df)

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

    sel = select(df, dict_sel['metric'])

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


theDir = 'Output_SN/Fakes'

fis = glob.glob('{}/SN_*.hdf5'.format(theDir))
# theFile = 'SN_conf_z_moon_-1_full_salt3.hdf5'

r = []
for fi in fis:
    ffi = fi.split('/')[-1]
    hname = ffi.split('_')[4]
    df = loadData_fakeSimu(theDir, ffi)
    zlim = gime_zlim(df)
    print(hname, zlim)
    r.append((int(hname), zlim))

df = pd.DataFrame.from_records(r, columns=['moon_frac', 'zlim'])

idx = df['moon_frac'] < 0
zlim_ref = df[idx]['zlim'].values[0]
print(zlim_ref)
df['delta_zlim'] = df['zlim']-zlim_ref
print(df)
fig, ax = plt.subplots()
idx = df['moon_frac'] > 0
sel = df[idx]
sel = sel.sort_values(by=['moon_frac'])
ax.plot(sel['moon_frac'], sel['delta_zlim'])
plt.show()
