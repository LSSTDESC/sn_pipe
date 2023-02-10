#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:47:18 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import matplotlib.pyplot as plt
import numpy as np
from sn_tools.sn_io import loopStack
import pandas as pd
import glob
import operator


def loadData(theDir, dbName):
    """
    Funtion to load data

    Parameters
    ----------
    theDir : str
        location dir.
    dbName : str
        dbName.

    Returns
    -------
    res : astropytable
        loaded data.

    """

    searchname = '{}/{}/*.hdf5'.format(theDir, dbName)
    files = glob.glob(searchname)
    res = loopStack(files, objtype='astropyTable')

    return res.to_pandas()


def load_complete(dbDir, dbName, alpha=0.4, beta=3):

    res = loadData(dbDir, dbName)
    res = complete_df(res, alpha, beta)

    return res


def complete_df(res, alpha=0.4, beta=3):
    """
    Function to complete df infos

    Parameters
    ----------
    res : pandas df
        df to complete.
    alpha : floar, optional
        alpha parameter for the estimation of sigma_mu. The default is 0.4.
    beta : float, optional
        beta parameter for the estimation of sigma_mu. The default is 3.

    Returns
    -------
    res : pandas df
        completed df.

    """

    res['sigmaC'] = np.sqrt(res['Cov_colorcolor'])
    res['sigmat0'] = np.sqrt(res['Cov_t0t0'])
    res['Cov_mbmb'] = (
        2.5 / (res['x0_fit']*np.log(10)))**2*res['Cov_x0x0']
    res['Cov_x1mb'] = -2.5*res['Cov_x0x1'] / \
        (res['x0_fit']*np.log(10))
    res['Cov_colormb'] = -2.5*res['Cov_x0color'] / \
        (res['x0_fit']*np.log(10))

    res['sigma_mu'] = res.apply(lambda x: sigma_mu(x, alpha, beta), axis=1)
    return res


def sigma_mu(grp, alpha, beta):
    """
    Function to estimate sigma_mu

    Parameters
    ----------
    grp : pandas df
        data to use to estimate sigma_mu.
    alpha : float
          alpha parameter to estimate sigma_mu.
    beta : float
        beta parameter to estimate sigma_mu.

    Returns
    -------
    float
        sigma_mu.

    """

    res = grp.Cov_mbmb+(alpha**2)*grp.Cov_x1x1+(beta**2)*grp.Cov_colorcolor\
        + 2*alpha*grp.Cov_x1mb-2*beta*grp.Cov_colormb\
        - 2*alpha*beta*grp.Cov_x1color

    return np.sqrt(res)


def histSN_params(data, vars=['x1', 'color', 'z', 'daymax']):
    """
    Function to plot SN parameters

    Parameters
    ----------
    data : pandas df
        data to plot.
    vars : list(str), optional
        params to plot. The default is ['x1', 'color', 'z', 'daymax'].

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    ipos = [(0, 0), (0, 1), (1, 0), (1, 1)]

    jpos = dict(zip(vars, ipos))

    for key, vals in jpos.items():
        ax[vals].hist(data[key])
        ax[vals].set_xlabel(key)
        ax[vals].set_ylabel('Number of Entries')


def plotSN_2D(data, varx='z', legx='z', vary='sigma_mu', legy='$\sigma_{\mu}$'):
    """
    function to perform 2D plots

    Parameters
    ----------
    data : pandas df
        data to plot.
    varx : str, optional
        x-axis var. The default is 'z'.
    legx : str, optional
        x-axis label. The default is 'z'.
    vary : str, optional
        y-axis var. The default is 'sigma_mu'.
    legy : str, optional
        y-axis label. The default is '$\sigma_{\mu}$'.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data[varx], data[vary], 'k.')

    ax.set_xlabel(legx)
    ax.set_ylabel(legy)


def plotSN_2D_binned(data, varx='z', legx='z', bins=np.arange(0.5, 0.6, 0.01),
                     vary='sigma_mu', legy='$\sigma_{\mu}$'):
    """
    function to perform 2D plots

    Parameters
    ----------
    data : pandas df
        data to plot.
    varx : str, optional
        x-axis var. The default is 'z'.
    legx : str, optional
        x-axis label. The default is 'z'.
    vary : str, optional
        y-axis var. The default is 'sigma_mu'.
    legy : str, optional
        y-axis label. The default is '$\sigma_{\mu}$'.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(10, 6))

    group = data.groupby(pd.cut(data[varx], bins))
    bin_centers = (bins[: -1] + bins[1:])/2
    y = group[vary].mean()
    yerr = group[vary].std()
    ax.errorbar(bin_centers, y, yerr=yerr, color='k', marker='.')

    ax.set_xlabel(legx)
    ax.set_ylabel(legy)


def plotSN_effi(data, xvar='n_epochs_aft', bins=range(1, 20, 1),
                var_cut='sigmaC', var_sel=0.04, op=operator.le):
    """
    Function to estimate and plot efficiency

    Parameters
    ----------
    data : pandas df
        data to process.
    xvar : str, optional
        x-axis var. The default is 'n_epochs_aft'.
    bins : list(int), optional
        bins for efficiency estimation. The default is range(0, 20, 1).
    var_cut : str, optional
        selection var. The default is 'sigmaC'.
    var_sel : float, optional
        selection val. The default is 0.04.
    op : operator, optional
        operator for sel. The default is operator.le.

    Returns
    -------
    None.

    """

    group = data.groupby(pd.cut(data[xvar], bins))
    idx = op(data[var_cut], var_sel)
    sel_data = data[idx]
    group_sel = sel_data.groupby(pd.cut(sel_data[xvar], bins))

    # estimate efficiency here
    effi = group_sel.size()/group.size()

    print(effi)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(bins[:-1], effi, 'ko')


def effi(resa, resb, xvar='z', bins=np.arange(0.01, 1.1, 0.02)):

    groupa = resa.groupby(pd.cut(resa[xvar], bins))
    groupb = resb.groupby(pd.cut(resb[xvar], bins))

    effi = groupb.size()/groupa.size()

    bin_centers = (bins[: -1] + bins[1:])/2

    return bin_centers, effi


def plot_effi(resa, resb, xvar='z', leg='',
              bins=np.arange(0.01, 1.1, 0.02), fig=None, ax=None):

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    x, y = effi(resa, resb, xvar=xvar, bins=bins)

    ax.plot(x, y, label=leg)


def select(res, list_sel):
    """
    Function to select a pandas df

    Parameters
    ----------
    res : pandas df
        data to select.

    Returns
    -------
    pandas df
        selected df.

    """
    idx = True
    for vals in list_sel:
        idx &= vals[1](res[vals[0]], vals[2])

    return res[idx]


dbName = 'draft_connected_v2.99_10yrs'
res = load_complete('Output_SN', dbName)
res_fast = load_complete('Output_SN_fast', dbName)

histSN_params(res_fast)
plotSN_2D(res_fast)
print(res.columns)

dict_sel = {}

dict_sel['G10'] = [('n_epochs_m10_p35', operator.ge, 4),
                   ('n_epochs_m10_p5', operator.ge, 1),
                   ('n_epochs_p5_p20', operator.ge, 1),
                   ('n_bands_m8_p10', operator.ge, 2),
                   ('sigmaC', operator.le, 0.04)]

sel = select(res, dict_sel['G10'])
fig, ax = plt.subplots()
ax.hist(sel['n_epochs_aft'], histtype='step')

fig, ax = plt.subplots()
ax.hist(sel['n_epochs_bef'], histtype='step')

plt.show()
combi = []
for nbef in range(3, 5):
    for naft in range(3, 11):
        combi.append((nbef, naft))

combi = [(3, 8), (4, 10), (3, 6)]
for (nbef, naft) in combi:
    print(nbef, naft)
    seltype = 'metric_{}_{}'.format(nbef, naft)
    dict_sel[seltype] = [('n_epochs_phase_minus_10', operator.ge, 1),
                         ('n_epochs_phase_plus_20', operator.ge, 1),
                         ('n_epochs_bef', operator.ge, nbef),
                         ('n_epochs_aft', operator.ge, naft),
                         ('sigmaC', operator.le, 0.04)]

fig, ax = plt.subplots(figsize=(10, 8))
for key, vals in dict_sel.items():
    sel = select(res, vals)
    selfast = select(res_fast, vals)
    print(key, len(sel), len(selfast))
    plot_effi(res, sel, leg=key, fig=fig, ax=ax)
    plot_effi(res, selfast, leg=key, fig=fig, ax=ax)

ax.legend()

# plt.show()

"""
plotSN_2D(sel, vary='sigmaC', legy='$\sigma_C$')
plotSN_2D(sel, vary='sigmat0', legy='$\sigma_{T_0}$')
plotSN_2D_binned(sel, bins=np.arange(0.1, 0.7, 0.01),
                 vary='sigmaC', legy='$\sigma_C$')
plotSN_2D(sel, varx='n_epochs_aft',
          legx='N$_{epochs}^{aft}$', vary='sigmaC', legy='$\sigma_C$')

"""


"""
plotSN_2D(sel, varx='n_epochs_bef',
          legx='N$_{epochs}^{bef}$', vary='sigmaC', legy='$\sigma_C$')
"""
plt.show()
"""
plotSN_effi(sel)
plt.show()
"""
