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


dbDir = 'Output_SN'
dbName = 'draft_connected_v2.99_10yrs'
res = loadData(dbDir, dbName)
res = complete_df(res, alpha=0.4, beta=3)

histSN_params(res)
plotSN_2D(res)
print(res.columns)

idx = res['n_epochs_phase_minus_10'] >= 1
idx &= res['n_epochs_phase_plus_20'] >= 1
idx &= res['n_epochs_bef'] >= 4
idx &= res['n_epochs_aft'] >= 10
sel = res[idx]

plotSN_2D(sel, vary='sigmaC', legy='$\sigma_C$')
plotSN_2D(sel, vary='sigmat0', legy='$\sigma_{T_0}$')
plotSN_2D_binned(sel, bins=np.arange(0.1, 0.7, 0.01),
                 vary='sigmaC', legy='$\sigma_C$')
plotSN_2D(sel, varx='n_epochs_aft',
          legx='N$_{epochs}^{aft}$', vary='sigmaC', legy='$\sigma_C$')

plotSN_2D(sel, varx='n_epochs_bef',
          legx='N$_{epochs}^{bef}$', vary='sigmaC', legy='$\sigma_C$')


plotSN_effi(sel)
plt.show()
