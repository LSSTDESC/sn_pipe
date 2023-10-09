#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:26:27 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
import matplotlib.pyplot as plt
from sn_analysis.sn_calc_plot import bin_it_combi
from astropy.cosmology import w0waCDM
import numpy as np
from sn_plotter_cosmology import plt


def cosmoVal(z, H0=70., Om=0.3, Ode=0.7, w0=-1., wa=0.0):
    """
    Function to grab cosmological values

    Parameters
    ----------
    z : list(float)
        Redshifts.
    H0 : float, optional
        Hubble constant. The default is 70..
    Om : float, optional
        Omega dark matter. The default is 0.3.
    Ode : float, optional
        Omega dark energy. The default is 0.7.
    w0 : float, optional
        dark energy equation-of-state parameter. The default is -1..
    wa : float, optional
        dark energy equation-of-state parameter. The default is 0.0.

    Returns
    -------
    df : pandas df
        Data.

    """

    cosmology = w0waCDM(H0=H0,
                        Om0=Om,
                        Ode0=Ode,
                        w0=w0, wa=wa)

    dL = cosmology.luminosity_distance(z).value*1.e6

    df = pd.DataFrame(dL, columns=['dL'])
    df['z'] = z
    df['mu'] = 5.*np.log10(df['dL']/1.e6)+25.
    df['age_universe'] = cosmology.age(z).value
    df['w'] = w0
    df['Om'] = Om
    df['Ode'] = Ode

    return df


def plot_cosmo(z, Om, Ode, w0, fig=None, ax=None,
               draw_age=False, ls='solid', color='k', legend=''):
    """
    Function to plot cosmo scenarios

    Parameters
    ----------
    z : list(float)
        Redshifts.
    Om : float
        Om parameter.
    Ode : float
        Omega DE.
    w0 : float
        W0 DE equation of state.
    fig : matplotlib fig, optional
        Figure where to draw. The default is None.
    ax : matplotlib axis, optional
        axis where to draw. The default is None.
    draw_age : bool, optional
        To plot the age of the Universe. The default is False.
    ls : str, optional
        matplotlib linestyle. The default is 'solid'.
    legend : str, optional
        Legend for the plot. The default is ''.

    Returns
    -------
    None.

    """

    if fig == None:
        fig, ax = plt.subplot(figsize=(14, 8))

    df_cosmo = cosmoVal(z, Om=Om, Ode=Ode, w0=w0)

    label = None
    if legend != '':
        label = legend
    ax.plot(df_cosmo['z'], df_cosmo['mu'],
            marker='None', color=color, linestyle=ls, lw=2, label=label)

    if draw_age:
        axb = ax.twiny()
        axb.plot(df_cosmo['age_universe'], df_cosmo['mu'],
                 color='white', alpha=0.)
        axb.invert_xaxis()
        axb.set_xlabel('Age of the Universe [Gy]')

    ax.set_xlabel('z')
    ax.set_ylabel('Distance modulus')
    # ax.legend(bbox_to_anchor=(0.8, 0.9), ncol=3, fontsize=12, frameon=False)
    ax.legend(frameon=False)


def movie_HD(data):
    """
    Function to make a set of HDs
    which are saved as png and can be converted as a movie.

    Parameters
    ----------
    data : pandas df
        Data to process.

    Returns
    -------
    None.

    """

    outDir = 'plot_HD_SN'
    import os
    cmd = 'mkdir -p {}'.format(outDir)
    os.system(cmd)

    print(data.columns)
    mjd_min = data['daymax'].min()

    data['night'] = data['daymax']-mjd_min+1
    data['night'] = data['night'].astype(int)

    data = data.sort_values(by=['night'])
    nights = data['night'].unique()

    for i, night in enumerate(nights):
        outName = '{}/HD_{}.png'.format(outDir, i)
        figtitle = 'night {}'.format(night)
        fig, ax = plt.subplots(figsize=(14, 8))
        idx = data['night'] <= night
        plot_HD(data[idx], '', '', color='grey', fig=fig, ax=ax)
        idxb = data['night'] == night
        plot_HD(data[idxb], '', figtitle, color='k', fig=fig, ax=ax)

        ax.grid()
        plt.savefig(outName)
        plt.close(fig)


def plot_HD(data, outName, figtitle='', color='k', fig=None, ax=None):
    """
    function to plot HD

    Parameters
    ----------
    data : pandas df
        Data to plot.
    outName : str
        Output file name.
    figtitle : str, optional
        fig title. The default is ''.
    color : str, optional
        plot color. The default is 'k'.
    fig : matplotlib figure, optional
        Figure where to plot. The default is None.
    ax : matplotlib axis, optional
        axis where to plot. The default is None.

    Returns
    -------
    None.

    """

    if fig is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    if figtitle != '':
        fig.suptitle(figtitle)

    zmin = 0.01
    zmax = 1.1
    mu_min = 34.
    mu_max = 45.

    ax.errorbar(data['z'], data['mu'], yerr=data['sigma_mu'],
                color=color, marker='.', linestyle='None')

    ax.set_xlim([zmin, zmax])
    ax.set_ylim([mu_min, mu_max])

    ax.grid()
    ax.set_xlabel('z')
    ax.set_ylabel('Distance modulus')

    if outName != '':
        plt.savefig(outName)
        plt.close(fig)


def plot_age_universe():
    """
    Function to plot the age of the universe depending on cosmo configs.

    Returns
    -------
    None.

    """

    Oms = np.arange(0., 1.01, 0.01)
    w0s = [-1.0, -0.666, -10, 0.0]
    config = ['flat Universe with $\Lambda$ (w=-1)',
              'flat Universe with $\Lambda$ (w=-2/3)',
              'flat Universe with $\Lambda$ (w=-10)',
              'open Universe without $\Lambda$']

    df = pd.DataFrame()

    for Om in Oms:
        for io, w0 in enumerate(w0s):
            Ode = 1.-Om
            if w0 >= 0.:
                Ode = 0.

            dfc = cosmoVal([0.0000001], H0=70., Om=Om, Ode=Ode, w0=w0, wa=0.0)
            dfc['config'] = config[io]
            df = pd.concat((df, dfc))

    fig, ax = plt.subplots(figsize=(14, 8))

    ido = df['age_universe'] <= 15.
    df = df[ido]

    ws = df['w'].unique()
    ls = ['solid', 'dashed', 'dotted', 'dashdot']
    color = ['red', 'k', 'b', 'm']

    # for i, w0 in enumerate(ws):
    for i, conf in enumerate(df['config'].unique()):
        idx = df['config'] == conf
        sel = df[idx]
        ax.plot(sel['Om'], sel['age_universe'],
                color=color[i], linestyle=ls[i], label='{}'.format(conf))

    xlims = [0, 1]
    ax.set_xlim(xlims)
    ax.set_ylabel('Age of the Universe [Gyr]')
    ax.set_xlabel('$\Omega_m^{0}$')
    ax.grid()
    ax.legend(frameon=False)

    age_stellar_min = 11
    age_stellar_max = 12.7

    ax.fill_between([0, 1, 1, 0], [age_stellar_min] *
                    2+[age_stellar_max]*2, color='yellow', alpha=0.5)
    ax.set_ylim([9, 15])

    ax.text(0.7, 11., 'oldest stellar ages', rotation=-30.)

    plt.show()


def plot_HD_with_cosmo(df):
    """
    Function to plot HD with various cosmo scenarios.

    Parameters
    ----------
    df : pandas df
        Data to process.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(14, 8))

    """
    ax.errorbar(df['z'], df['mu'], yerr=df['sigma_mu'],
                marker='.', color='k', linestyle='None')
    """
    z = np.arange(0.01, 1.12, 0.005)

    dfb = bin_it_combi(df, bins=z)

    ax.errorbar(dfb['z'], dfb['mu'], yerr=dfb['sigma_mu'],
                marker='.', color='k', linestyle='None', mfc='None')

    # get cosmology here
    zb = np.arange(0.01, 1.1, 0.001)
    w0s = [0., -1., -10.]
    Oms = [0.3, 0.3, 1, 5.]
    Odes = [0.7, 0., 0., 0.]
    w0s = [-1]*len(Odes)
    # ls = dict(zip(w0s, ['dotted', 'solid', 'dashed']))

    ls = ['solid', 'dotted', 'dashed', 'dashdot']
    color = ['red', 'blue', 'green', 'orange']

    olam = '\Omega_{\Lambda}'
    # w0s = [-1]
    for i, w0 in enumerate(w0s):
        Om = Oms[i]
        Ode = Odes[i]
        # legend = '($\Omega_m={},{}={},w={}$)'.format(Om, olam, Ode, w0)
        legend = '($\Omega_m={},{}={}$)'.format(Om, olam, Ode)
        plot_cosmo(zb, Om=Om, Ode=Ode, w0=w0, fig=fig,
                   ax=ax, ls=ls[i], color=color[i], legend=legend, draw_age=0)
    ax.grid()

    plt.show()


fName = 'SN_baseline_v3.0_10yrs_baseline_v3.0_10yrs_0.hdf5'

df = pd.read_hdf(fName)

# movie_HD(df)

# print(test)

# plot_HD_with_cosmo(df)

plot_age_universe()
