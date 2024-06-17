#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:20:30 2024

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def load_data(dbDir, dbName, runType, timescale, seasons):
    """
    Function to load data

    Parameters
    ----------
    dbDir : str
        Data dir.
    dbName : str
        dbname.
    runType : str
        run type.
    timescale : str
        time scale.
    seasons : list(int)
        List of seasons.

    Returns
    -------
    df : pandas df
        Data for plot.

    """

    mainDir = '{}/{}/{}'.format(dbDir, dbName, runType)

    df = pd.DataFrame()
    for seas in seasons:
        path = '{}/*_{}_{}.hdf5'.format(mainDir, timescale, seas)
        fis = glob.glob(path)
        for fi in fis:
            print('loading', fi)
            tt = pd.read_hdf(fi)
            df = pd.concat((df, tt))
            # pull estimation
            df['pull_x1'] = (df['x1']-df['x1_fit'])/df['sigmax1']
            df['pull_c'] = (df['color']-df['color_fit'])/df['sigma_c']
            df['pull_daymax'] = (df['daymax']-df['t0_fit'])/df['sigma_t0']
            df['diff_x1'] = (df['x1']-df['x1_fit'])/df['x1']
            df['diff_c'] = (df['color']-df['color_fit'])/df['color']

    return df


def gauss(x, *p):
    """
    gaussian function 

    Parameters
    ----------
    x : float
        x values.
    *p : list(float)
        gaussian parameters.

    Returns
    -------
    list(float)
        function values.

    """
    A, mu, sigma = p
    return A/np.sqrt(sigma)*np.exp(-(x-mu)**2/(2.*sigma**2))


def plot_pull(dfa, pullvar, figtitle='', fitgauss=True):
    """
    Function to plot and fit of a pull distribution

    Parameters
    ----------
    dfa : pandas df
        Data container.
    pullvar : str
        pull variable.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots()
    fig.suptitle(figtitle)
    # ax.hist(dfa['pull_x1'], histtype='step')
    idx = np.abs(dfa[pullvar]) <= 5
    sel = dfa[idx]
    print(len(sel)/len(dfa))
    ax.hist(sel[pullvar], histtype='step', bins=80)

    # Get the fitted curve
    if fitgauss:
        coeff = fit_pull(sel, pullvar)
        xmin = sel[pullvar].min()
        xmax = sel[pullvar].max()
        newbins = np.arange(xmin, xmax, 0.01)
        hist_fit = gauss(newbins, *coeff)
        mean = np.round(coeff[1], 2)
        sigma = np.round(coeff[2], 2)
        leg = 'pull= {} +- {}'.format(mean, sigma)
        ax.plot(newbins, hist_fit, label=leg)
        print('bbb', coeff[0], coeff[1], coeff[2])
    print(figtitle, np.mean(sel[pullvar]))
    """
    ttb = gauss(bin_centres, *coeff)
    print('chi2', np.sum(np.power(hist-ttb, 2))/(len(bin_centres)-3))
    print(hist)
    print(ttb)
    print(hist-ttb)
    """
    ax.legend()


def fit_pull(sel, pullvar):

    hist, bins = np.histogram(sel[pullvar], bins=80)
    bin_centres = (bins[:-1] + bins[1:])/2
    p0 = [1., 0., 1.]

    coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)

    return coeff


def plot_hist(dfa, var, figtit='', fig=None, ax=None, label='', bins=10):
    """
    Function to plot hist

    Parameters
    ----------
    dfa : pandas df
        Data container.
    var : str
        Variable to plot.

    Returns
    -------
    None.

    """

    if fig is None:
        fig, ax = plt.subplots()
    if figtit != '':
        fig.suptitle(figtit)
    ax.hist(dfa[var], histtype='step', bins=bins, label=label)


def plot_nsn_hist(dfa):
    ccols = ['healpixID', 'RA', 'Dec']
    print(np.unique(dfa['healpixID']))
    dfb = dfa.groupby(['healpixID']).size().reset_index(name='nsn')
    print(dfb)
    plt.hist(dfb['nsn'], histtype='step')


parser = OptionParser(description='Script to analyze SN prod')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v3.4_10yrs',
                  help='OS name [%default]')
parser.add_option('--runType', type=str,
                  default='WFD_spectroz_nosat',
                  help='run type [%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='timescale [%default]')
parser.add_option('--seasons', type=str,
                  default='1',
                  help='seasons/years to process [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName.split(',')
runType = opts.runType
timescale = opts.timescale
seasons = opts.seasons.split(',')

dfa = pd.DataFrame()
print('kkkkkk', dbName)
for dbNam in dbName:
    dff = load_data(dbDir, dbNam, runType, timescale, seasons)
    dff['dbName'] = dbNam
    dfa = pd.concat((dfa, dff))


ninit = len(dfa)

idx = dfa['sigma_c'] <= 0.04
# idx &= dfa['Nfilt_10'] > 2
idx &= dfa['n_epochs_m10_p5'] >= 5
idx &= dfa['n_epochs_phase_minus_10'] >= 2
idx &= dfa['n_epochs_bef'] >= 5
idx &= dfa['n_epochs_aft'] >= 10
dfa = dfa[idx]


print(dfa.columns, len(dfa)/30.)


plot_pull(dfa, 'pull_x1', figtitle='pull x1')
plot_pull(dfa, 'pull_c', figtitle='pull color')
"""
plot_pull(dfa, 'diff_x1', figtitle='diff x1', fitgauss=False)
plot_pull(dfa, 'diff_c', figtitle='diff color', fitgauss=False)
"""
# plot_pull(dfa, 'pull_daymax')
"""
ccols = ['n_epochs_bef', 'n_epochs_aft', 'Nfilt_10', 'Nfilt_15', 'Nfilt_20',
         'n_epochs_phase_minus_10',
         'n_epochs_phase_plus_20', 'n_epochs_m10_p35', 'n_epochs_m10_p5',
         'n_epochs_p5_p20', 'n_bands_m8_p10']
# ccols = ['sigmax1', 'sigmaC', 'sigmat0']
for vv in ccols:
    fig, ax = plt.subplots()
    for dbNam in dbName:
        idx = dfa['dbName'] == dbNam
        sel = dfa[idx]
        plot_hist(sel, vv, figtit=vv, fig=fig, ax=ax, label=dbNam, bins=15)
    ax.legend()
"""
plt.show()
