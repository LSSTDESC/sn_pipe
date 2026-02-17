#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:46:58 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_analysis.sn_tools import load_data, complete_df, pull_it
from sn_analysis.sn_fit_tools import fit_hist,gauss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_pull_hists(data, fig=None, ax=None, figtit='',fitgauss=True):
    """
    Function to plot (and fit) pulls

    Parameters
    ----------
    data : pandas df
        Data to plot.
    fig : matplotlib figure, optional
        Figure for the plot. The default is None.
    ax : matplotlib axis, optional
        Axis for the plot. The default is None.
    figtit : str, optional
        Figure title. The default is ''.
    fitgauss : bool, optional
        To fit the pulls (gauss). The default is True.

    Returns
    -------
    None.

    """

    if fig is None:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    if figtit != '':
        fig.suptitle(figtit)

    ipos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    bins = np.arange(-5, 5, 0.5)
    bins='auto'
    fit_with_errors = False
    
    for i, vv in enumerate(['x1', 'color', 'mb', 'daymax']):
        pp = ipos[i]
        xp = pp[0]
        yp = pp[1]
        pullvar = 'pull_{}'.format(vv)
        axa = ax[xp,yp]
        plot_pull_hist(data,vv,pullvar,fig=fig,ax=axa,bins=bins,
                       fitgauss=fitgauss,fit_with_errors=fit_with_errors)


def plot_pull_hist(data,varx,pullvar,
                   fig=None,ax=None,figtit='',
                   bins='auto',fitgauss=True,fit_with_errors=False):
    """
    Function to plot (and fit) a single pull histo

    Parameters
    ----------
    data : pandas df
        Data to plot.
    varx : str
        variable to plot.
    pullvar : str
        pull variable name.
    fig : matplotlib figure, optional
        figure for the plot. The default is None
    ax : matplotlib axis, optional
        axis for the plot. The default is None
    bins : str,array(float), or int, optional
        bins for histogram. The default is 'auto'.
    fitgauss : bool, optional
        to fit the distribution or not. The default is True.
    fit_with_errors : bool, optional
        to include bin errors in the fit. The default is False.

    Returns
    -------
    None.

    """
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(12,8))
        
    if figtit != '':
        fig.suptitle(figtit)
    
    idx = np.abs(data[pullvar]) <= 5.
    sel = data[idx]
    ax.hist(sel[pullvar],bins=bins, histtype='step')
    ax.set_ylabel('Number of Entries',fontweight='bold',fontsize=15)
    ax.set_xlabel(r'{}'.format(varx),fontweight='bold',fontsize=15)
    
    # Get the fitted curve
    if fitgauss:
        coeff,err_coeff,chi_square,ndof = fit_hist(sel, pullvar,bins=bins,
                                                   fit_with_errors=fit_with_errors)
        xmin = sel[pullvar].min()
        xmax = sel[pullvar].max()
        newbins = np.arange(xmin, xmax, 0.01)
        hist_fit = gauss(newbins, *coeff)
        mean = np.round(coeff[1], 2)
        sigma = np.round(coeff[2], 2)
        leg = 'pull= {} +- {}'.format(mean, sigma)
        ax.plot(newbins, hist_fit, label=leg)
        leg_str = '$\mu$='+'{}'.format(np.round(coeff[1],1))
        leg_str +='$\pm$'+'{}'.format(np.round(err_coeff[1],1))
        leg_str += '\n $\sigma$='+'{}'.format(np.round(coeff[2],1))
        leg_str +='$\pm$'+'{}'.format(np.round(err_coeff[2],1))
        ax.text(0.65,0.8,leg_str,transform=ax.transAxes,fontsize=15)

    ax.grid(visible=True)
    
def plot_pull_vs(data, fig=None, ax=None, figtit='',varx='z'):
    """
    Function to plot a set of pull var vs varx

    Parameters
    ----------
    data : pandas df
        Data to process.
    fig : matplotlib figure, optional
        Figure for the plot. The default is None.
    ax : matplotlib axis, optional
        axis for the plot. The default is None.
    figtit : str, optional
        Figure title. The default is ''.
    varx : str, optional
        x-axis variable. The default is 'z'.

    Returns
    -------
    None.

    """

    if fig is None:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    if figtit != '':
        fig.suptitle(figtit)

    ipos = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i, vv in enumerate(['x1', 'color', 'mb', 'daymax']):
        pp = ipos[i]
        axa = ax[pp[0], pp[1]]
        vary = 'pull_{}'.format(vv)
        plot_pull_vs_indiv(data,'z','z',vary,vary,fig=fig,ax=axa)
        

def plot_pull_vs_indiv(data,varx,legx,
                       pullvar,legy,fig=None,ax=None,figtit='',binIt=True):
    """
    Function to plot pull vs varx plus binned data

    Parameters
    ----------
    data : pandas df
        Data to plot.
    varx : str
        x-axis variable.
    legx : str
        x-axis label.
    pullvar : str
        pull variable.
    legy : str
        y-axis label.
    fig : matplotlib figure, optional
        plot figure. The default is None.
    ax : matplotlib axis, optional
        plot axis. The default is None.
    figtit : str, optional
        figure title. The default is ''.
    binIt : bool, optional
        to plot binned data. The default is True.

    Returns
    -------
    None.

    """

    if fig is None:
          fig, ax = plt.subplots(figsize=(12,8))
            
    if figtit != '':
        fig.suptitle(figtit)
    
    idx = np.abs(data[pullvar]) <= 5.
    sel = data[idx]
    
    if binIt:
        from sn_analysis.sn_calc_plot import bin_it_mean
        tt = bin_it_mean(sel,
                         xvar=varx,yvar=pullvar,
                         bins=np.arange(0.01, 1.1, 0.05))
        yerr ='{}_std'.format(pullvar)
        ax.errorbar(tt[varx],tt[pullvar],yerr=tt[yerr],color='r')
    
    ax.plot(sel[varx], sel[pullvar], 
            color='k',marker='o',mfc='None',
            markersize=5,linestyle='None')
    ax.set_ylabel(r'{}'.format(legy),fontweight='bold',fontsize=15)
    ax.set_xlabel(r'{}'.format(legx),fontweight='bold',fontsize=15)
    ax.grid(visible=True)
    
parser = OptionParser(description='Script to plot SN parameter pulls')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_newatm_lc_coadd_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v5.0.0_10yrs',
                  help='OS name [%default]')
parser.add_option('--runType', type=str,
                  default='DDF_spectroz',
                  help='run type [%default]')
parser.add_option('--timescale', type=str,
                  default='season',
                  help='timescale [%default]')
parser.add_option('--field', type=str,
                  default='COSMOS',
                  help='field to display [%default]')
parser.add_option('--healpixID', type=int,
                  default=108958,
                  help='pixel to display [%default]')
parser.add_option('--plots', type=str,
                  default='pull_hist,pull_vs_z',
                  help='plots to show [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
runType = opts.runType
timescale = opts.timescale
field = opts.field
healpixID = opts.healpixID
plots = opts.plots.split(',')

# load data

df = load_data(dbDir, dbName, runType)

# complete data
df = complete_df(df)
df['SNR_red'] = df['SNR_i']+df['SNR_z']+df['SNR_y']

idx = df['fitstatus'] == 'fitok'

df = df[idx]

# estimate pull
df = pull_it(df)

idx = df['field'] == field
idx &= df['healpixID'] == healpixID

sel = df[idx]
sel = sel.sort_values(by=['season'])
seasons = sel['season'].unique()

ccols = ['healpixID', 'z', 'x1', 'color', 'daymax', 'x0', 'season', 'epsilon_x0',
         'epsilon_x1', 'epsilon_color', 'epsilon_daymax', 'SNID',
         'minRFphase', 'minRFphaseQual', 'maxRFphase', 'maxRFphaseQual']

for seas in seasons:
    figtit = '{} - pixel {} \n season {}'.format(field,healpixID,seas)
    idxb = sel['season'] == seas
    selb = sel[idxb]
    if 'pull_hist' in plots:
        plot_pull_hists(selb, figtit=figtit)
    if 'pull_vs_z' in plots:
        plot_pull_vs(selb, figtit=figtit,varx='z')
    plt.show()
    """
    ido = selb['pull_color'] < -4.
    seld = selb[ido]
    if len(seld) >= 1:
        snids = seld['SNID'].to_list()
        # select df data
        idf = df['SNID'].isin(snids)
        seldf = df[idf]
        seldf[ccols].to_hdf('simuparams_COSMOS.hdf5', key='simuparams')
    """