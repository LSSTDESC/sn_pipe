#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:23:19 2026

@author: philippe.gris@clermont.in2p3.fr
"""
import glob
from optparse import OptionParser
import pandas as pd
from sn_analysis.sn_selection import selection_criteria
import numpy as np
import matplotlib.pyplot as plt
from sn_tools.sn_io import checkDir
from sn_tools.sn_utils import clean_level
from sn_tools.sn_obs import load_season
from sn_analysis.sn_calc_plot import bin_it_mean,bin_it_sum

def plot_season(dres,seasons,xvar='z',yvar='diff_mu_mean',
                yvar_err='',binIt=False):
    """
    Function to display seasons

    Parameters
    ----------
    dres : pandas df
        Data to plot.
    seasons : list(int)
        List of seasons.
    xvar : str, optional
        x-axis variable. The default is 'z'.
    yvar : str, optional
        y-axis variable. The default is 'diff_mu_mean'.
    yvar_err : str, optional
        y-axis err variable. The default is ''.
    binIt : bool, optional
        to bin results or not. The default is False.

    Returns
    -------
    None.

    """
    
    for seas in seasons:
        fig, ax = plt.subplots()
        
        for key, res in dres.items():
            idx = res['season'] == seas
            sel = res[idx]
        
            plot_configs(sel,seas,xvar,yvar,yvar_err=yvar_err,fig=fig,ax=ax)

def plot_configs(res,season,
                 xvar='z',yvar='diff_mu_mean',yvar_err='',
                 fig=None,ax=None,binIt=False):
    """
    Function to plot configs per season

    Parameters
    ----------
    res : pandas df
        Data to plot.
    season : int
        season.
    xvar : str, optional
        x-axis variable. The default is 'z'.
    yvar : str, optional
        y-axis variable. The default is 'diff_mu_mean'.
    yvar_err : str, optional
        y-axis err variable. The default is ''.
    fig : matplotlib figure, optional
        figure for the plot. The default is None.
    ax : matplotlib axis, optional
        y-axis for the plot. The default is None.
    binIt : bool, optional
        To show binned results. The default is False.

    Returns
    -------
    None.

    """
    
    
    if fig is None:
        fig, ax = plt.subplots()

    fig.suptitle('season {}'.format(season))

    confs = res['config'].unique()

    bins = np.arange(0.0,1.1,0.1)
    for conf in confs:
        idx = res['config'] == conf
        sel = res[idx]
        if yvar == 'effi':
            sel['effi'] *= 100
            sel['effi_err'] *= 100
        
        if binIt:
            if yvar != 'nsn':
                sel = bin_it_mean(sel, xvar=xvar,yvar=yvar,bins=bins)
            else:
                sel = bin_it_sum(sel, xvar=xvar,yvar=yvar,bins=bins)
        yerr = None
        if yvar_err != '':
            yerr = sel[yvar_err]
        
        ax.errorbar(sel[xvar],sel[yvar],yerr=yerr,label=conf)
    
    ax.legend()



parser = OptionParser('script to plot dist mod, ...')

parser.add_option('--files', type=str, default='comp_distmod.hdf5,comp_distmod_sigmaC.hdf5',
                  help='files to analyze [%default]')
parser.add_option('--seasons', type=str, default='1-10',
                  help='seasons to show [%default]')

opts, args = parser.parse_args()

files = opts.files
seasons = load_season(opts.seasons)

fis = files.split(',')

#load data
res = {}
for i,fi in enumerate(fis):
    res[i] = pd.read_hdf(fi)
    
        

plot_season(res,seasons)
"""
plot_season(res,seasons,yvar='nsn')
"""

#plot_season(res,seasons,yvar='effi',yvar_err='effi_err')
    
plt.show()

