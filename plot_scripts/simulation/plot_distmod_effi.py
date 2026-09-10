#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:23:19 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sn_tools.sn_obs import load_season
from sn_analysis.sn_calc_plot import bin_it_mean,bin_it_sum
from scipy.interpolate import interp1d

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
        sel = pd.DataFrame(res[idx])
        
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
    ax.grid(visible=True)

def plot_season_correl(dres,seasons,xvar='diff_mu',
                       yvar='effi'):
    
    
    for seas in seasons:
        fig, ax = plt.subplots()
        
        for key, res in dres.items():
            idx = res['season'] == seas
            sel = res[idx]
        
            plot_configs_correl(sel,seas,xvar,yvar,fig=fig,ax=ax)

def plot_configs_correl(res,season,
                       xvar='diff_mu_mean',yvar='effi',
                       fig=None,ax=None):
    
    if fig is None:
        fig, ax = plt.subplots()

    fig.suptitle('season {}'.format(season))

    confs = res['config'].unique()

    bins = np.arange(0.0,1.1,0.1)
    for conf in confs:
        idx = res['config'] == conf
        sel = pd.DataFrame(res[idx])
        
        ax.plot(sel[xvar],sel[yvar],label=conf)
    
    ax.legend()
    ax.grid(visible=True)
    
    
def get_zlim(grp):
    
    grp = grp.sort_values(by=['z'])
    
    
    idx = grp['z']>=0.1
    idx &= grp['z'] < 0.5
    sel = grp[idx]
    
    ratio_min = sel['effi'].min()/100.
    ratio_max = sel['effi'].max()/100.
    ratio_mean = 0.5*(ratio_min+ratio_max)
    
    ro = sel['nsn_obs'].sum()/sel['nsn_exp'].sum()
    ro = sel['effi'].mean()/100.
    
    for vv in [ratio_min,ratio_max,ratio_mean]:
        norm_factor = 1./vv
        calc_zlim(grp.copy(),norm_factor)
    
 
    
def calc_zlim(grpa,norm_factor,vref=[95.,98.]):
    
    idx = grpa['z']>=0.1
    idx &= grpa['z'] < 0.5
    sela = grpa[idx]
    
    print('allo',norm_factor,sela['effi'].min(),sela['effi'].max())
    
    grp = pd.DataFrame(grpa)
    grp['effi'] *= norm_factor
    grp['nsn_exp'] *= norm_factor
    
    idx = grp['z']>=0.1
    idx &= grp['z'] < 0.5
    sel = grp[idx]
    
    print('allo',norm_factor,sel['effi'].min(),sel['effi'].max())
    grp = calc_var(grp)
    
    
    effi_z = interp1d(grp['effi'],grp['z'],bounds_error=False, fill_value=0.)
    rat_z = interp1d(grp['z'],grp['nsn_ratio'],bounds_error=False, fill_value=0.)
    
    for vv in vref:
        zlim = effi_z(vv)
        frac = rat_z(zlim)
    
        print(norm_factor,vv,zlim,frac)
    
    show_me(grp)
    
def calc_var(df):
    
    df['nsn_exp_sum'] = np.cumsum(df['nsn_exp'])
    df['nsn_obs_sum'] = np.cumsum(df['nsn_obs'])
    
    df['nsn_ratio'] =1.-df['nsn_obs_sum']/ df['nsn_exp_sum']
    df['nsn_exp_sum'] /= df['nsn_exp_sum'].max()
    df['nsn_obs_sum'] /= df['nsn_obs_sum'].max()    
    
    
    return df
    
def show_me(grp):
    
   fig, ax = plt.subplots()
   
   ax.errorbar(grp['z'],grp['effi'],yerr=grp['effi_err'])
   
   axb = ax.twinx()
   """
   axb.plot(grp['z'],grp['nsn_exp_sum'])
   axb.plot(grp['z'],grp['nsn_obs_sum'])
   """
   axb.plot(grp['z'],grp['nsn_ratio'])
   
   ax.grid(visible=True)
   plt.show()    
    
    
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
    
    
print(res[0].columns)    
    
"""
plot_season(res,seasons,yvar='diff_mu',yvar_err='diff_mu_std')

plot_season(res,seasons,yvar='nsn')


plot_season(res,seasons,yvar='effi',yvar_err='effi_err')
    
plot_season_correl(res,seasons,xvar='diff_mu_std',yvar='effi')

"""
cols = ['healpixID','pixRA','pixDec','config','season']

for key, vals in res.items():
    dd = vals.groupby(cols).apply(lambda x: get_zlim(x))    



plt.show()

