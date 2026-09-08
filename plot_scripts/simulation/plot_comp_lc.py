#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 09:51:26 2026

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
import glob
from sn_plotter_analysis import plt

def load_data(theDir):
    """
    Functio to load the data

    Parameters
    ----------
    theDir : str
        Data dir.

    Returns
    -------
    df : pandas df
        Data.

    """
    
    fis = glob.glob('{}/*.hdf5'.format(theDir))
    
    df = pd.DataFrame()
    
    for fi in fis:
        dfa = pd.read_hdf(fi)
        df = pd.concat((df,dfa))
    
    print(df['season'].unique())
    
    return df

def plot_season(df,season):
    """
    Function to plot a season

    Parameters
    ----------
    df : pandas df
        Data to plot.
    season : int
        season to plot.

    Returns
    -------
    None.

    """
    
    bands = df['filter'].unique()

    print(df.columns)

    for b in bands:
        idx = df['filter'] == b
        sel = df[idx]
        sel = sel.sort_values(by=['config_test'])
        confs = sel['config_test'].unique()
        
        fig, ax = plt.subplots(figsize=(12,8))
        fig.suptitle('{} band - season {}'.format(b,season))
        
        for conf in confs:
            idxb = sel['config_test'] == conf
            selb = sel[idxb]
            selb = selb.sort_values(by=['z'])
            ax.plot(selb['z'],100.*selb['sigma'],label=conf)
            
        ax.legend()
        ax.grid(visible=True)
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$\sigma^{LC\/ error\/ ratio}$ [%]')

theDir = '../comp_lc_fit_coadd'


df = load_data(theDir)

seasons = df['season'].unique()

for seas in seasons:
    idx = df['season'] == seas
    sel = df[idx]
    plot_season(sel,seas)


plt.show()
    
    

