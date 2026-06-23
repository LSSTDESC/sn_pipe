#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:49:55 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
#import matplotlib.pyplot as plt
from sn_plotter_metrics import plt,filtercolors
import numpy as np

def plot_season_length(data,vary='season_length', 
                       labely='season length [nights]'):
    """
    Function to plot season length vs season for each DDF

    Parameters
    ----------
    data : pandas df
        Data to plot.
    vary : str, optional
        var to plot. The default is 'season_length'.

    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots(figsize=(12, 8))

    fig.subplots_adjust(top=0.85)

    fields = data['field'].unique()
    ls = ['solid', 'dotted', 'dashed', 'dashdot']*2
    marker = ['.', 's', 'o', '^', 'P', 'h']
    colors = ['k', 'r', 'b', 'm', 'g', 'c']

    for io, field in enumerate(np.unique(data['field'])):
        
        idx = data['field'] == field
        
        sel = data[idx]
        
        ax.plot(sel['season'],sel[vary],
                linestyle=ls[io],marker=marker[io],color=colors[io],
                mfc='None',label=corresp[field])
       
    ax.grid(visible=True)
    ax.legend(bbox_to_anchor=(0.5, 1.17), ncol=3,
                      frameon=False, loc='upper center')
    ax.set_ylabel(r'{}'.format(labely))
    ax.set_xlabel(r'season')

def plot_nvisits(data,field='DD:COSMOS'):
    """
    Function to plot the number of visits per band/season

    Parameters
    ----------
    data : pandas df
        Data to process.
    field : str, optional
        Field to plot. The default is 'DD:COSMOS'.

    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(corresp[field])
    fig.subplots_adjust(top=0.85)

    
    ls = ['solid', 'dotted', 'dashed', 'dashdot']*2
    marker = ['.', 's', 'o', '^', 'P', 'h']

    idx = data['field'] == field

    
    sel = pd.DataFrame(data[idx])
    
    bands = 'ugrizy'
    
    for io,b in enumerate(bands):
        vara = '{}_med'.format(b)
        varb = '{}_med_ud'.format(b)
        vary = '{}_med_all'.format(b)
        sel[vary] = sel[vara]
        
        idx = sel[varb] > 0
        sel.loc[idx,vary] = sel[varb]
        
        ax.plot(sel['season'],sel[vary],
                linestyle=ls[io],marker=marker[io],color=filtercolors[b],
                mfc='None',label='$'+b+'$')
       
    ax.grid(visible=True)
    ax.legend(bbox_to_anchor=(0.5, 1.1), ncol=6,
                      frameon=False, loc='upper center')
    ax.set_ylabel(r'$N_{visits}$')
    ax.set_xlabel(r'season')
    
parser = OptionParser(
    description='Script to analyse DDF visits on a nightly basis from pointings')

parser.add_option("--dbDir", type="str",
                  default='../ddf_visits_night_ud',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='baseline_v5.3.0_10yrs',
                  help="OS name [%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName


# load the data

fName = '{}/{}/ddf_visits_season_ud.hdf5'.format(dbDir, dbName)
data = pd.read_hdf(fName)

idx = data['season'] < 11
data = data[idx]

cola = ['DD:COSMOS','DD:XMM_LSS','DD:ECDFS','DD:ELAISS1','DD:EDFS_a','DD:EDFS_b']
colb = ['COSMOS','XMM-LSS','CDFS','ELAISS1','EDFS_a','EDFS_b']

corresp = dict(zip(cola,colb))


plot_season_length(data)

plot_season_length(data,'season_length_ud')

plot_season_length(data,'deltat_beg_survey',
                   labely='$\Delta t = t_{start}^{UD}-t_{start}^{season}$ [nights]')

plot_season_length(data,'deltat_end_survey',
                   labely='$\Delta t = t_{end}^{UD}-t_{end}^{season}$ [nights]')

plot_nvisits(data)

plt.show()
    


