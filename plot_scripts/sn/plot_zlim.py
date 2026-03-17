#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:15:51 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
from sn_plotter_analysis import plt

def plot_zlim(df, yvar='zlim_0.98',
              yleg='$z_{complete}^{0.98}$',
              field='COSMOS',fig=None,ax=None,
              color='k',marker='o'):
    """
    Function to plot zlim vs season

    Parameters
    ----------
    df : pandas df
        Data to plot.
    yvar : str, optional
        y-axis variable. The default is 'zlim_0.98'.
    yleg : str, optional
        y-axis legend. The default is '$z_{complete}^{0.98}$'.
    field : str, optional
        Field name. The default is 'COSMOS'.
    fig : matplotlib figure, optional
        Figure for the plot. The default is None.
    ax : matplotlib axis, optional
        Axis for the plot. The default is None.

    Returns
    -------
    None.

    """
    
    idx = df['field'] == field
    sel = df[idx]
    
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(12,8))
    
    ax.plot(sel['season'],sel[yvar],
            marker=marker,mfc='None',markersize=15,
            color=color,label=field)
    
    ax.grid(visible=True)
    ax.set_xlabel(r'season')
    ax.set_ylabel(r'{}'.format(yleg))
    ax.legend()

parser = OptionParser(description='Script to plot z lim values')

parser.add_option('--dbDir', type=str,
                  default='../zlim',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v5.0.0_10yrs',
                  help='OS name [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFS_a,EDFS_b',
                  help='fields to process [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
fields = opts.fields.split(',')

fName = '{}/zlim_{}.hdf5'.format(dbDir,dbName)

df = pd.read_hdf(fName)

idx = df['season'] < 11
df = df[idx]

print(df[['field','season','zlim_0.98','zlim_0.95']])
df['zlim_0.98'] = df['zlim_0.98'].astype(float)
df['zlim_0.95'] = df['zlim_0.95'].astype(float)
df = df.dropna()
dfb = df.groupby(['field','season'])[['zlim_0.98','zlim_0.95']].mean().reset_index()
print(dfb)

fig, ax = plt.subplots(figsize=(12,8))

markers=['s','o','P','h','v','^']
colors = ['r','b','green','orange','violet','violet']
mmarkers = dict(zip(fields,markers))
ccolors = dict(zip(fields,colors))

for field in fields:
    plot_zlim(dfb,field=field,fig=fig,ax=ax,
              color=ccolors[field],marker=mmarkers[field])

plt.show()