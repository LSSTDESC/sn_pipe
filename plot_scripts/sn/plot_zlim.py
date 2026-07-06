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

def get_dist(grp,distval='dist_center',yvar='zlim_0.98'):
    
    from sn_plotter_metrics.utils import get_dist
    import numpy as np
    df_dist = get_dist(grp)
    
    xmin, xmax = df_dist[distval].min(), df_dist[distval].max()
    bins = np.linspace(xmin-1.e-6, xmax, 12)
    # bins = np.arange(0.1, 2.22, 0.22)
    group = df_dist.groupby(pd.cut(df_dist[distval], bins), observed=False)
    plot_centers = (bins[:-1] + bins[1:])/2
    plot_values = group[yvar].mean()
    dd = pd.DataFrame(plot_centers, columns=[distval])

    dd[yvar] = plot_values.to_list()
    
    return dd

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



markers=['s','o','P','h','v','^']
colors = ['r','b','green','orange','violet','violet']
mmarkers = dict(zip(fields,markers))
ccolors = dict(zip(fields,colors))

# show zlim vs season
"""
fig, ax = plt.subplots(figsize=(12,8))
for field in fields:
    plot_zlim(dfb,field=field,fig=fig,ax=ax,
              color=ccolors[field],marker=mmarkers[field])
"""

#grab mean zlim vs dist

df_dist = df.groupby(['field','season']).apply(lambda x:get_dist(x),
                                                include_groups=False).reset_index()

print(df_dist)
idx = df_dist['field'] == 'COSMOS'
idx &= df_dist['season'] == 3

sel = df_dist[idx]

fig, ax = plt.subplots()

ax.plot(sel['dist_center'],sel['zlim_0.98'],'ko')


plt.show()