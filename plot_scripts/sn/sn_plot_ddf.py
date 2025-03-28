#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:38:48 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import numpy as np
from sn_plotter_analysis.sn_plot import plot_nsn_year_all
import matplotlib.pyplot as plt


def load_process(dataDir, dbName):
    """
    Function to load and process data

    Parameters
    ----------
    dataDir : str
        Data directory.
    dbName : str
        OS of interest.

    Returns
    -------
    dfb : pandas df
        output data.

    """

    fName = '{}/{}/nsn_ddf_G10_JLA.hdf5'.format(dataDir, dbName)
    df = pd.read_hdf(fName)

    """
    dfb = df.groupby(['field', 'year', 'zmeas']).apply(
        lambda x: get_val(x)).reset_index()
    """
    return df


def get_val(grp):
    """
    Function to get sum(nsn) and errors

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    pandas df
        Output data.

    """

    nsn = (grp['nsn']*grp['frac']).sum()
    err_nsn = np.sqrt((grp['nsn_err']*grp['frac']**2).sum())

    ddout = {}
    ddout['nsn'] = [nsn]
    ddout['nsn_err'] = [err_nsn]

    return pd.DataFrame.from_dict(ddout)


def plot_nsn_z(data, figtit='', xvar='season'):
    """
    Function to plot nsn vs z

    Parameters
    ----------
    data : pandas df
        Data to plot.
    figtit : str, optional
        Figure title. The default is ''.

    Returns
    -------
    None.

    """

    idx = data[xvar] > 0
    idx &= data[xvar] < 11
    data = pd.DataFrame(data[idx])

    print('rrr', data)
    years = data[xvar].unique()
    print('hello', years)
    ttimes = range(1, 12)
    lls = ['solid']*4+['dashed']*4+['dotted']*4
    mmarkers = ['o', '*', '^', 'h']*3
    listy = dict(zip(ttimes, lls))
    marks = dict(zip(ttimes, mmarkers))
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(figtit, fontweight='bold')
    for yy in years:
        idx = data[xvar] == yy
        sel = data[idx]
        ax.errorbar(sel['zmeas'], sel['nsn'], yerr=sel['nsn_err'], ls=listy[yy],
                    color='k', marker=marks[yy], label='year {}'.format(yy), mfc='None')

    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'N$_{SN}$')
    ax.grid(visible=True)
    ax.set_ylim([0, None])
    ax.legend()


parser = OptionParser(
    'Script to plot nsn for DDF')

parser.add_option("--dataDir", type=str,
                  default='../sn_ddf',
                  help="data dir[%default]")
parser.add_option('--config', type=str,
                  default='input/plots/config_ana.csv',
                  help='OS DD list[%default]')
opts, args = parser.parse_args()

dataDir = opts.dataDir
config = opts.config

# read config file
conf_df = pd.read_csv(config, comment='#')

# load the data

dbNames = conf_df['dbName'].unique()

resdf = pd.DataFrame()
for dbName in dbNames:
    df = load_process(dataDir, dbName)
    df['dbName'] = dbName
    resdf = pd.concat((resdf, df))


# resdf['frac'] = 1
idx = resdf['field'] == 'XMM-LSS'
sel = resdf[idx]

print(sel['season'].unique())


nsn_tot = resdf.groupby(['dbName', 'year']).apply(
    lambda x: get_val(x)).reset_index()

print(nsn_tot)
print(nsn_tot['nsn'].sum())
fields = ['COSMOS', 'CDFS', 'XMM-LSS', 'ELAISS1', 'EDFS_a', 'EDFS_b']
plot_nsn_year_all(nsn_tot, conf_df, cumul=True, figtit=','.join(fields))
plot_nsn_year_all(nsn_tot, conf_df, cumul=False, figtit=','.join(fields))

fields = ['COSMOS', 'XMM-LSS']
idx = resdf['field'].isin(fields)
nsn_totb = resdf[idx].groupby(['dbName', 'season']).apply(
    lambda x: get_val(x)).reset_index()
print(nsn_totb)
plot_nsn_year_all(nsn_totb, conf_df, xvar='season',
                  xlab='season', cumul=True, figtit=','.join(fields))
plot_nsn_year_all(nsn_totb, conf_df, xvar='season',
                  xlab='season', cumul=False, figtit=','.join(fields))

"""
idx = resdf['field'] == 'XMM-LSS'
idx &= resdf['year'] == 4

for i, row in resdf[idx].iterrows():
    print(i, row.values)
print(resdf[idx])
"""

nsn_field = resdf.groupby(['dbName', 'year', 'field']).apply(
    lambda x: get_val(x)).reset_index()

print(resdf.columns)
nsn_year = resdf.groupby(['dbName', 'season', 'field', 'zmeas']).apply(
    lambda x: get_val(x)).reset_index()
print('iii', nsn_year['nsn'].sum())
fields = ['XMM-LSS']
idx = nsn_year['field'].isin(fields)
sel = nsn_year[idx]
print(sel)
plot_nsn_z(sel, figtit=','.join(fields), xvar='season')
plt.show()
