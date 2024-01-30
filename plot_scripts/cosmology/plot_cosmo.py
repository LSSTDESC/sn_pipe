#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:08:41 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
from sn_plotter_cosmology import plt
import numpy as np
from sn_plotter_cosmology.cosmoplot import Process_OS, plot_allOS
from optparse import OptionParser
import glob


def load_data(dbDir, config):
    """
    Method to load data

    Parameters
    ----------
    dbDir : str
        Data dir.
    config : pandas df
        config (dbName+OS params).

    Returns
    -------
    df : pandas df
        output data.

    """

    df = pd.DataFrame()

    for i, row in config.iterrows():

        search_path = '{}/cosmo_{}_*.hdf5'.format(dbDir, row['dbName'])
        fis = glob.glob(search_path)
        for fi in fis:
            dfa = pd.read_hdf(fi)
            # print(dfa)
            dfa = dfa.replace([np.inf, -np.inf, np.nan], 0)

            df = pd.concat((df, dfa))

    return df


def load_data_old(dbDir, config):
    """
    Method to load data

    Parameters
    ----------
    dbDir : str
        Data dir.
    config : pandas df
        config (dbName+OS params).

    Returns
    -------
    df : pandas df
        output data.

    """

    df = pd.DataFrame()

    for i, row in config.iterrows():
        fi = '{}/cosmo_{}_*.hdf5'.format(dbDir, row['dbName'])
        dfa = pd.read_hdf(fi)
        # print(dfa)
        dfa = dfa.replace([np.inf, -np.inf, np.nan], 0)

        df = pd.concat((df, dfa))

    return df


def plot_cosmo_summary(data, udfs, dfs, comment_on_plot):

    grpCol = [timescale, 'dbName_DD', 'prior']
    # resdf = process_OS(data, grpCol)
    resdf = Process_OS(data, grpCol, udfs, dfs).res

    resdf['UD_mean'] = 0
    resdf['DD_mean'] = 0

    for UD in udfs:
        resdf['UD_mean'] += resdf['{}_mean'.format(UD)]

    for DD in dfs:
        resdf['DD_mean'] += resdf['{}_mean'.format(DD)]

    resdf['DDF_mean'] = resdf['UD_mean']+resdf['DD_mean']

    for tt in ['DDF', 'UD', 'DD', 'WFD']:
        resdf['{}_std'.format(tt)] = 0

    print(resdf, config)
    print(resdf.columns)

    vvars = ['MoM', 'sigma_w0', 'sigma_wa']
    leg = dict(zip(vvars, [r'$SMoM$',
               r'$\sigma_{w_0}$ [%]', r'$\sigma_{w_a}$ [%]']))
    """
    priors = ['prior', 'noprior']

    dd = dict(zip(priors, ['with prior', 'no prior']))

    vvars = ['MoM']
    leg = dict(zip(vvars, [r'$SMoM$']))
    """
    priors = ['prior']

    dd = dict(zip(priors, ['']))

    for vary in vvars:
        for prior in priors:
            plot_allOS(resdf, config, varx=timescale,
                       legx=timescale, vary=vary,
                       legy=leg[vary], prior=prior,
                       figtitle=dd[prior], dbNorm='',
                       comment_on_plot=comment_on_plot)

    """
    vvarsb = ['DDF', 'UD', 'DD', 'WFD']
    # DDFs = ['COSMOS', 'XMM-LSS', 'CDFS', 'EDFSa', 'EDFSb', 'ELAISS1']
    DDFs = ['COSMOS', 'XMM-LSS', 'CDFS', 'EDFS', 'ELAISS1']
    vvarsb += DDFs
    legb = [r'$N^{DDF}_{SN}$', r'$N^{UD}_{SN}$',
            r'$N^{DD}_{SN}$', r'$N^{WFD}_{SN}$']
    legbb = []
    for ddf in DDFs:
        bb = r'$N^{'+ddf+'}_{SN}$'
        legb.append(bb)

    legb += legbb

    lleg = dict(zip(vvarsb, legb))

    for vary in vvarsb:
        plot_allOS(resdf, config, varx=timescale, legx=timescale, vary=vary,
                   legy=lleg[vary], prior='prior', figtitle='', dbNorm='')

    """
    # ax.legend()
    # idx = resdf['prior'] == 'noprior'
    # cosmo_plot(resdf[idx])

    # cosmo_four(resdf)
    plt.show()


def plot_pulls(data, timescale, config):
    """
    Function to plot the pulls

    Parameters
    ----------
    data : pandas df
        Data to plot.
    timescale : str
        Timescale (season/year).
    config : pandas df
        config for plot.

    Returns
    -------
    None.

    """

    # first grab mean and std of the pulls
    pull_str = ['w0', 'wa']
    rt = []
    for dbName in data['dbName_DD'].unique():
        idx = data['dbName_DD'] == dbName
        sel_data = data[idx]
        sel_data['w0'] = -1.0
        sel_data['wa'] = 0.0
        seasons = data[timescale].unique()

        for seas in seasons:
            r = [dbName, seas]
            idxb = sel_data[timescale] == seas
            selb = sel_data[idxb]
            for strval in pull_str:
                vv = selb['{}_fit'.format(strval)]-selb[strval]
                vv /= np.sqrt(selb['Cov_{}_{}_fit'.format(strval, strval)])
                r.append(np.mean(vv))
                r.append(np.std(vv))
            rt.append(r)

    df = pd.DataFrame(rt, columns=['dbName', timescale,
                                   'pull_w0_mean',
                                   'pull_w0_std',
                                   'pull_wa_mean',
                                   'pull_wa_std'])
    plot_pull(df, config)
    plot_pull(df, config, what='wa',
              ylabel_up=r'<$\frac{w_{a}-w_{a}^{fit}}{\sigma_{w_{a}}}$>',
              ylabel_down=r'std($\frac{w_{a}-w_{a}^{fit}}{\sigma_{w_{a}}}$)')

    plt.show()


def plot_pull(df, config, what='w0',
              ylabel_up=r'<$\frac{w_{0}-w_{0}^{fit}}{\sigma_{w_{0}}}$>',
              ylabel_down=r'std($\frac{w_{0}-w_{0}^{fit}}{\sigma_{w_{0}}}$)'):
    """
    Function to plot mena and std of the pulls vs year/season

    Parameters
    ----------
    df : pandas df
        Data to plot.
    config : pandas df
        config for plot.
    what : str, optional
        The variable to plot. The default is 'w0'.
    ylabel_up : str, optional
        ylabel (upper plot). 
        The default is r'<$\frac{w_{0}-w_{0}^{fit}}{\sigma_{w_{0}}}$>'.
    ylabel_down : str, optional
        ylabel (down plot). 
        The default is r'std($\frac{w_{0}-w_{0}^{fit}}{\sigma_{w_{0}}}$)'.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(nrows=2, figsize=(13, 9))
    fig.subplots_adjust(hspace=0., left=0.15, right=0.75)
    for dbName in config['dbName'].unique():
        dbNameb = dbName.split('_')
        dbNameb = '_'.join(dbNameb[:-1])
        ida = config['dbName'] == dbName
        color = config[ida]['color'].values[0]
        ls = config[ida]['ls'].values[0]
        marker = config[ida]['marker'].values[0]
        idx = df['dbName'] == dbName
        sel_data = df[idx]
        sel_data = sel_data.sort_values(by=[timescale])
        ax[0].plot(sel_data[timescale],
                   sel_data['pull_{}_mean'.format(what)],
                   color=color,
                   linestyle=ls, marker=marker)
        ax[1].plot(sel_data[timescale],
                   sel_data['pull_{}_std'.format(what)],
                   color=color,
                   linestyle=ls, marker=marker, label=dbNameb)
    ax[0].grid()
    ax[1].grid()

    ax[1].set_xlabel(r'{}'.format(timescale))
    ax[0].set_ylabel(ylabel_up)
    ax[1].set_ylabel(ylabel_down)

    ax[1].legend(loc='upper center',
                 bbox_to_anchor=(1.20, 1.6),
                 ncol=1, fontsize=15, frameon=False)


parser = OptionParser(description='Script to analyze SN prod')

parser.add_option('--dbDir', type=str, default='../cosmo_fit',
                  help='OS location dir[%default]')
parser.add_option('--dbList', type=str,
                  default='input/DESC_cohesive_strategy/config_ana.csv',
                  help='OS name[%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='timescale for plot - year or season [%default]')
parser.add_option('--UDFs', type=str,
                  default='COSMOS,XMM-LSS',
                  help='UD fields [%default]')
parser.add_option('--DFs', type=str,
                  default='CDFS,EDFS,ELAISS1',
                  help='Deep fields [%default]')
parser.add_option('--comment_on_plot', type=str,
                  default='',
                  help='comment for the SMoM plot [%default]')


opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList
timescale = opts.timescale
udfs = opts.UDFs.split(',')
dfs = opts.DFs.split(',')
comment_on_plot = opts.comment_on_plot

config = pd.read_csv(dbList, comment='#')

data = load_data(dbDir, config)
print(data.columns)
# data['dbName_DD'] += '_{}'.format(budget)

#plot_pulls(data, timescale, config)
# cosmo summary plot
plot_cosmo_summary(data, udfs, dfs, comment_on_plot)
