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


def plot_cosmo_summary(data, udfs, dfs):

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
            plot_allOS(resdf, config, varx=timescale, legx=timescale, vary=vary,
                       legy=leg[vary], prior=prior, figtitle=dd[prior], dbNorm='')

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


opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList
timescale = opts.timescale
udfs = opts.UDFs.split(',')
dfs = opts.DFs.split(',')

config = pd.read_csv(dbList, comment='#')

data = load_data(dbDir, config)
print(data.columns)
# data['dbName_DD'] += '_{}'.format(budget)
plot_cosmo_summary(data, udfs, dfs)
