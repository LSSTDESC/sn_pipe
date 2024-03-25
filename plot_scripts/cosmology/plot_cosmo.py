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
from sn_analysis.sn_tools import load_cosmo_data
from optparse import OptionParser
import glob


def load_data_deprecated(dbDir, config):
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


def plot_cosmo_summary(data, udfs, dfs, comment_on_plot, fill_between=False):
    """
    Function to plot summary cosmo

    Parameters
    ----------
    data : pandas df
        Data to plot.
    udfs : list(str)
        List of ultradeep fields.
    dfs : list(str)
        List of deep fields (DF).
    comment_on_plot :str 
        To add a comment on the plot
    fill_between : bool, optional
        To fill in area +-1sigma. The default is False.

    Returns
    -------
    None.

    """

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
                       comment_on_plot=comment_on_plot,
                       fill_between=fill_between)

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


# def plot(dbDir,dbList, timescale, config, vars=[['MoM', 'nsn_z_0.8']], comment_plot=''):


def process_cosmo(config, cols_group,
                  cols=['MoM', 'WFD_TiDES', 'all_Fields', 'nsn_z_0.8']):
    """
    Function to process cosmo files

    Parameters
    ----------
    config : pandas df
        configuration params.
    cols_group : list(str)
        List of cols for groupby.
    cols : list(str), optional
        Observable to process. The default is ['MoM', 'WFD_TiDES', 'all_Fields', 'nsn_z_0.8'].

    Returns
    -------
    df_tot : pandas df
        Processed data.

    """

    df_tot = pd.DataFrame()

    for i, row in config.iterrows():
        dbName = row['dbName']
        dbDir = row['dbDir']
        spectro_config = row['spectro_config']
        df = load_cosmo_data(dbDir, dbName, cols_group,
                             spectro_config, cols=cols)
        df['dbName_DD'] = dbName
        df['dbNamePlot'] = row['dbNamePlot']
        df_tot = pd.concat((df_tot, df))

    return df_tot


parser = OptionParser(description='Script to analyze SN prod')

"""
parser.add_option('--dbDir', type=str, default='../cosmo_fit',
                  help='OS location dir[%default]')
"""
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
                  default='Host spectro-z only',
                  help='comment for the SMoM plot [%default]')
parser.add_option('--fill_between', type=int,
                  default=0,
                  help='to fill +-1 sigma area with yellow [%default]')

opts, args = parser.parse_args()

# dbDir = opts.dbDir
dbList = opts.dbList
timescale = opts.timescale
udfs = opts.UDFs.split(',')
dfs = opts.DFs.split(',')
comment_on_plot = opts.comment_on_plot
fill_between = opts.fill_between

config = pd.read_csv(dbList, comment='#')
"""
dbDir = '../cosmo_fit_WFD_paper_spectroz_nolowzopti'
data = load_data(dbDir, config)
print(data.columns)
print(test)
"""

# data['dbName_DD'] += '_{}'.format(budget)

#plot_pulls(data, timescale, config)
# cosmo summary plot
#plot_cosmo_summary(data, udfs, dfs, comment_on_plot, fill_between)

cols = ['MoM', 'WFD_TiDES', 'all_Fields',
        'nsn_z_0.8', 'WFD_DESI1', 'WFD_DESI2']
cols = ['MoM', 'WFD_TiDES', 'all_Fields',
        'WFD_desi2_footprint', 'WFD_desi_lrg_footprint', 'WFD_desi_bgs_footprint']
"""
cols = ['MoM', 'all_Fields',
        'nsn_z_0.8', 'nsn_z_0.8_sigma_mu', 'nsn_rat_highz']
"""
data = process_cosmo(
    config, [timescale, 'prior', 'dbName_DD', 'dbName_WFD'], cols=cols)
print(data)

priors = ['prior']

dd = dict(zip(priors, ['']))

print(data.columns)

for prior in priors:
    plot_allOS(data, config, varx=timescale,
               legx=timescale, vary='MoM_mean',
               legy='$SMoM$', vary_std='MoM_std', prior=prior,
               figtitle=dd[prior], dbNorm='',
               comment_on_plot=comment_on_plot,
               fill_between=fill_between)
    plot_allOS(data, config, varx=timescale, legx=timescale,
               vary='WFD_desi_lrg_footprint_mean',
               legy='$N_{spectro-z}^{desi~lrg}$', vary_std='', prior=prior,
               figtitle=dd[prior], dbNorm='',
               comment_on_plot=comment_on_plot,
               fill_between=fill_between)
    plot_allOS(data, config, varx=timescale, legx=timescale,
               vary='WFD_desi_bgs_footprint_mean',
               legy='$N_{spectro-z}^{desi~bgs}$', vary_std='', prior=prior,
               figtitle=dd[prior], dbNorm='',
               comment_on_plot=comment_on_plot,
               fill_between=fill_between)
    plot_allOS(data, config, varx=timescale, legx=timescale,
               vary='WFD_desi2_footprint_mean',
               legy='$N_{spectro-z}^{desi2}$', vary_std='', prior=prior,
               figtitle=dd[prior], dbNorm='',
               comment_on_plot=comment_on_plot,
               fill_between=fill_between)

    """
    plot_allOS(data, config, varx=timescale, legx=timescale,
               vary='nsn_z_0.8_mean',
               legy='$N_{SN}^{z\geq0.8}$', vary_std='nsn_z_0.8_std', prior=prior,
               figtitle=dd[prior], dbNorm='',
               comment_on_plot=comment_on_plot,
               fill_between=fill_between)
    """
    """
    plot_allOS(data, config, varx=timescale, legx=timescale,
               vary='nsn_rat_highz_mean',
               legy='$N_{SN}^{z\geq0.8,\sigma_{\mu}\leq0.12}$', vary_std='nsn_rat_highz_std', prior=prior,
               figtitle=dd[prior], dbNorm='',
               comment_on_plot=comment_on_plot,
               fill_between=fill_between)
    """
    """
    plot_allOS(data, config, varx=timescale, legx=timescale,
               vary='nsn_z_0.8_sigma_mu_mean',
               legy='$N_{SN}^{z\geq0.8,\sigma_{\mu}\leq0.12}$', vary_std='nsn_z_0.8_sigma_mu_std', prior=prior,
               figtitle=dd[prior], dbNorm='',
               comment_on_plot=comment_on_plot,
               fill_between=fill_between)

    plot_allOS(data, config, varx=timescale,
               legx=timescale, vary='WFD_TiDES_mean',
               legy='$N_{SN}^{TiDES}$', vary_std='WFD_TiDES_std', prior=prior,
               figtitle=dd[prior], dbNorm='',
               comment_on_plot=comment_on_plot,
               fill_between=fill_between)
    plot_allOS(data, config, varx=timescale,
               legx=timescale, vary='WFD_DESI1_mean',
               legy='$N_{SN}^{DESI}$', vary_std='WFD_DESI1_std', prior=prior,
               figtitle=dd[prior], dbNorm='',
               comment_on_plot=comment_on_plot,
               fill_between=fill_between)
    """
plt.show()
#plot(data, timescale, config)
