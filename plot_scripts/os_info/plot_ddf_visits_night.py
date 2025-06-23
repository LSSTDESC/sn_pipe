#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:46:22 2025

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from optparse import OptionParser
import matplotlib.pyplot as plt
from sn_plotter_metrics.plot4metric import plot_vs_OS
from sn_tools.sn_utils import multiproc
import numpy as np
from sn_plotter_os_info.ddf_visits_night_tools import plot_stat_visits_vs_exp
from sn_plotter_os_info.ddf_visits_night_tools import ana_seq_multi
from sn_plotter_os_info.ddf_visits_night_tools import summary_seq, get_ratios
from sn_plotter_os_info.ddf_visits_night_tools import get_stat_indiv
from sn_plotter_os_info.ddf_visits_night_tools import plot_obs_time_night
import operator as op


def plot_seq_frac(data, dbName, field='COSMOS', season=1,
                  what='seq_frac', legy='sequence fraction [%]'):
    """
    Funtion to plot DDF sequence fraction

    Parameters
    ----------
    data : pandas df
        Data to process.
    dbName : str
        OS name.
    field : str, optional
        Field considered. The default is 'COSMOS'.
    season : int, optional
        season of interest. The default is 1.
    what : str, optional
        What to plot. The default is 'seq_frac'.
    legy : str, optional
        y-axis legend. The default is 'sequence fraction [%]'.

    Returns
    -------
    None.

    """

    idx = data['target_name'] == field
    idx &= data['year'] == season
    sel = data[idx]

    figtit = dbName
    figtit += '\n {} - year {}'.format(field, season)

    plot_vs_OS(sel, varx='seq_tot',
               vary=what,
               legy=legy,
               title=figtit, fig=None, ax=None,
               label='', color='k', marker='.', ls='solid', mfc='k', mec='k')

    selb = sel.sort_values(by=[what], ascending=False)
    print(selb[['seq_tot', what]][:2])


def plot_all(ro, dbName, field='DD:COSMOS', season=3):
    """
    Function to plot a serie of results

    Parameters
    ----------
    ro : pandas df
        Data to process.
    dbName : str
        Db name.
    field : str, optional
        DDF field name. The default is 'DD:COSMOS'.
    season : int, optional
        season/year to show. The default is 3.

    Returns
    -------
    None.

    """

    idx = ro['dbName'] == dbName
    idx &= ro['target_name'] == field
    idx &= ro['year'] == season
    ro = ro[idx]

    plot_seq_frac(ro, dbName, field=field, season=season)

    idx = ro['y'] > 0
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='nnights', legy='Number of nights')
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='seq_frac')

    idx = ro['u'] > 0
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='nnights', legy='Number of nights')
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='seq_frac')

    idx = ro['u'] == 0
    idx &= ro['y'] == 0
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='nnights', legy='Number of nights')
    """
    plot_seq_frac(ro[idx], dbName, field=field, season=season,
                  what='seq_frac')

    plt.show(block=False)


def plot_summary(data, field='DD:COSMOS',
                 varx='year', labx='year',
                 vary='seq_tot_y', laby='',
                 figtit='DD:COSMOS',
                 df_config=pd.DataFrame()):
    """
    Summary plot

    Parameters
    ----------
    data : pandas df
        Data to process.
    field : str, optional
        Field type. The default is 'DD:COSMOS'.
    varx : str, optional
        x-axis variable. The default is 'year'.
    labx : str, optional
        x-axis label. The default is 'year'.
    vary : str, optional
        y-axis variable. The default is 'seq_tot_y'.
    laby : str, optional
        y-axis label. The default is ''.
    figtit : str, optional
        Figure title. The default is 'DD:COSMOS'.
    df_config : pandas df, optional
        config for the plot. The default is pd.DataFrame().

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(figtit)
    fig.subplots_adjust(right=0.75)

    idx = data['target_name'] == field

    sel = data[idx]

    dbNames = sel['dbName'].unique()

    for dbName in dbNames:
        io = sel['dbName'] == dbName
        selb = sel[io]
        idxb = df_config['dbName'] == dbName
        selp = df_config[idxb]
        ls = selp['ls'].values[0]
        marker = selp['marker'].values[0]
        color = selp['color'].values[0]
        dbNameb = selp['dbName_plot'].values[0]

        ax.plot(selb[varx], selb[vary],
                ls=ls, marker=marker, color=color, mfc='None', label=dbNameb)

    ax.grid(visible=True)
    ax.set_xlabel(r'{}'.format(labx))
    ax.set_ylabel(r'{}'.format(laby))
    if laby == '':
        ax.tick_params(axis='y', labelrotation=20, labelsize=10)

    ax.legend(loc='upper center',
              bbox_to_anchor=(1.20, 0.7),
              ncol=1, fontsize=12, frameon=False)

    # plt.tight_layout()


def plot_fields(dft, config, fields):
    """
    Function to make a set of plots

    Parameters
    ----------
    dft : pandas df
        Data to plot.
    config : str
        config file name for the plot.
    fields: list(str)
        list of fields to plot
    Returns
    -------
    None.

    """

    # load config for the plot
    df_config = pd.read_csv(config, comment='#')

    for field in fields:
        plot_summary(dft, field=field, df_config=df_config,
                     figtit=field.split(':')[-1])
        """
        plot_summary(dft, field=field, vary='seq_frac_y',
                     laby='Fraction of nights [%]', df_config=df_config)
        plot_summary(dft, field=field, vary='nvisits_y',
                     laby='$N_{visits}^{y}$', df_config=df_config)
        """
    # plt.show()


def plot_stat(dft, config, fields=['DD:COSMOS', 'DD:XMM_LSS']):
    """
    Function to plot nnights vs year per OS for 3 cases:
        nvisits_obs=nvisits_expeccted
        nvisits_obs > nvisits_expected
        nvisits_obs < nvisits_expected

    Parameters
    ----------
    dft : pandas df
        Data to process.
    config : str
        config file nema (csv).

    Returns
    -------
    None.

    """

    # load config for the plot
    df_config = pd.read_csv(config, comment='#')
    for field in fields:
        figtot = field
        figtit = figtot + '\n $\\frac{N_{visits}^{obs}}{N_{visits}^{exp}}$=1'
        plot_summary(dft, field=field, figtit=figtit,
                     df_config=df_config, vary='frac_equal',
                     laby='Fraction of nights [%]')
        plt.tight_layout()
        figtit = figtot + '\n $\\frac{N_{visits}^{obs}}{N_{visits}^{exp}}$>1'
        plot_summary(dft, field=field, figtit=figtit, vary='frac_plus',
                     laby='Fraction of nights [%]', df_config=df_config)
        plt.tight_layout()
        figtit = figtot + '\n $\\frac{N_{visits}^{obs}}{N_{visits}^{exp}}$<1'
        plot_summary(dft, field=field, figtit=figtit, vary='frac_minus',
                     laby='Fraction of nights [%]', df_config=df_config)

        plt.tight_layout()
    plt.show(block=False)


def get_stats(df_summary, df_orig):
    """
    Funtion to get the number of nights corresponding to a sequence
    (ie nnights with the sequence, nnights with nvisits < n_sequence, 
     and nnights with nvisits > n_sequence)

    Parameters
    ----------
    df_summary : pandas df
        Data to process.
    df_orig : pandas df
        Data to process.

    Returns
    -------
    rr : pandas df
        Result.

    """

    rr = df_summary.groupby(['dbName', 'target_name', 'year']).apply(
        lambda x: get_stat_indiv(x, df_orig), include_groups=False).reset_index()

    return rr


def analysis_sequences(df_summary, df_orig,
                       dbName='baseline_v4.3.1_10yrs',
                       target_name='DD:COSMOS', year=3):
    """
    Function to analyze sequences for each field/season

    Parameters
    ----------
    df_summary : pandas df
        Summary results.
    df_orig : pandas df
        original results.
    dbName: str, optional.
       OS name. The default is 'baseline_v4.3.1_10yrs'.
    target_name : str, optional
        field name. The default is 'DD:COSMOS'.
    year : int, optional
        year. The default is 3.
    Returns
    -------
    None.

    """
    idx = df_summary['dbName'] == dbName
    idx &= df_summary['target_name'] == target_name

    idx &= df_summary['year'] == year

    rr = df_summary[idx].groupby(['dbName', 'target_name', 'year']).apply(
        lambda x: get_ratios(x, df_orig), include_groups=False)

    plot_stat_visits_vs_exp(rr, op.ge, '>',
                            bins=np.arange(1.0, 2.5, 0.01), field=target_name)
    plot_stat_visits_vs_exp(rr, op.le, '<',
                            bins=np.arange(0.0, 1.1, 0.01), field=target_name)

    plt.show(block=False)


def ana_plot(ro, config, fields):

    dft = ro.groupby(['dbName', 'target_name', 'year']).apply(
        lambda x: summary_seq(x), include_groups=False).reset_index()

    # plots here
    plot_fields(dft, config, fields)

    rr = get_stats(dft, data)

    plot_stat(rr, config, fields)

    while (1):
        print('**** Nvisits status result ****')
        print('list of OS:')
        print(dft['dbName'].unique())
        answer = input('dbName?')
        dbName = answer
        if answer == 'exit':
            break
        idxa = dft['dbName'] == dbName
        sela = dft[idxa]
        print('list of fields')
        print(sela['target_name'].unique())
        answer = input('target?')
        target_name = answer
        idxb = sela['target_name'] == target_name
        selb = sela[idxb]
        print('years:')
        print(selb['year'].to_list())
        answer = input('year?')
        year = int(answer)

        analysis_sequences(dft, data, dbName=dbName,
                           target_name=target_name, year=year)


def ana_indiv(ro):
    """
    Function to plot individual channel results

    Parameters
    ----------
    ro : pandas df
        Data to process.

    Returns
    -------
    None.

    """

    while 1:
        print('********Individual channel analysis********')
        print('list of OS:')
        print(ro['dbName'].unique())
        answer = input('dbName?')
        dbName = answer
        if answer == 'exit':
            break
        idxa = ro['dbName'] == dbName
        sela = ro[idxa]
        print('list of fields')
        print(sela['target_name'].unique())
        answer = input('target?')
        target_name = answer
        idxb = sela['target_name'] == target_name
        selb = sela[idxb]
        print('years:')
        print(selb['year'].unique())
        answer = input('year?')
        year = int(answer)

        plot_all(ro, dbName, field=target_name, season=year)


def ana_night(data):
    """
    Function to plot the distrib of visits vs night

    Parameters
    ----------
    data : pandas df
        Data to process.

    Returns
    -------
    None.

    """
    while 1:
        print('******** Observations vs night********')
        print('list of OS:')
        print(data['dbName'].unique())
        answer = input('dbName?')
        dbName = answer
        if answer == 'exit':
            break
        idxa = data['dbName'] == dbName
        sela = data[idxa]

        plot_obs_time_night(sela)

        plt.show(block=False)


parser = OptionParser(
    description='Script to analyse DDF visits on a nightly basis from pointings')

parser.add_option("--dbDir", type="str",
                  default='../ddf_visits_night',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='desc_ddf_v4.2.1_10yrs',
                  help="OS name [%default]")
parser.add_option("--dbList", type="str",
                  default='dbList.csv',
                  help="dbList to process [%default]")
parser.add_option("--nproc", type=int,
                  default=8,
                  help="number of procs for multiprocessing [%default]")
parser.add_option("--configplot", type=str,
                  default='config_ana_selplot.csv',
                  help="configuration for the plot [%default]")
parser.add_option("--DDF", type=str,
                  default='DD:COSMOS,DD:XMM_LSS',
                  help="DDF to consider [%default]")
parser.add_option("--plots", type=str,
                  default='plot_indiv,plot_all,plot_night',
                  help="plots to make [%default]")

opts, args = parser.parse_args()

# Load parameters
dbDir = opts.dbDir
dbName = opts.dbName
dbList = opts.dbList
nproc = opts.nproc
config = opts.configplot
fields = opts.DDF.split(',')
plots = opts.plots.split(',')

# load dbNames
df_db = pd.read_csv(dbList, comment='#')

data = pd.DataFrame()
for i, row in df_db.iterrows():
    fName = '{}/{}.hdf5'.format(dbDir, row['dbName'])

    dat_ = pd.read_hdf(fName)
    data = pd.concat((data, dat_))

if 'plot_night' in plots:
    ana_night(data)

# ro = ana_seq(data)
idx = data['target_name'].isin(fields)
data = data[idx]

dbNames = data['dbName'].unique().tolist()

params = {}

params['timescale'] = 'year'
params['data'] = data

ro = multiproc(dbNames, params, ana_seq_multi, nproc)

# this is to plot the seq fraction for a field/season
if 'plot_indiv' in plots:
    ana_indiv(ro)


if 'plot_ana' in plots:
    ana_plot(ro, config, fields)


# analyze_simu_exp(data)
