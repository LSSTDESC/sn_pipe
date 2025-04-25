#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 10:09:49 2025

@author: philippe.gris@clermont.in2p3.fr
"""


from optparse import OptionParser
# import matplotlib.pyplot as plt
import pandas as pd
from sn_plotter_analysis import plt
import numpy as np


def rename_selection_criteria(df):
    """
    Function to rename selection criteria for plot custom.

    Parameters
    ----------
    df : pandas df
        Data to process.

    Returns
    -------
    df : pandas df
        df with modified selection criteria name.

    """

    rorig = ['nosel',
             'n_epochs_phase_minus_10 >= 1',
             'n_epochs_phase_plus_20 >= 1',
             'n_epochs_m10_p35 >= 4',
             'n_epochs_m10_p5 >= 1',
             'n_epochs_p5_p20 >= 1',
             'n_bands_m8_p10 >= 2',
             'fitstatus == fitok',
             'sigmat0 <= 2.0',
             'sigmax1 <= 1']
    rorig += ['Nfilt_2 >= 3']
    # rorig += ['Nfilt_5 >= 2', 'sigmaC <= 0.04']
    renew = ['no selection',
             '$N_{epochs}(p\leq-10)\geq 1$',
             '$N_{epochs}(p\geq+20)\geq 1$',
             '$N_{epochs}(-10 \leq p\leq+35)\geq 4$',
             '$N_{epochs}(-10 \leq p\leq+5)\geq 1$',
             '$N_{epochs}(+15 \leq p\leq+20)\geq 1$',
             '$N_{epochs}(-8 \leq p\leq+10)\geq 2$',
             '                 fit ok            ',
             '$\sigma_{T_0}\leq 2$',
             '$\sigma_{x_1}\leq 1$']
    renew += ['$N_{band}(SNR\geq 2)\geq 3$']
    """
    renew += ['$N_{band}(SNR\geq 5)\geq 2$',
          '$\sigma_C \leq 0.04$']
    """
    torep = dict(zip(rorig, renew))
    for key, vals in torep.items():
        df['sel_str'] = df['sel_str'].str.replace(key, vals)

    return df


def plot_effi(data, field, thestyle,
              varx='sel_str', legx='',
              varya='effi', err_varya='err_effi',
              legya='Observing Efficiency [%]',
              varyb='nsn', err_varyb='err_nsn',
              legyb='N$_{SN}$'):
    """
    Main function plot

    Parameters
    ----------
    data : pandas df
        Data to plot.
    field : str
        field.
    thestyle : str
        plot style finle name.
    varx : str, optional
        x-axis variable. The default is 'sel_str'.
    legx : str, optional
        x-axis label. The default is ''.
    varya : str, optional
        y-axis variable for the first plot. The default is 'effi'.
    err_varya : str, optional
        y-axis variable error for the first plot. The default is 'err_effi'.
    legya : str, optional
        y-axis label for the first plot. The default is 'Observing Efficiency [%]'.
    varyb : str, optional
        y-axis variable for the second plot. The default is 'nsn'.
    err_varyb : str, optional
        y-axis variable error for the second plot. The default is 'err_nsn'.
    legyb : str, optional
        y-axis label for the second plot. The default is 'N$_{SN}$'.

    Returns
    -------
    None.

    """

    data = rename_selection_criteria(data)
    dbName = data['dbName'].unique().tolist()[0]

    idx = data['field'] == field
    sel = data[idx]

    sel['season'] = sel['season'].astype(int)

    # sel = sel.sort_values(by=['season'])
    seasons = sel['season'].unique().tolist()

    seasons.sort()

    fig, ax = plt.subplots(figsize=(12, 10), nrows=2)
    fig.suptitle('{} \n {}'.format(dbName, field))
    fig.subplots_adjust(hspace=0.05, right=0.82)

    """
    ttimes = range(1, 12)
    lls = ['solid']*4+['dashed']*4+['dotted']*4
    mmarkers = ['o', '*', '^', 'h']*3
    listy = dict(zip(ttimes, lls))
    marks = dict(zip(ttimes, mmarkers))
    """

    for seas in seasons:
        if seas > 10:
            continue
        idxb = sel['season'] == seas
        selb = sel[idxb]

        idxs = thestyle['season'] == seas
        sels = thestyle[idxs]
        marker = sels['marker'].values[0]
        ls = sels['ls'].values[0]
        color = sels['color'].values[0]

        erry_a = None
        erry_b = None

        if err_varya != '':
            erry_a = selb[err_varya]

        if err_varyb != '':
            erry_b = selb[err_varyb]

        ax[1].errorbar(selb[varx], selb[varya],
                       yerr=erry_a,
                       marker=marker,
                       linestyle=ls,
                       label='season {}'.format(seas),
                       mfc='None', ms=10, color=color)

        ax[0].errorbar(selb[varx], selb[varyb],
                       yerr=erry_b,
                       marker=marker,
                       linestyle=ls,
                       label='season {}'.format(seas),
                       mfc='None', ms=10, color=color)

    for i in range(2):
        ax[i].grid(visible=True)

    # ax[1].set_ylim([0., 101.])

    ax[1].set_ylabel(r'{}'.format(legya),
                     fontsize=15, fontweight='bold')
    ax[0].set_ylabel(r'{}'.format(legyb), fontsize=15, fontweight='bold')
    ax[0].set_xticklabels([])
    ax[1].tick_params(axis='x', labelrotation=20., labelsize=12)
    ax[1].tick_params(axis='y', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)

    ax[1].legend(loc='upper center',
                 bbox_to_anchor=(1.14, 1.4),
                 ncol=1, fontsize=15, frameon=False)


def plotIt(df_effi, plot_style_file, plot_style_dir,
           field='COSMOS', plots=['effi_nsn']):
    """
    Function to draw a set of plots

    Parameters
    ----------
    df_effi : pandas df
        Data to plot.
    plot_style_file : csv file
        plot style.
    plot_style_dir: str.
         loc dir of the plot style files.
    field : str, optional
        field to plot. The default is 'COSMOS'.
    plots: list(str)
        list of plots to be drawned

    Returns
    -------
    None.

    """

    thestyle = pd.read_csv(
        '{}/{}'.format(plot_style_dir, plot_style_file), comment='#')

    if 'effi_nsn' in plots:
        plot_effi(df_effi, field, thestyle)
    if 'pull' in plots:
        plot_effi(df_effi, field, thestyle,
                  varya='mu_mu', err_varya='', legya='mu',
                  varyb='sigma_mu', err_varyb='', legyb='sigma_mu')
        plot_effi(df_effi, field, thestyle,
                  varya='mean_mu', err_varya='', legya='mean mu',
                  varyb='std_mu', err_varyb='', legyb='std mu')
        plot_effi(df_effi, field, thestyle,
                  varya='mu_color', err_varya='', legya='$\mu_{pull}^{color}$',
                  varyb='sigma_color', err_varyb='', legyb='$\sigma_{pull}^{color}$')
        plot_effi(df_effi, field, thestyle,
                  varya='mean_color', err_varya='', legya='$<pull^{color}>$',
                  varyb='std_mu', err_varyb='', legyb='$std(pull^{color})$')
        plot_effi(df_effi, field, thestyle,
                  varya='mean_x1', err_varya='', legya='$<pull^{x1}>$',
                  varyb='std_x1', err_varyb='', legyb='$std(pull^{x1})$')
        plot_effi(df_effi, field, thestyle,
                  varya='pvalue_kurtosis_mu', err_varya='', legya='mu - kurtosis pv',
                  varyb='pvalue_kurtosis_color', err_varyb='', legyb='color kurtosis pv')
        plot_effi(df_effi, field, thestyle,
                  varya='kurtosis_mu', err_varya='', legya='mu - kurtosis',
                  varyb='kurtosis_color', err_varyb='', legyb='color kurtosis')


def plot_nsn_selection(data, config):
    """
    Function to plot ns vs selection criteria for each OS

    Parameters
    ----------
    data : pandas df
        Data to process.
    config : pandas df
        configuration for the plot.

    Returns
    -------
    None.

    """

    dbNames = data['dbName'].unique()

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.subplots_adjust(right=0.75)

    data = rename_selection_criteria(data)
    for dbName in dbNames:
        idx = data['dbName'] == dbName
        sel_data = data[idx]
        sel_data = sel_data.sort_values(by=['nsn'], ascending=False)
        idc = config['dbName'] == dbName
        sel_conf = config[idc]

        dbName_plot = sel_conf['dbName_plot'].values[0]
        ls = sel_conf['ls'].values[0]
        marker = sel_conf['marker'].values[0]
        color = sel_conf['color'].values[0]

        ax.errorbar(sel_data['sel_str'], sel_data['nsn'],
                    yerr=sel_data['err_nsn'],
                    linestyle=ls, color=color,
                    marker=marker, mfc='None', label=dbName_plot)

    ax.set_ylabel(r'$N_{SN}$')
    ax.grid(visible=True)

    ax.legend(loc='upper center',
              bbox_to_anchor=(1.2, 0.7),
              ncol=1, fontsize=12, frameon=False)

    ax.tick_params(axis='x', labelrotation=20., labelsize=12)

    plt.tight_layout()


def get_summary(data, df_config):
    """
    Function to grab summary infos

    Parameters
    ----------
    data : pandas df
        Data to process.
    df_config : pandas df
        config for the plot.

    Returns
    -------
    None.

    """

    print(data.columns)
    df_sum = data.groupby(['dbName', 'sel_str']).apply(
        lambda x: get_stat(x)).reset_index()

    print(df_sum)

    # plot summary here
    # plot_nsn_selection(df_sum, df_config)

    df_sum_field = data.groupby(['dbName', 'sel_str', 'field']).apply(
        lambda x: get_stat(x)).reset_index()

    print(df_sum_field)

    last_sel = 'Nfilt_2 >= 3'

    idx = df_sum['sel_str'] == last_sel
    df_sum = df_sum[idx]

    idxb = df_sum_field['sel_str'] == last_sel
    df_sum_field = df_sum_field[idxb]

    # now merge

    df_m = df_sum_field.merge(df_sum, left_on=['dbName'], right_on=['dbName'])

    print(df_m[['dbName', 'field', 'nsn_x', 'nsn_y']])

    latexIt(df_m)


def latexIt(df_m):
    """
    Function to generate a table in latex format

    Parameters
    ----------
    df_m : pandas df
        Data to process.

    Returns
    -------
    None.

    """
    df_m['rat_nsn'] = 100.*df_m['nsn_x']/df_m['nsn_y']

    fields = df_m['field'].unique()

    print('\begin{table}[!htbp]')
    print('\begin{center}')
    print('\caption\{\}\label\{tab:pzreq_final\}')
    print('\begin{tabular}{l|c|c|c|c|c|c|c|c}')
    print('\hline')
    print('\hline')

    entete = 'Observing Strategy & nsn'

    for fi in fields:
        entete += ' & {}'.format(fi)

    print(entete)
    df_m = df_m.sort_values(by=['nsn_y'], ascending=False)
    dbNames = df_m['dbName'].unique()
    for dbName in dbNames:

        idx = df_m['dbName'] == dbName
        sel = df_m[idx]
        mystr = '{} '.format(dbName.split('_v')[0])
        nsn = int(sel['nsn_y'].mean())
        err_nsn = int(sel['err_nsn_y'].mean())
        mystr += '& {} \pm {}'.format(nsn, err_nsn)
        for field in fields:
            idxf = sel['field'] == field
            sol = sel[idxf]
            rat_nsn = sol['rat_nsn'].values[0]
            mystr += '& {} \% '.format(np.round(rat_nsn, 1))

        mystr += '\\\\'
        print(mystr)

    print('\hline')
    print('\hline')
    print('\end{tabular}')
    print('\end{center}')
    print('\end{table}')


def get_stat(x):
    """
    Function to estimated sum and error

    Parameters
    ----------
    x : pandas df
        Data to process.

    Returns
    -------
    df : pandas df
        Result.

    """

    rr = {'nsn': [x['nsn'].sum()],
          'err_nsn': np.sqrt((x['err_nsn']*x['err_nsn']).sum())}

    df = pd.DataFrame.from_dict(rr)

    df['err_nsn'] = df['err_nsn'].astype(int)

    return df


parser = OptionParser(
    description='Script to plot SN selection criteria (efficiencies, pull mean and sigma)')

parser.add_option('--dbDir', type=str,
                  default='../effi_pull_stat',
                  help='OS location dir[%default]')
parser.add_option('--config', type=str,
                  default='config_ana_selplot.csv',
                  help='config file [%default]')
parser.add_option('--plot_style_file', type=str,
                  default='effi_pull_style.csv',
                  help='plot style [%default]')
parser.add_option('--plot_style_dir', type=str,
                  default='input/plots/effi_pull_stat',
                  help='plot style [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFS_a,EDFS_b',
                  help='fields to process [%default]')
parser.add_option('--plots', type=str,
                  default='effi_nsn,pull,plot_summary',
                  help='fields to process [%default]')


opts, args = parser.parse_args()

dbDir = opts.dbDir
config = opts.config
plot_style_file = opts.plot_style_file
plot_style_dir = opts.plot_style_dir
fields = opts.fields.split(',')
plots = opts.plots.split(',')


# load the config file
df_config = pd.read_csv(config, comment='#')

# load the plot style file
fName = '{}/{}'.format(plot_style_dir, plot_style_file)
print('loading', fName)
thestyle = pd.read_csv(fName, comment='#')

# concat both

df_config = df_config.merge(thestyle, left_on=['dbName'], right_on=[
    'dbName'], suffixes=['', ''])
dbNames = df_config['dbName'].unique()
# load the data to plot
data = pd.DataFrame()
for i, row in df_config.iterrows():
    dat_ = pd.read_hdf('{}/{}.hdf5'.format(dbDir, row['dbName']))
    data = pd.concat((data, dat_))

# plots here
for i, row in df_config.iterrows():
    idx = data['dbName'] == row['dbName']
    for field in fields:
        plotIt(data[idx], row['plot_style_file'],
               plot_style_dir, field=field, plots=plots)

# make sum nsn

if 'plot_summary' in plots:
    get_summary(data, df_config)

plt.show()
