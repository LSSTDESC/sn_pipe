#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 10:09:49 2025

@author: philippe.gris@clermont.in2p3.fr
"""


from optparse import OptionParser
import matplotlib.pyplot as plt
import pandas as pd


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

    dbName = data['dbName'].unique().tolist()[0]

    idx = data['field'] == field
    sel = data[idx]

    print('alors', len(sel))
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


def plotIt(df_effi, thestyle, field='COSMOS', plots=['effi_nsn']):
    """
    Function to draw a set of plots

    Parameters
    ----------
    df_effi : pandas df
        Data to plot.
    thestyle : pandas df
        plot style.
    field : str, optional
        field to plot. The default is 'COSMOS'.
    plots: list(str)
        list of plots to be drawned

    Returns
    -------
    None.

    """

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


parser = OptionParser(
    description='Script to plot SN selection criteria (efficiencies, pull mean and sigma)')

parser.add_option('--dbDir', type=str,
                  default='../effi_pull_stat',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v4.3.1_10yrs',
                  help='OS name [%default]')
parser.add_option('--plot_style_file', type=str,
                  default='plot_style_4_udy.csv',
                  help='plot style [%default]')
parser.add_option('--plot_style_dir', type=str,
                  default='input/plots/effi_pull_stat',
                  help='plot style [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFS_a,EDFS_b',
                  help='fields to process [%default]')
parser.add_option('--plots', type=str,
                  default='effi_nsn,pull',
                  help='fields to process [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
plot_style_file = opts.plot_style_file
plot_style_dir = opts.plot_style_dir
fields = opts.fields.split(',')
plots = opts.plots.split(',')

# load the plot style file
thestyle = pd.read_csv('{}/{}'.format(plot_style_dir, plot_style_file))

# load the data to plot
data = pd.read_hdf('{}/{}.hdf5'.format(dbDir, dbName))


# plots here
for field in fields:
    plotIt(data, thestyle, field=field, plots=plots)

plt.show()
