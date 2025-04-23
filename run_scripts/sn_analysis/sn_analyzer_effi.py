#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:58:22 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import glob
import pandas as pd
from sn_analysis.sn_selection import selection_criteria, select
from sn_analysis.sn_tools import complete_df, get_pulls, sel_for_pull
import numpy as np
import re
import operator
import matplotlib.pyplot as plt


def load_data(dbDir, dbName, runType, field):
    """
    Function to load the data

    Parameters
    ----------
    dbDir : str
        data dir.
    dbName : str
        OS to process.
    runType : str
        runtype.
    field : str
        field.

    Returns
    -------
    df : pandas df
        loaded data.

    """

    theDir = '{}/{}/{}'.format(dbDir, dbName, runType)

    print('scanning', theDir)
    fis = glob.glob('{}/*{}*.hdf5'.format(theDir, field))

    df = pd.DataFrame()

    for fi in fis:

        df_ = pd.read_hdf(fi)

        df = pd.concat((df, df_))

    return df


def get_nsn(vala, valb, norm_factor):
    """
    Function to grab infos

    Parameters
    ----------
    vala : int
        number of sn after sel.
    valb : int
        number of sn before sel.
    norm_factor : float
        normalization factor.

    Returns
    -------
    list
        DESCRIPTION.

    """

    effi = vala/valb

    err_effi = np.sqrt(effi*(1.-effi)/valb)

    nsn = int(effi*valb/norm_factor)

    err_nsn = int(err_effi*valb/norm_factor)

    effi *= 100.
    err_effi *= 100.

    res = [(nsn, err_nsn, np.round(effi, 1), np.round(err_effi, 1))]
    cols = ['nsn', 'err_nsn', 'effi', 'err_effi']

    return pd.DataFrame(res, columns=cols)


def select_str(res, list_sel):
    """
    Function to select a pandas df

    Parameters
    ----------
    res : pandas df
        data to select.

    Returns
    -------
    pandas df
        selected df.

    """
    idx = True
    for vals in list_sel:
        idx &= vals[1](res[vals[0]], vals[2])
        mystr = '{} {} {}'.format(
            vals[0], get_symbol(vals[1].__doc__), vals[2])

    return mystr, res[idx]


def get_symbol(opdoc):
    """
    function to estimate symbol from operator.__doc__

    Parameters
    ----------
    opdoc : str
        operator.__doc__.

    Returns
    -------
    sym : str
        corresponding sym.

    """
    # sym = re.sub(r'.*\w\s?(\S+)\s?\w.*', '\\1', getattr(operator, op).__doc__)
    sym = re.sub(r'.*\w\s?(\S+)\s?\w.*', '\\1', opdoc)
    if re.match('^\\W+$', sym):
        return sym


def plot_effi(data, field, thestyle,
              varx='sel_str', legx='',
              varya='effi', err_varya='err_effi', legya='Observing Efficiency [%]',
              varyb='nsn', err_varyb='err_nsn', legyb='N$_{SN}$'):

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

    plt.show()


def process_season(data, seas, field, norm_factor):
    """
    Function to process a season

    Parameters
    ----------
    data : pandas df
        Data to process.
    seas : int
        season number.
    field : str
        Field of interest.
    norm_factor : float
        normalization factor.

    Returns
    -------
    df_effi : pandas df
        processed data.

    """

    idx = data['season'] == seas
    mysel = data[idx]

    n_nosel = int(len(mysel)/norm_factor)
    print('no sel', n_nosel)
    ra = get_pulls(mysel)
    ra['sel_str'] = 'nosel'
    ra['field'] = field
    ra['season'] = seas
    # dfa = pd.concat((dfa, ra))

    ro = get_nsn(len(mysel), len(mysel), norm_factor)
    ro['sel_str'] = 'nosel'
    ro['field'] = field
    ro['season'] = seas
    # dfb = pd.concat((dfb, ro))
    # get_pulls(mysel)
    for i in range(1, len(sellist)+1):
        # ro = [field, int(seas)]
        mystr, sel = select_str(mysel, sellist[:i])
        rasel = get_pulls(sel)
        rasel['sel_str'] = mystr
        rasel['field'] = field
        rasel['season'] = seas
        ra = pd.concat((ra, rasel))
        rosel = get_nsn(len(sel), len(mysel), norm_factor)
        rosel['sel_str'] = mystr
        rosel['field'] = field
        rosel['season'] = seas
        # dfb = pd.concat((dfb, ro))
        ro = pd.concat((ro, rosel))

    # merge the two Dataframes

    df_effi = ro.merge(ra,
                       left_on=['field', 'season', 'sel_str'],
                       right_on=['field', 'season', 'sel_str'],
                       suffixes=['', ''])

    return df_effi


def process_db(dbDir, dbName, runType, fields,
               norm_factor, zmin=0.01, zmax=1.1):
    """
    Function to process OS data

    Parameters
    ----------
    dbDir : str
        Data dir.
    dbName : str
        OS to process.
    runType : str
        run type.
    fields : list(str)
        List of fields to process.
    norm_factor : float
        normalization factor.
    zmin: float, optional.
        redshift min for data. The default is 0.01.
    zmax: float, optional.
        redshift max for data. The default is 1.11.   

    Returns
    -------
    df_effi : pandas df
        processed data.

    """

    df_effi = pd.DataFrame()

    for field in fields:
        data = load_data(dbDir, dbName, runType, field)
        data = complete_df(data)

        idxz = data['z'] >= zmin
        idxz &= data['z'] <= zmax

        data = data[idxz]
        print(field, len(data), len(data)/norm_factor)

        seasons = data['season'].unique()

        for seas in seasons:
            print('processing', zmin, zmax, seas)
            dd = process_season(data, seas, field, norm_factor)
            df_effi = pd.concat((df_effi, dd))

    df_effi['dbName'] = dbName
    df_effi['zmin'] = np.round(zmin, 2)
    df_effi['zmax'] = np.round(zmax, 2)

    return df_effi


def plots(df_effi, thestyle, field='COSMOS'):
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

    Returns
    -------
    None.

    """

    plot_effi(df_effi, field, thestyle)
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


parser = OptionParser(description='Script to analyze SN selection criteria')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v4.3.1_10yrs',
                  help='OS name [%default]')
parser.add_option('--runType', type=str,
                  default='DDF_spectroz',
                  help='run type [%default]')
parser.add_option('--timescale', type=str,
                  default='season',
                  help='timescale [%default]')
parser.add_option('--seasons', type=str,
                  default='1',
                  help='seasons/years to process [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFS_a,EDFS_b',
                  help='fields to process [%default]')
parser.add_option('--norm_factor', type=float,
                  default=30.,
                  help='normalization factor [%default]')
parser.add_option("--selconfig", type=str,
                  default='G10_JLA', help="sel config name [%default]")
parser.add_option("--zrange", type=int,
                  default=0, help="to process data per zrange [%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
runType = opts.runType
runType = opts.runType
timescale = opts.timescale
seasons = opts.seasons.split(',')
seasons = list(map(int, seasons))
fields = opts.fields.split(',')
norm_factor = opts.norm_factor
selconfig = opts.selconfig
zrange = opts.zrange

# selection criteria
sellist = selection_criteria()[selconfig]

# add criteria
sellist.append(('Nfilt_2', operator.ge, 3, 7))
sellist.append(('Nfilt_5', operator.ge, 2, 7))
sellist.append(('sigmaC', operator.le, 0.04, 7))

print(sellist)
rb = []
# dfa = pd.DataFrame()
# dfb = pd.DataFrame()

zmin = 0.0
zmax = 1.1
deltaz = 1.1

if zrange:
    deltaz = 0.10

zvals = np.arange(zmin, zmax, deltaz)

# zvals[0] += 0.01
print(zvals)

for vv in zvals:
    zmi = vv
    if zmi < 0.001:
        zmi = 0.01
    zma = vv+deltaz
    df_effi = process_db(dbDir, dbName, runType, fields,
                         norm_factor, zmin=zmi, zmax=zma)


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
rorig += ['Nfilt_2 >= 3', 'Nfilt_5 >= 2', 'sigmaC <= 0.04']
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
renew += ['$N_{band}(SNR\geq 2)\geq 3$',
          '$N_{band}(SNR\geq 5)\geq 2$',
          '$\sigma_C \leq 0.04$']

torep = dict(zip(rorig, renew))
for key, vals in torep.items():
    df_effi['sel_str'] = df_effi['sel_str'].str.replace(key, vals)


thestyle = pd.read_csv('plot_style_4_udy.csv')


print(df_effi.columns)

plots(df_effi, thestyle)
