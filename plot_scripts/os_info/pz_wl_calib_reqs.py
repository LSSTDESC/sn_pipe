#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:08:52 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
from sn_plotter_os_info.os_info_util import plot_summary
import numpy as np
from sn_plotter_analysis.sn_analyser_tools import clean_level


def plot_nvisits_all(data, dfconfig, fields, df_calib=pd.DataFrame(),
                     bands='ugrizy', calib_label=''):
    """
    Function to plot the total number of visits

    Parameters
    ----------
    data : pandas df
        Data to process.
    dfconfig : pandas df
        config file.
    fields : list(str)
        Fields to consider.
    df_calib : pandas df, optional
        calib reqs. The default is pd.DataFrame().
    bands : atr, optional
        list of bands to consider. The default is 'ugrizy'.
    calib_label : str, optional
        calibration label. The default is ''.

    Returns
    -------
    None.

    """
    print('alors', data.columns)
    print(df_calib.columns)

    for field in fields:
        for b in bands:
            plot_nvisits(data, df_config, b, field=field,
                         df_calib=df_calib, cumsum=True, calib_label=calib_label)


def plot_nvisits(data, df_config, b, field='DD:XMM_LSS',
                 df_calib=pd.DataFrame(), cumsum=False, calib_label=''):
    """
    Function to plot nvisits for a field and a band

    Parameters
    ----------
    data : pandas df
        Data to process.
    df_config : pandas df
        config for the plot.
    b : str
        band to consider.
    field : str, optional
        Field to consider. The default is 'DD:XMM_LSS'.
    df_calib : pandas df, optional
        calib reqs. The default is pd.DataFrame().
    cumsum : bool, optional
        To use the cumsum. The default is False.
    calib_label : str, optional
        calib label. The default is ''.

    Returns
    -------
    None.

    """

    bband = '$'+b+'$-band'
    laby = '$\Sigma N_{visits}$ - ' + bband
    thevar = 'Nvisits_{}'.format(b)
    ax = plot_summary(data, field=field,
                      varx='year', labx='year',
                      vary=thevar, laby=laby,
                      figtit=field,
                      df_config=df_config, cumsum=cumsum)

    tp = df_calib[b]
    if cumsum:
        tp = np.cumsum(df_calib[b])
    ax.plot(df_calib['year'], tp, color='k', lw=3, label=calib_label)

    ax.legend(loc='upper center',
              bbox_to_anchor=(1.20, 0.7),
              ncol=1, fontsize=12, frameon=False)


def get_reqs(bands='ugrizy',
             nvisits=[48, 72, 184, 184, 152, 160],
             years=range(1, 11)):
    """
    Function to convert a list of visits per year to a df.

    Parameters
    ----------
    bands : list(str), optional
        colnames for output df. The default is 'ugrizy'.
    nvisits : list(int), optional
        List of visits. The default is [48, 72, 184, 184, 152, 160].
    years : list(int), optional
        List of years. The default is range(1, 11).

    Returns
    -------
    df_calib : pandas df
        Outout df.

    """

    nvisits = map(lambda x: [x], nvisits)
    dd = dict(zip(bands, nvisits))
    df_a = pd.DataFrame.from_dict(dd)

    ddb = {}
    # years = map(lambda x: [x], years)
    ddb['year'] = years
    df_b = pd.DataFrame.from_dict(ddb)

    df_calib = df_a.merge(df_b, how='cross')

    return df_calib


def ana_reqs(data, reqs,
             req_name='WL reqs',
             bands='ugrizy',
             fields=['DD:COSMOS']):
    """
    Function to analyse reqs wrt OS results

    Parameters
    ----------
    data : pandas df
        Data to process.
    reqs : pandas df
        Requirements.
    req_name : str, optional
        Req name (for the plot). The default is 'WL reqs'.
    bands : list(str), optional
        Bands to consider. The default is 'ugrizy'.
    fields: list(str), optional
      list of DDF to analyze. The default is ['DD:COSMOS'].

    Returns
    -------
    None.

    """

    idx = data['target_name'].isin(fields)

    data = pd.DataFrame(data[idx])

    # take the cumulative
    for b in bands:
        reqs = reqs.sort_values(by=['year'])
        reqs['Nvisits_sum_{}'.format(b)] = np.cumsum(
            reqs['{}'.format(b)])

    data = data.sort_values(by=['year'])
    data = data.groupby(['dbName', 'target_name']
                        ).apply(lambda x: get_cumsum(x, bands), include_groups=False).reset_index()
    cols = ['year', 'Nvisits_y', 'Nvisits_sum_y']

    data = clean_level(data)
    # select years 5, 7 and 10
    years = [1, 4, 8]

    idxa = reqs['year'].isin(years)

    reqs_sel = pd.DataFrame(reqs[idxa])

    idxb = data['year'].isin(years)

    data_sel = pd.DataFrame(data[idxb])

    data_m = data_sel.merge(reqs_sel, left_on=['year'], right_on=[
                            'year'], suffixes=['', '_ref'])
    for b in bands:
        data_m['diff_sum_{}'.format(b)] = data_m['Nvisits_sum_{}'.format(
            b)]-data_m['Nvisits_sum_{}_ref'.format(b)]

    return data_m


def ana_reqs_m5(data, reqs,
                req_name='PZ reqs',
                bands='ugrizy'):
    """
    Function to analyse reqs wrt OS results

    Parameters
    ----------
    data : pandas df
        Data to process.
    reqs : pandas df
        Requirements.
    req_name : str, optional
        Req name (for the plot). The default is 'WL reqs'.
    bands : list(str), optional
        Bands to consider. The default is 'ugrizy'.

    Returns
    -------
    None.

    """

    data = clean_level(data)
    data = data.sort_values(by=['year'])
    datab = data.groupby(['dbName', 'target_name', 'field']
                         ).apply(lambda x: get_cum_m5(x, bands), include_groups=False).reset_index()
    cols = ['year', 'target_name', 'field',
            'fiveSigmaDepth_r', 'fiveSigmaDepth_z']
    # print(datab[cols])

    data = clean_level(data)

    data_m = datab.merge(reqs, left_on=['year'], right_on=[
        'year'], suffixes=['', '_ref'])

    # print(data_m)
    thevar = 'fiveSigmaDepth'
    for b in bands:
        data_m['diff_{}_{}'.format(thevar, b)] = data_m['{}_{}'.format(thevar,
                                                                       b)]-data_m['{}_{}_ref'.format(thevar, b)]

    # print(data_m)

    print(data_m.columns)
    return data_m
    # select ddf_ocean_ocean6_v4.3.5_10yrs
    idx = data_m['dbName'] == 'ddf_ocean_ocean6_v4.3.5_10yrs'

    sel_m = data_m[idx]

    plot_calib(sel_m, bands=bands, req_name=req_name,
               prefix='diff_fiveSigmaDepth',
               laby='$\Delta m_5=m_5^{obs}-m_5^{req}$')


def get_cum_m5(grp, bands, years=[[1, 1], [2, 10]]):
    """
    get the m5_cumul for a set of years

    Parameters
    ----------
    grp : pandas df
        Data to process.
    bands : list(str)
        Bands to consider.
    years : list(list(int)), optional
        Year ranges [ymin,ymax] to consider for the estimation.
        The default is [[1, 1], [2, 10]].

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """

    df = pd.DataFrame()
    for yy in years:
        y_min = yy[0]
        y_max = yy[1]

        idx = grp['year'] >= y_min
        idx &= grp['year'] <= y_max

        sel = grp[idx]

        dd = {}
        for b in bands:
            vv = 'fiveSigmaDepth_{}'.format(b)
            tt = 1.25*np.log10(np.sum(10**(0.8*sel[vv])))
            dd[vv] = [tt]

        dd['year'] = y_max
        dfa = pd.DataFrame.from_dict(dd)
        df = pd.concat((df, dfa))

    return df


def get_cumsum(grp, bands):
    """
    Function to estimate the cumulative sum on Nvisits

    Parameters
    ----------
    grp : pandas df
        Data to process.
    bands : list(str)
        Filters to consider.

    Returns
    -------
    grp : pandas df
        Original data plus cumsum.

    """

    grp = grp.sort_values(by=['year'])
    for b in bands:
        grp['Nvisits_sum_{}'.format(b)] = np.cumsum(
            grp['Nvisits_{}'.format(b)])

    return grp


def plot_calib(data, bands, req_name,
               prefix='diff_sum',
               laby='$\Delta N_{visits}=N_{visits}^{obs}-N_{visits}^{req}$'):
    """
    Function to plot calb req results

    Parameters
    ----------
    data : pandas df
        Data to process.
    bands : list(str)
        Filters to consider.
    req_name : str
        req name (for the plot).
    prefix : str, optional
        What to plot. The default is 'diff_sum'.
    Returns
    -------
    None.

    """

    fields = ['COSMOS', 'ECDFS',
              'XMM_LSS', 'ELAISS1',
              'EDFS_a', 'EDFS_b']

    colors = ['r', 'g', 'r', 'm', 'k', 'b']
    linestyles = ['solid', 'dashed', 'dotted', 'solid', 'dashed', 'dotted']
    markers = ['o', 'h', 's', 'v', '^', 'P']

    for b in bands:
        fig, ax = plt.subplots(figsize=(16, 8))
        # fig.suptitle(figtit)
        figtit = req_name
        figtit += ' \n {}-band'.format(b)
        fig.suptitle(figtit)
        fig.subplots_adjust(right=0.75)

        target = data['target_name'].unique()

        for tt in target:
            idx = data['target_name'] == tt
            sel = data[idx]
            ii = fields.index(tt)
            ax.plot(sel['year'], sel['{}_{}'.format(prefix, b)],
                    marker=markers[ii],
                    color=colors[ii],
                    linestyle=linestyles[ii], label=tt, markersize=8, mfc='None')

        ax.grid(visible=True)
        ax.set_xlabel(r'year')
        ax.set_ylabel(r'{}'.format(laby))

        ax.legend(loc='upper center',
                  bbox_to_anchor=(1.20, 0.7),
                  ncol=1, fontsize=12, frameon=False)


def plot_calib_db(data, req_name, conf_df, field='COSMOS',
                  prefix='diff_sum',
                  laby='$\Delta N_{visits}=N_{visits}^{obs}-N_{visits}^{req}$',
                  bands='ugrizy'):
    """
    Function to plot calib req results

    Parameters
    ----------
    data : pandas df
        Data to process.
    req_name : str
        tag req name.
    conf_df : pandas df
        config file.
    field : str, optional
        field to plot. The default is 'COSMOS'.
    prefix : str, optional
        var prefix to plot. The default is 'diff_sum'.
    laby : str, optional
        y-axis label. The default is '$\Delta N_{visits}=N_{visits}^{obs}-N_{visits}^{req}$'.
    bands : str, optional
        List of filters to consider. The default is 'ugrizy'.

    Returns
    -------
    None.

    """

    idx = data['field'] == field
    sel = data[idx]

    dbNames = sel['dbName'].unique()

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.subplots_adjust(right=0.75)
    figtit = '{} - {}'.format(field, req_name)
    fig.suptitle(figtit)
    for dbName in dbNames:
        ido = sel['dbName'] == dbName
        selb = sel[ido]
        idc = conf_df['dbName'] == dbName
        selp = conf_df[idc]
        ls = selp['ls'].values[0]
        marker = selp['marker'].values[0]
        color = selp['color'].values[0]
        dbNameb = selp['dbName_plot'].values[0]

        for b in bands:
            vrtp = '{}_{}'.format(prefix, b)
            if b == 'u':
                ax.plot(selb['year'], selb[vrtp],
                        ls=ls, marker=marker, color=color, mfc='None',
                        label=dbNameb)
            else:
                ax.plot(selb['year'], selb[vrtp],
                        ls=ls, marker=marker, color=color, mfc='None')

    ax.grid(visible=True)
    ax.set_xlabel(r'year')
    ax.set_ylabel(r'{}'.format(laby))
    ax.set_xlim([0.9, 8.1])
    ax.legend(loc='upper center',
              bbox_to_anchor=(1.20, 0.7),
              ncol=1, fontsize=12, frameon=False)


def print_csv_reqs(res, tit, prefix='diff_sum',):

    df = pd.DataFrame()

    for b in 'ugrizy':
        idx = res['{}_{}'.format(prefix, b)] < 0
        sel = res[idx]
        sel['filter'] = b
        df = pd.concat((df, sel))

    dbNames = df['dbName'].unique()

    print(df[['dbName', 'field', 'filter', 'year']])

    idxx = df['dbName'] == 'ddf_acc_early_v5.0.0_10yrs'
    sel = df[idxx]
    print('ooooo', sel)

    fields = df['field'].unique()
    bands = 'ugrizy'

    dfb = df.groupby(['field', 'dbName', 'filter']).apply(
        lambda x: get_years(x), include_groups=False).reset_index()

    dfc = dfb.groupby(['field', 'dbName']).apply(
        lambda x: get_format(x), include_groups=False).reset_index()

    dfc = clean_level(dfc)

    tit_csv = '_'.join(tit.split(' '))
    tit_csv = 'constraints_{}.csv'.format(tit_csv.lower())
    dfc.to_csv(tit_csv, index=False)


def get_years(grp):

    # grab the years
    ll = grp['year'].to_list()
    # transform to str
    ll = list(map(str, ll))

    res = [','.join(ll)]

    df = pd.DataFrame(res, columns=['years'])

    return df


def get_format(grp):

    bands = grp['filter'].to_list()
    years = grp['years'].to_list()

    bb = '/'.join(bands)
    yy = '/'.join(years)

    dd = {}
    dd['filters'] = [bb]
    dd['years'] = [yy]

    res = pd.DataFrame.from_dict(dd)

    return res


parser = OptionParser(description='Script to plot calib reqs from PZ and WL')

parser.add_option('--config', type=str,
                  default='config_ana_selplot.csv',
                  help='OS DD list[%default]')
parser.add_option("--dirFile", type="str",
                  default='../nvisits_m5',
                  help="file directory [%default]")
parser.add_option("--fields", type="str", default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFS_a,EDFS_b',
                  help="fields to process [%default]")
parser.add_option("--plots", type="str", default='plot_global_wl,plot_global_agn,summary_reqs_wl,summary_reqs_agn,summary_reqs_pz',
                  help="plots to draw [%default]")

opts, args = parser.parse_args()

dirFile = opts.dirFile
config = opts.config
fields = opts.fields.split(',')
plots = opts.plots.split(',')

# load config
df_config = pd.read_csv(config, comment='#')

# load the data
data = pd.DataFrame()
for i, row in df_config.iterrows():
    df = pd.read_hdf('{}/{}.hdf5'.format(dirFile, row['dbName']))
    data = pd.concat((data, df))

# WL and requirements per year
bands = list('ugrizy')
nvisits_wl = [48, 72, 184, 184, 152, 160]
nvisits_agn = [360, 90, 90, 270, 450, 360]
df_wl = get_reqs(bands, nvisits_wl)
df_agn = get_reqs(bands, nvisits_agn)

if 'plot_global_wl' in plots:
    plot_nvisits_all(data, df_config, fields,
                     df_wl, calib_label='WL calib reqs')

if 'plot_global_agn' in plots:
    plot_nvisits_all(data, df_config, fields, df_agn, calib_label='AGN reqs')

if 'summary_reqs_wl' in plots:
    res = ana_reqs(data, df_wl, req_name='WL reqs.', fields=fields)
    plot_calib_db(res, 'WL reqs.', df_config, field='XMM-LSS')
    print_csv_reqs(res, 'WL reqs')

if 'summary_reqs_agn' in plots:
    res = ana_reqs(data, df_agn, req_name='AGN reqs.', fields=fields)
    plot_calib_db(res, 'AGN reqs.', df_config)
    print_csv_reqs(res, 'AGN reqs')

# pz requirements


pz_y1 = [26.7, 27.0, 26.2, 25.8, 25.6, 24.7]
pz_y10 = [27.8, 28.1, 27.8, 27.6, 27.2, 26.5]

bands = list(map(lambda x: 'fiveSigmaDepth_' + x, bands))
req_pz_y1 = get_reqs(bands, pz_y1, years=[1])
req_pz_y10 = get_reqs(bands, pz_y10, years=[10])
req_pz = pd.concat((req_pz_y1, req_pz_y10))

if 'summary_reqs_pz' in plots:
    res = ana_reqs_m5(data, req_pz, req_name='PZ reqs.')
    plot_calib_db(res, 'PZ reqs.', df_config, field='XMM-LSS',
                  laby='$\Delta m_5=m_5^{obs}-m_5^{req}$',
                  prefix='diff_fiveSigmaDepth')
    print_csv_reqs(res, 'PZ reqs', prefix='diff_fiveSigmaDepth')

plt.show()
