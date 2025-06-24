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
    ax = plot_summary(data, field=field,
                      varx='year', labx='year',
                      vary='Nvisits_{}'.format(b), laby=laby,
                      figtit=field,
                      df_config=df_config, cumsum=cumsum)

    tp = df_calib[b]
    if cumsum:
        tp = np.cumsum(df_calib[b])
    ax.plot(df_calib['year'], tp, color='k', lw=3, label=calib_label)

    ax.legend(loc='upper center',
              bbox_to_anchor=(1.20, 0.7),
              ncol=1, fontsize=12, frameon=False)


def get_reqs(nvisits=[48, 72, 184, 184, 152, 160]):
    """
    Function to convert a list of visits per year to a df.

    Parameters
    ----------
    nvisits : list(int), optional
        Number of visits per band. The default is [48, 72, 184, 184, 152, 160].

    Returns
    -------
    df_calib : pandas df
        output data.

    """
    bands = 'ugrizy'

    nvisits = map(lambda x: [x], nvisits)
    dd = dict(zip(bands, nvisits))
    df_a = pd.DataFrame.from_dict(dd)

    ddb = {}
    years = range(1, 11)
    # years = map(lambda x: [x], years)
    ddb['year'] = years
    df_b = pd.DataFrame.from_dict(ddb)

    df_calib = df_a.merge(df_b, how='cross')

    return df_calib


def ana_reqs(data, reqs, bands='ugrizy', dbName='ddf_ocean_ocean6_v4.3.5_10yrs'):
    """
    Function to analyse reqs wrt OS results

    Parameters
    ----------
    data : pandas df
        Data to process.
    reqs : pandas df
        Requirements.
    bands : list(str), optional
        Bands to consider. The default is 'ugrizy'.
    dbName : str, optional
        OS of interest. The default is 'ddf_ocean_ocean6_v4.3.5_10yrs'.

    Returns
    -------
    None.

    """

    idx = data['dbName'] == dbName
    data = pd.DataFrame(data[idx])

    # take the cumulative
    for b in bands:
        reqs = reqs.sort_values(by=['year'])
        reqs['Nvisits_sum_{}'.format(b)] = np.cumsum(
            reqs['{}'.format(b)])
        """
        data['Nvisits_sum_{}'.format(b)] = np.cumsum(
            data['Nvisits_{}'.format(b)])
        """

    print('allo', reqs)
    data = clean_level(data)
    data = data.sort_values(by=['year'])
    data = data.groupby(['dbName', 'target_name']
                        ).apply(lambda x: get_cumsum(x, bands), include_groups=False).reset_index()
    cols = ['year', 'Nvisits_y', 'Nvisits_sum_y']
    print(data[cols])

    data = clean_level(data)
    # select years 5, 7 and 10
    years = [5, 8, 10]

    idxa = reqs['year'].isin(years)

    reqs_sel = pd.DataFrame(reqs[idxa])

    idxb = data['year'].isin(years)

    data_sel = pd.DataFrame(data[idxb])

    print(data_sel)

    data_m = data_sel.merge(reqs_sel, left_on=['year'], right_on=[
                            'year'], suffixes=['', '_ref'])

    print(data_m)
    for b in bands:
        data_m['diff_sum_{}'.format(b)] = data_m['Nvisits_sum_{}'.format(
            b)]-data_m['Nvisits_sum_{}_ref'.format(b)]

    print(data_m)

    # select ddf_ocean_ocean6_v4.3.5_10yrs
    idx = data_m['dbName'] == 'ddf_ocean_ocean6_v4.3.5_10yrs'

    sel_m = data_m[idx]

    plot_calib(sel_m, bands=bands)


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


def plot_calib(data, bands):
    """
    Function to plot calb req results

    Parameters
    ----------
    data : pandas df
        Data to process.
    bands : list(str)
        Filters to consider.

    Returns
    -------
    None.

    """

    fields = ['DD:COSMOS', 'DD:ECDFS',
              'DD:XMM_LSS', 'DD:ELAISS1',
              'DD:EDFS_a', 'DD:EDFS_b']

    colors = ['r', 'g', 'r', 'm', 'k', 'b']
    linestyles = ['solid', 'dashed', 'dotted', 'solid', 'dashed', 'dotted']
    markers = ['o', 'h', 's', 'v', '^', 'P']

    for b in bands:
        fig, ax = plt.subplots(figsize=(16, 8))
        # fig.suptitle(figtit)
        fig.suptitle('{}-band'.format(b))
        fig.subplots_adjust(right=0.75)

        target = data['target_name'].unique()

        for tt in target:
            idx = data['target_name'] == tt
            sel = data[idx]
            ii = fields.index(tt)
            ax.plot(sel['year'], sel['diff_sum_{}'.format(b)],
                    marker=markers[ii],
                    color=colors[ii],
                    linestyle=linestyles[ii], label=tt, markersize=8, mfc='None')

        ax.grid(visible=True)

        ax.legend(loc='upper center',
                  bbox_to_anchor=(1.20, 0.7),
                  ncol=1, fontsize=12, frameon=False)


parser = OptionParser(description='Script to plot calib reqs from PZ and WL')

parser.add_option('--config', type=str,
                  default='config_ana_selplot.csv',
                  help='OS DD list[%default]')
parser.add_option("--dirFile", type="str",
                  default='../nvisits_m5',
                  help="file directory [%default]")
parser.add_option("--fields", type="str", default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                  help="fields to process [%default]")

opts, args = parser.parse_args()

dirFile = opts.dirFile
config = opts.config
fields = opts.fields.split(',')

# load config
df_config = pd.read_csv(config, comment='#')

# load the data
data = pd.DataFrame()
for i, row in df_config.iterrows():
    df = pd.read_hdf('{}/{}.hdf5'.format(dirFile, row['dbName']))
    data = pd.concat((data, df))

# WL and requirements per year
nvisits_wl = [48, 72, 184, 184, 152, 160]
nvisits_agn = [360, 90, 90, 270, 450, 360]
df_wl = get_reqs(nvisits_wl)
df_agn = get_reqs(nvisits_agn)

idx = data['dbName'] == 'ddf_ocean_ocean6_v4.3.5_10yrs'
plot_nvisits_all(data[idx], df_config, fields,
                 df_agn, calib_label='WL calib reqs')
"""
plot_nvisits_all(data, df_config, fields, df_agn, calib_label='AGN reqs')
"""

ana_reqs(data, df_agn)
plt.show()
