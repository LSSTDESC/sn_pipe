#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 11:01:44 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import matplotlib.pyplot as plt
from sn_plotter_analysis import plt
from optparse import OptionParser
import pandas as pd
import numpy as np


def get_nvisits(grp):
    """
    Function to estimate the number of visits per season

    Parameters
    ----------
    grp : pandas df
        Data to use for the estimation of Nvisits.

    Returns
    -------
    df : pandas df
        original data plus nvisits_season column.

    """

    grp['nvisits_season'] = grp['nvisits_night']*grp['sl']/grp['cad']

    res = grp['nvisits_season'].sum()

    df = pd.DataFrame([res], columns=['nvisits'])

    return df


def plot_summary(res, df_conf):
    """
    Function to plot (DD budget vs zcomp)

    Parameters
    ----------
    res : pandas df
        Data to plot.
    df_conf: pandas df
        config for the plot

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.suptitle('LSST DDF cohesive surveys', weight='bold')
    surveys = res['survey_label'].unique()

    for survey in surveys:
        idx = res['survey_label'] == survey
        idx &= res['zcomp'] >= 0.6
        sel = res[idx]
        resb = sel.groupby(['zcomp']).apply(
            lambda x: get_nvisits(x)).reset_index()
        resb['budget'] = 100. * resb['nvisits']/nvisits_lsst
        idxb = df_conf['label'] == survey
        selp = df_conf[idxb]
        marker = selp['marker'].values[0]
        color = selp['color'].values[0]
        ls = selp['ls'].values[0]
        ax.plot(resb['zcomp'], resb['budget'], linestyle=ls,
                marker=marker, color=color, label=survey,
                mfc='None', markersize=10)

    ax.grid(visible=True)
    ymin, ymax = ax.get_ylim()
    ax.set_xlabel(r'$z_{comp}$')
    ax.set_ylabel(r'DD budget [%]')
    ax.set_xlim([0.6, 0.8])

    ax.plot([0.7]*2, [ymin, ymax], color='b')
    ax.set_ylim([4., 15.])
    ax.legend()


def plot_reqs(data, df_conf, fields=['COSMOS'],
              zcomp=0.70, bands='ugrizy',
              cumsum=False):
    """
    Function to plot nvisits per band


    Parameters
    ----------
    data : pandas df
        Data to process.
    df_conf : pandas df
        Plot config.
    fields : list(str), optional
        List of fields to plot. The default is ['COSMOS'].
    zcomp : float, optional
        zcomp value. The default is 0.70.
    bands : str, optional
        List of the bands to plot. The default is 'ugrizy'.
    cumsum : bool, optional
        To plot with cumsum. The default is False.

    Returns
    -------
    None.

    """

    idx = data['zcomp'] == zcomp

    sel = data[idx]

    nvisits = [360, 139, 212, 288, 450, 360]
    nvisits = list(map(lambda x:  [x], nvisits))

    reqs = dict(zip(bands, nvisits))

    reqs = pd.DataFrame.from_dict(reqs)

    for b in bands:
        reqs['{}_survey'.format(b)] = 10.*reqs[b]

    for field in fields:
        idx = sel['field'] == field
        selb = sel[idx]

        for b in bands:
            fig, ax = plt.subplots(figsize=(15, 9))
            fig.subplots_adjust(right=0.75)
            fig.suptitle(field, fontweight='bold')
            ccol = '{}_season'.format(b)

            for lab in selb['survey_label'].unique():
                idxb = selb['survey_label'] == lab
                selc = selb[idxb]
                idxc = df_conf['label'] == lab
                selp = df_conf[idxc]
                marker = selp['marker'].values[0]
                color = selp['color'].values[0]
                ls = selp['ls'].values[0]
                tp = selc[ccol]
                if cumsum:
                    tp = np.cumsum(selc[ccol])
                ax.plot(selc['year'], tp, linestyle=ls,
                        marker=marker, color=color, label=lab,
                        mfc='None', markersize=10)

            ax.grid(visible=True)
            ax.set_xlabel(r'year')
            vv = '$N_{visits}^{'+b+',season}$'
            if cumsum:
                vv = '$\Sigma N_{visits}^{'+b+',season}$'
                nvisits_survey = reqs['{}_survey'.format(b)].mean()
                ax.plot([0, 11], [nvisits_survey]*2,
                        color='b', linestyle='dashed', lw=2)
            else:
                nvisits_survey = reqs['{}'.format(b)].mean()
                ax.plot([0, 11], [nvisits_survey]*2,
                        color='b', linestyle='dashed', lw=2)
            ax.set_ylabel(r'{}'.format(vv))
            ax.legend(loc='lower center', bbox_to_anchor=(1.15, 0.5),
                      ncol=1, fontsize=12, frameon=False)
            ax.set_xlim([0.8, 10.2])
    plt.tight_layout()


parser = OptionParser(description='Plot LSST DDF cohesive strategy results')

parser.add_option("--nvisits_lsst", type=float,
                  default=2.e6,
                  help="Total number of LSST visits (10 years) [%default]")
parser.add_option("--config", type=str,
                  default='config_lsst_ddf.csv',
                  help="List of files to process [%default]")
parser.add_option("--inputDir", type=str,
                  default='../lsst_ddf_cohesive',
                  help="input file dir[%default]")

opts, args = parser.parse_args()

nvisits_lsst = opts.nvisits_lsst
config = opts.config
inputDir = opts.inputDir

# load config file
df_conf = pd.read_csv(config)

# load files
res = pd.DataFrame()

for i, row in df_conf.iterrows():
    fName = '{}/{}.csv'.format(inputDir, row['fName'])
    df_ = pd.read_csv(fName)
    df_['survey_label'] = row['label']
    res = pd.concat((res, df_))


# summary plot
plot_summary(res, df_conf)

# requirements checks

# estimate the number of visits per season
bands = list('ugrizy')
bbou = list(map(lambda x:  x+'_season', bands))

for b in bands:
    ccol = '{}_season'.format(b)
    res[ccol] = res[b]*res['sl']/res['cad']

print(res[bbou])

print(res.columns)
plot_reqs(res, df_conf, fields=['XMM_LSS'], cumsum=False)
plt.show()
