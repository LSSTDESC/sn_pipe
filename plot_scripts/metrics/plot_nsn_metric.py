#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 09:54:50 2023

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
from sn_plotter_metrics.utils import MetricValues
from sn_plotter_metrics.utils import dumpcsv_medcad, get_dist
from sn_plotter_metrics.plot4metric import plot_vs_OS_dual, plot_series_fields
from sn_plotter_metrics.plot4metric import plot_field, plot_pixels
from sn_plotter_metrics.plot4metric import plotMollview, plotMollview_seasons
from sn_plotter_metrics.plot4metric import plot_xy
import numpy as np


def zcomp_frac(grp, frac=0.95):
    """
    Fonction to estimate the redshift completeness
    from ns distribution over pixels

    Parameters
    ----------
    grp : pandas df
        data to process.
    frac : float, optional
        NSN fraction to estimate zcomp. The default is 0.95.

    Returns
    -------
    pandas df
        df with results.

    """

    nsn = np.sum(grp['nsn'])
    selfi = get_dist(grp)
    nmax = np.max(selfi['nsn'])

    idx = selfi['nsn'] <= frac*nmax

    res = selfi[idx]
    if len(res) > 0:
        dist_cut = np.min(res['dist'])
        idd = selfi['dist'] <= dist_cut
        zcomp = np.median(selfi[idd]['zcomp'])
    else:
        zcomp = np.median(selfi['zcomp'])

    """
    print('zcomp_frac', grp.name, dist_cut,
              nmax, selfi[['zcomp', 'nsn', 'dist']])
    fig, ax = plt.subplots()
    ax.plot(selfi['dist'], selfi['nsn'], 'ko')
    plt.show()
    """

    return pd.DataFrame({'nsn': [nsn], 'zcomp': [zcomp]})


def zcomp_weighted(grp):
    """
    Function to estimated weighted (by nsn) zcomp

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    pandas df
        with weighted zcomp.

    """

    nsn = np.sum(grp['nsn'])
    zcomp = np.sum(grp['nsn']*grp['zcomp'])/nsn

    return pd.DataFrame({'nsn': [nsn], 'zcomp': [zcomp]})


def zcomp_cumsum(grp, frac=0.95):
    """
    Function to estimated zcomp from nsn cumulative dist

    Parameters
    ----------
    grp : pandas df
        data to process.
    frac : float, optional
        frac for cumulative. The default is 0.95.

    Returns
    -------
    pandas df
        with zcomp from cumulative.

    """

    xvar, yvar = 'zcomp', 'nsn'
    nsn = np.sum(grp[yvar])
    from scipy.interpolate import interp1d

    selp = grp.sort_values(by=[xvar], ascending=False)
    print('ee', grp.name, selp[[xvar, yvar]])
    if len(selp) >= 2:
        cumulnorm = np.cumsum(selp[yvar])/np.sum(selp[yvar])
        interp = interp1d(
            cumulnorm, selp[xvar], bounds_error=False, fill_value=0.)
        zcompl = interp(frac)
        io = selp[xvar] >= zcompl
        zcomp = np.median(selp[io][xvar])
    else:
        zcomp = np.median(selp[xvar])

    return pd.DataFrame({'nsn': [nsn], 'zcomp': [zcomp]})


def zcomp(metricTot):

    summary_season = zcomp_field_season(metricTot)

    summary_field = zcomp_field(summary_season)

    summary = zcomp_med(summary_field)

    return summary


def zcomp_field_season(metricTot):

    summary_season = \
        metricPlot.groupby(['dbName', 'fieldname', 'season']).apply(
            lambda x: zcomp_frac(x)).reset_index()

    return summary_season


def zcomp_field(summary_season):

    summary_field = summary_season.groupby(['dbName', 'fieldname']).apply(
        lambda x: zcomp_cumsum(x)).reset_index()

    return summary_field


def zcomp_med(summary):

    summary_db = summary.groupby(['dbName']).agg({'nsn': 'sum',
                                                  'zcomp': 'median',
                                                  }).reset_index()

    return summary_db


def merge(dfa, dfb, left_on=['dbName'], right_on=['dbName']):

    dfa = dfa.merge(dfb, left_on=left_on, right_on=right_on)

    return dfa


def add_dist(grp):

    grp = get_dist(grp)

    return grp


parser = OptionParser(
    description='Display (NSN,zlim) metric results')
parser.add_option("--dirFile", type="str",
                  default='/sps/lsst/users/gris/MetricOutput',
                  help="file directory [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type - DD, WFD, Fake [%default]")
parser.add_option("--dbList", type="str",
                  default='plot_scripts/cadenceCustomize_fbs14.csv',
                  help="list of cadences to display[%default]")
parser.add_option("--fieldNames", type="str",
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                  help="fields to process - for DD only[%default]")
parser.add_option("--metric", type="str", default='NSNY',
                  help="metric name [%default]")
parser.add_option("--prefix_csv", type="str", default='metric_summary_DD',
                  help="prefix for csv output files [%default]")
parser.add_option("--plot_level", type='str', default='global',
                  help="type of plot to perform [%default]")
parser.add_option("--seasons", type='str', default='3',
                  help="seasons to display (pixel plot) [%default]")


opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
nside = opts.nside
fieldType = opts.fieldType
metricName = opts.metric
fieldNames = opts.fieldNames.split(',')
prefix_csv = opts.prefix_csv
plot_level = opts.plot_level
seasons = opts.seasons.split(',')
seasons = list(map(int, seasons))

# Loading input file with the list of cadences and display features
filename = opts.dbList

forPlot = pd.read_csv(filename, comment='#')

print(forPlot)

# load metricValues corresponding to filename

metricTot = MetricValues(dirFile, forPlot['dbName'].to_list(), metricName,
                         fieldType, fieldNames, nside).data

# clean and merge with plotter char
var = 'nsn'
varz = 'zcomp'
idx = metricTot[var] > 0.
idx &= metricTot[varz] > 0.

metricPlot = metricTot[idx]
metricPlot = metricPlot.merge(forPlot, left_on=['dbName'], right_on=['dbName'])

vary = ['nsn', 'zcomp']
legy = ['N$_{SN}^{z \leq z_{complete}}$', '$z_{complete}$']
ls = ['dotted', 'dotted']
ls = ['dashed', 'dashed']
color = ['k', 'k']
mec = ['r', 'r']

# now plot according to the plot_level
if plot_level == 'global':
    if fieldType == 'DD':
        idx = metricPlot['fieldname'].isin(fieldNames)
        summary = zcomp(metricPlot[idx])
        title = opts.fieldNames
    else:
        summary = zcomp_med(metricPlot)
        title = 'WFD survey'
    summary = summary.sort_values(by=['nsn'])
    summary = merge(summary, forPlot)
    plot_vs_OS_dual(summary, vary=vary, title=title,
                    legy=legy, ls=ls, color=color, mec=mec)

if plot_level == 'fields':
    if fieldType == 'DD':
        summary_season = zcomp_field_season(metricTot)
        summary_field = zcomp_field(summary_season)
        summary_field['field'] = summary_field['fieldname']
        summary_field = merge(summary_field, forPlot)
        plot_series_fields(summary_field, what=vary, leg=legy)

if plot_level == 'season':
    if fieldType == 'DD':
        summary_season = zcomp_field_season(metricTot)
        summary_season = merge(summary_season, forPlot)

        for field in fieldNames:
            idx = summary_season['fieldname'] == field
            sel = summary_season[idx]
            plot_field(sel, yvars=vary, ylab=legy, title=field)

if plot_level == 'pixels':
    import matplotlib.pyplot as plt
    if fieldType == 'DD':
        res = metricTot.groupby(['dbName', 'fieldname', 'season']).apply(
            lambda x: add_dist(x))
        print(res.columns)
        dbName = 'draft_connected_v2.99_10yrs'
        field = 'COSMOS'
        season = 3
        idx = res['dbName'] == dbName
        idx &= res['fieldname'] == field
        idx &= res['season'] == season
        sel = res[idx]
        print('total number of SN', np.sum(sel['nsn']))
        fig, ax = plt.subplots(figsize=(14, 8))
        figtitle = '{} - {} \n season {}'.format(dbName, field, season)
        yleg = '$N_{SN}^{z \leq z_{complete}}$/pixel(0.21deg$^2$)'
        plot_pixels(sel, yvar='nsn', yleg=yleg, fig=fig, ax=ax,
                    figtitle=figtitle, marker='s', color='k', showIt=False)
        axb = ax.twinx()
        plot_pixels(sel, yvar='cadence', fig=fig, ax=axb, figtitle=figtitle,
                    marker='s', color='b', showIt=True, ls='dotted')
        xmin = np.min(sel['nsn'])
        xmax = np.max(sel['nsn'])
        tit = 'DDF'
        plotMollview(nside, sel, 'nsn', tit, np.sum, xmin, xmax)
        plt.show()
    else:
        dbName = 'draft_connected_v2.99_10yrs'
        idx = metricTot['dbName'] == dbName
        idx &= metricTot['season'] < 11
        idx &= metricTot['zcomp'] > 0.
        sel = metricTot[idx]
        sel = sel.sort_values(by=['season'])
        plotMollview_seasons(nside, sel, dbName, 'nsn',
                             'N$_{SN}^{z \leq z_{complete}}$', np.sum, seasons)
        plotMollview_seasons(nside, sel, dbName, 'zcomp',
                             '$z_{complete}$', np.median, seasons)
        plotMollview_seasons(nside, sel, dbName, 'cadence',
                             'cadence [day]', np.median, seasons)

        plot_xy(sel,  yvar='zcomp',
                yleg='$z_{complete}^{mean}$', seasons=seasons)
        plot_xy(sel, yvar='nsn', yleg='$\Sigma N_{SN}^{z \leq z_{complete}}$',
                seasons=seasons, op='sum')
        plt.show()
