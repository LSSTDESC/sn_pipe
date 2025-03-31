#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:34:09 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from sn_plotter_metrics.plot4metric import multiplot_dist
from sn_plotter_analysis.sn_analyser_ddf import get_nsn, get_val
from sn_plotter_analysis.sn_analyser_summary import load_data
from optparse import OptionParser
import matplotlib.pyplot as plt


def plot_nsn_dist(data, fields=['COSMOS'],
                  timescale='year', norm_factor=30,
                  nside=128, zmin=-1, sigmaC=-1):
    """
    Function to plot ns vs pixel dist

    Parameters
    ----------
    data : pandas df
        Data to plot.
    fields : list(str), optional
        List of fields to plot. The default is ['COSMOS'].
    timescale : str, optional
        Time scale for the plot. The default is 'year'.
    norm_factor : float, optional
        simulation norm factor. The default is 30.
    nside : int, optional
        healpix nside parameter. The default is 128.

    Returns
    -------
    None.

    """

    idx = data['field'].isin(fields)

    ylab_add = ''

    if zmin > -1:
        idx &= data['zmeas'] >= zmin
        ylab_add = '$z \geq $'+'{}'.format(zmin)

    if sigmaC > -1:
        idx &= data['sigmaC'] <= sigmaC
        ylab_add += ', $\sigma_C \leq $'+'{}'.format(sigmaC)

    sel = data[idx]

    ylab = '$N_{SN}$'
    if ylab_add != '':
        ylab += '({})'.format(ylab_add)

    cols = [timescale, 'dbName', 'healpixID', 'pixRA', 'pixDec']
    nsn = get_nsn(sel, norm_factor, nside, cols=cols)

    print(nsn)

    multiplot_dist(nsn, yvar='nsn', yleg=ylab, timescale='year')


parser = OptionParser(description='Script to plot nsn vs pixel dist')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v4.3.1_10yrs',
                  help='OS to process[%default]')
parser.add_option('--norm_factor', type=int,
                  default=30,
                  help='normalization factor [%default]')
parser.add_option('--runType', type=str,
                  default='spectroz',
                  help='run type  [%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='timescale of the files to process [%default]')
parser.add_option('--timeslots', type=str,
                  default='1-10',
                  help='time slot (season or year) to process [%default]')
parser.add_option('--dataType', type=str,
                  default='DataFrame',
                  help='data type [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,XMM-LSS,CDFS,ELAISS1,EDFS_a,EDFS_b',
                  help='data type [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
norm_factor = opts.norm_factor
runType = opts.runType
timeslots = get_val(opts.timeslots)
timescale = opts.timescale
dataType = opts.dataType
fields = opts.fields.split(',')

fieldType = 'DDF'

df = load_data(dbDir, dbName, dataType, runType,
               timescale, timeslots, fieldType)


plot_nsn_dist(df, fields=fields,
              timescale=timescale, norm_factor=norm_factor,
              nside=128)

plot_nsn_dist(df, fields=fields,
              timescale=timescale, norm_factor=norm_factor,
              nside=128, zmin=0.8)

plot_nsn_dist(df, fields=fields,
              timescale=timescale, norm_factor=norm_factor,
              nside=128, zmin=0.8, sigmaC=0.04)
plt.show()
