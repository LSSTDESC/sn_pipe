#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:50:54 2023

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
from sn_analysis import plt
from sn_tools.sn_io import checkDir
import os


def get_val(var):
    """
    Function to grab values from parser

    Parameters
    ----------
    var : str
        var to process.

    Returns
    -------
    var : list(int)
        Result.

    """
    if '-' in var:
        seas_spl = var.split('-')
        seas_min = int(seas_spl[0])
        seas_max = int(seas_spl[1])
        var = range(seas_min, seas_max+1)
    else:
        var = var.split(',')
        var = list(map(int, var))

    return var


parser = OptionParser(description='Script to analyze SN prod after selection')

parser.add_option('--config', type=str,
                  default='input/plots/config_ana.csv',
                  help='OS list[%default]')
parser.add_option('--dbDir', type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--norm_factor', type=int,
                  default=10,
                  help='Normalization factor [%default]')
parser.add_option('--runType', type=str,
                  default='spectroz_nosat',
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
parser.add_option('--plots', type=str,
                  default='summary,mollweid,density,density_season',
                  help='plots to draw [%default]')
parser.add_option('--outDir', type=str,
                  default='../sn_wfd',
                  help='output dir [%default]')
parser.add_option('--outName', type=str,
                  default='nsn_wfd.hdf5',
                  help='output name [%default]')
parser.add_option('--nside', type=int,
                  default=64,
                  help='healpix nside parameter [%default]')
parser.add_option('--vartoplot', type=str,
                  default='nsn',
                  help='var to plot (nsn/nsn_cosmo) [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
norm_factor = opts.norm_factor
runType = opts.runType
config = opts.config
timeslots = opts.timeslots
timescale = opts.timescale
timeslots = get_val(timeslots)
dataType = opts.dataType
plots = opts.plots.split(',')
outDir = opts.outDir
outName = opts.outName
nside = opts.nside
vartoplot = opts.vartoplot

# read config file
conf = pd.read_csv(config, comment='#')

# check outputdir
checkDir(outDir)

dbNames = conf['dbName_WFD'].unique()

wfd = pd.DataFrame()
for dbName in dbNames:
    # check outputdir
    fName = f'{outDir}/{dbName}/{outName}'
    if not os.path.isfile(fName):
        from sn_plotter_analysis.sn_analyser_wdf import process_WFD
        print('file not found', fName)
        # load wfds
        checkDir(f'{outDir}/{dbName}')
        idx = conf['dbName_WFD'] == dbName
        process_WFD(conf[idx], dataType, dbDir, runType,
                    timescale, timeslots, norm_factor, fName)
    wfda = pd.read_hdf(fName)
    wfd = pd.concat((wfd, wfda))

if 'summary' in plots:
    from sn_plotter_analysis.sn_analyser_wdf import plot_summary_wfd
    plot_summary_wfd(wfd, conf, timescale, cumul=True)

if 'mollweid' in plots:
    from sn_plotter_analysis.sn_analyser_wdf import plot_mollview_wfd
    plot_mollview_wfd(wfd, timescale, timeslots, nside,
                      varp=vartoplot, outDir=outDir)

if 'density' in plots:
    from sn_plotter_analysis.sn_analyser_wdf import plot_density_wfd
    plot_density_wfd(wfd, timescale, timeslots, nside, conf,
                     varp=vartoplot, plot_indiv=True)
if 'density_season' in plots:
    from sn_plotter_analysis.sn_analyser_wdf import plot_density_wfd_season
    plot_density_wfd_season(wfd, timescale, timeslots, nside, conf,
                            varp=vartoplot)

plt.show()
