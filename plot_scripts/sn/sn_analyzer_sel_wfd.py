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
from sn_tools.sn_utils import get_val
import os

parser = OptionParser(description='Script to analyze SN prod after selection')

parser.add_option('--config', type=str,
                  default='input/plots/config_ana.csv',
                  help='OS list[%default]')
parser.add_option('--dbDir', type=str,
                  default='../sn_wfd',
                  help='OS location dir[%default]')
parser.add_option('--timeslots', type=str,
                  default='1-10',
                  help='time slot (season or year) to process [%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='timescale of the files to process [%default]')
parser.add_option('--plots', type=str,
                  default='summary,mollweid,density,density_indiv,density_season',
                  help='plots to draw [%default]')
parser.add_option('--outDir', type=str,
                  default='../sn_wfd',
                  help='output dir [%default]')
parser.add_option('--fName', type=str,
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
config = opts.config
timeslots = opts.timeslots
timeslots = get_val(timeslots)
timescale = opts.timescale
plots = opts.plots.split(',')
outDir = opts.outDir
fName = opts.fName
nside = opts.nside
vartoplot = opts.vartoplot

# read config file
conf = pd.read_csv(config, comment='#')

dbNames = conf['dbName'].unique()

wfd = pd.DataFrame()
for dbName in dbNames:
    # check outputdir
    fNameb = f'{dbDir}/{dbName}/{fName}'
    if not os.path.isfile(fNameb):
        print('missing data', fNameb)
    else:
        wfda = pd.read_hdf(fNameb)
        wfd = pd.concat((wfd, wfda))

if 'summary' in plots:
    from sn_plotter_analysis.sn_analyser_wdf import plot_summary_wfd
    plot_summary_wfd(wfd, conf, timescale, cumul=True)

res = list(filter(lambda x: 'mollweid' in x, plots))
if len(res) > 0:
    from sn_plotter_analysis.sn_analyser_wdf import plot_mollview_wfd
    for_ffmpeg = False
    if 'mollweid_ffmpeg' in res:
        for_ffmpeg = True
    plot_mollview_wfd(wfd, timescale, timeslots, nside,
                      varp=vartoplot, outDir=outDir, for_ffmpeg=for_ffmpeg)

if 'density' in plots:
    from sn_plotter_analysis.sn_analyser_wdf import plot_density_wfd
    plot_indiv = False
    if 'density_indiv' in plots:
        plot_indiv = True
    plot_density_wfd(wfd, timescale, timeslots, nside, conf,
                     varp=vartoplot, plot_indiv=plot_indiv)
if 'density_season' in plots:
    from sn_plotter_analysis.sn_analyser_wdf import plot_density_wfd_season
    plot_density_wfd_season(wfd, timescale, timeslots, nside, conf,
                            varp=vartoplot)

plt.show()
