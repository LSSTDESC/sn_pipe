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
from sn_tools.sn_utils import get_val, multiproc
import os
from sn_plotter_analysis.sn_analyser_tools import print_nsn_latex


def cumul_multiproc(df):
    """
    Estimate sum(nsn,nsn_cosmo) per pixel

    Parameters
    ----------
    df : pandas df
        Data to process.

    Returns
    -------
    res : pandas df
        result.

    """

    hpixes = df['healpixID'].unique()

    params = {}

    params['data'] = df
    params['timescale'] = timescale

    res = multiproc(hpixes, params, cumul_set_pixels, nproc=8)

    return res


def cumul_set_pixels(hpixes, params, j, output_q=None):
    """
    Process to estimate sum(nsn) for a set of pixels

    Parameters
    ----------
    hpixes : list(int)
        list of pixels to consider.
    params : dict
        parameters for the processing.
    j : int
        internal tag for multiprocessing.
    output_q : multiprocessing queue, optional
        where to pu the result. The default is None.

    Returns
    -------
    pandas df
        output data.

    """

    timescale = params['timescale']
    data = params['data']

    idx = data['healpixID'].isin(hpixes)
    data = pd.DataFrame(data[idx])
    res = data.groupby(['dbName', 'healpixID']).apply(
        lambda x: cumul_pixel(x, timescale=timescale), include_groups=False).reset_index()

    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


def cumul_pixel(grp, cols=['nsn', 'nsn_cosmo'], timescale='year'):
    """
    Functio to estimate sum(nsn) for a pixel

    Parameters
    ----------
    grp : pandas df
        Data to process.
    cols : list(str), optional
        df columns to consider. The default is ['nsn', 'nsn_cosmo'].
    timescale : str, optional
        Timescale for the processing (year/season). The default is 'year'.

    Returns
    -------
    df : pandas df
        output data.

    """

    df = pd.DataFrame()
    for i in range(1, 11):
        idx = grp[timescale] <= i
        sel = grp[idx]
        dd = dict(zip(cols, [0, 0]))
        if len(sel) > 0:
            for cc in cols:
                dd[cc] = [sel[cc].sum()]
        dd[timescale] = [i]
        dfa = pd.DataFrame.from_dict(dd)
        df = pd.concat((df, dfa))

    return df


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
                  default='summary,mollweid,mollweid_ffmpeg,density,density_indiv,density_season',
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
parser.add_option('--OS_ref', type=str,
                  default='None',
                  help='ref OS for normalization [%default]')
parser.add_option('--print_nsn', type=int,
                  default=0,
                  help='to print nsn as a latex table [%default]')
parser.add_option('--cumul', type=int,
                  default=0,
                  help='to plot sum(nsn) [%default]')
parser.add_option('--savepng', type=int,
                  default=1,
                  help='to save the plot (Mollview) [%default]')

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
OS_ref = opts.OS_ref
print_nsn = opts.print_nsn
cumul = opts.cumul
savepng = opts.savepng

# read config file
conf = pd.read_csv(config, comment='#')

dbNames = conf['dbName'].unique()

dict_leg = dict(zip(['nsn', 'nsn_cosmo'], ['N$_{SN}$', 'N$_{SN}^{COSMO}$']))

wfd = pd.DataFrame()
for dbName in dbNames:
    # check outputdir
    fNameb = f'{dbDir}/{dbName}/{fName}'
    if not os.path.isfile(fNameb):
        print('missing data', fNameb)
    else:
        wfda = pd.read_hdf(fNameb)
        wfd = pd.concat((wfd, wfda))

if print_nsn:
    print_nsn_latex(wfd)
    print_nsn_latex(wfd, nsn_var='nsn_cosmo', err_nsn_var='err_nsn_cosmo')

if 'summary' in plots:
    from sn_plotter_analysis.sn_analyser_wdf import plot_summary_wfd
    from sn_plotter_analysis.sn_analyser_wdf import plot_summary_wfd_norm
    plot_summary_wfd(wfd, conf, timescale, cumul=True)
    plot_summary_wfd_norm(wfd, conf, timescale, cumul=True)

    if OS_ref != 'None':
        plot_summary_wfd_norm(wfd, conf, timescale,
                              yvar_err='None', cumul=True, dbNorm=OS_ref)


res = list(filter(lambda x: 'mollweid' in x, plots))
if len(res) > 0:
    from sn_plotter_analysis.sn_analyser_wdf import plot_mollview_wfd
    for_ffmpeg = False
    if 'mollweid_ffmpeg' in res:
        for_ffmpeg = True

    pp = dict_leg[vartoplot]
    pp = '{}='.format(pp)
    if cumul:
        wfd = cumul_multiproc(wfd)
        pp = '$\Sigma$'+pp
    plot_mollview_wfd(wfd, timescale, timeslots, nside,
                      varp=vartoplot, varleg=pp,
                      outDir=outDir, for_ffmpeg=for_ffmpeg, savepng=savepng)

if 'density' in plots:
    from sn_plotter_analysis.sn_analyser_wdf import plot_density_wfd
    plot_indiv = False
    plot_density_wfd(wfd, timescale, timeslots, nside, conf,
                     varp=vartoplot, plot_indiv=plot_indiv)
if 'density_indiv' in plots:
    from sn_plotter_analysis.sn_analyser_wdf import plot_density_wfd
    plot_indiv = True
    plot_density_wfd(wfd, timescale, timeslots, nside, conf,
                     varp=vartoplot, plot_indiv=plot_indiv)

if 'density_season' in plots:
    from sn_plotter_analysis.sn_analyser_wdf import plot_density_wfd_season
    plot_density_wfd_season(wfd, timescale, timeslots, nside, conf,
                            varp=vartoplot)

plt.show()
