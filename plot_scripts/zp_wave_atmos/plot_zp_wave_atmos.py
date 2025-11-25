#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 13:38:56 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
import glob
from optparse import OptionParser
from sn_tools.sn_utils import multiproc
from sn_plotter_analysis import plt
import warnings
warnings.simplefilter("ignore", category=SyntaxWarning)


def process(fName):
    """
    Function to load and process a file

    Parameters
    ----------
    fName : str
        File name to process.

    Returns
    -------
    dfa: pandas df
       process data
    """

    # load the data
    df = pd.read_hdf(fName)

    df = df.round({'mean_airmass': 1})
    """
    print(df.columns)
    print(df['mean_airmass'].unique())
    """
    dfa = df.groupby(['mean_airmass']).apply(
        lambda x: stat(x), include_groups=False).reset_index()

    return dfa


def stat(grp):
    """
    Function to estimate some stats

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    dft : pandas df
        processed data.

    """

    str_ch = ['zp', 'mean_wave']
    bands = 'ugrizy'

    ccols = []

    for vva in str_ch:
        for b in bands:
            vvb = 'std_{}_{}'.format(vva, b)
            ccols.append(vvb)

    dfa = grp[ccols].mean().to_frame().T
    dfa.columns = dfa.columns.str.replace("std", "delta", regex=True)

    dfb = grp[ccols].std().to_frame().T
    dfb.columns = dfb.columns.str.replace("std", "std_delta", regex=True)

    dft = dfa.merge(dfb, how='cross')

    lparams = ['ozone', 'pwv', 'aerosol', 'beta']
    ccolsc = []
    for pps in ['mean', 'sigma']:
        for pp in lparams:
            ccolsc.append('{}_{}'.format(pps, pp))

    df_atm_param = grp[ccolsc].mean().to_frame().T

    dft = dft.merge(df_atm_param, how='cross')

    return dft


def process_multi(toproc, params, j=0, output_q=None):
    """
    Function to process data using multiprocessing

    Parameters
    ----------
    toproc : list(str)
        List of files to process.
    params : dict
        Parameter dict.
    j : int, optional
        tag for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        where to put the result. The default is None.

    Returns
    -------
    pandas df
        Output data.

    """

    res = pd.DataFrame()

    for pp in toproc:
        df = process(pp)
        res = pd.concat((df, res))

    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


def plot_this(dft, varx='mean_airmass', xlabel='airmass',
              vary='delta_zp_y', ylabel='$\Delta zp_y$',
              varz='sigma_aerosol', zlabel='$\sigma_{aerosol}$', figtitle=''):
    """
    Function to plot results

    Parameters
    ----------
    dft : pandas df
        Data to plot.
    varx : str, optional
        x-axis var. The default is 'mean_airmass'.
    xlabel : str, optional
        x-axis label. The default is 'airmass'.
    vary : str, optional
        y-axis var. The default is 'delta_zp_y'.
    ylabel : str, optional
        y-axis label. The default is '$\Delta zp_y$'.       
    varz : str, optional
        z-axis (third dim) var. The default is 'sigma_aerosol'.
    zlabel : str, optional
        z-axis label. The default is '$\sigma_{aerosol}$'.
    figtitle: str, optional.
      Figure title. The default is ''

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(right=0.75)
    if figtitle != '':
        fig.suptitle(figtitle)

    print(dft.columns)

    dft = dft.round({varx: 5, varz: 5})
    print(dft[varz].unique())

    dft = dft.sort_values(by=[varz])
    sigmas = dft[varz].unique().tolist()
    markers = ['+', 'x', 'X', 's', 'P', '1', 'o']
    colors = ['yellow', 'orange', 'violet',
              'cyan', 'red', 'green', 'lightgrey']
    mm = dict(zip(sigmas, markers))
    ccolors = dict(zip(sigmas, colors))

    print(ccolors)
    for sig in sigmas:
        idx = dft[varz] == sig
        sel = dft[idx]
        sel_m = sel.groupby([varx])[vary].max().reset_index()
        sel_p = sel.groupby([varx])[vary].min().reset_index()

        ax.fill_between(sel_m[varx], sel_m[vary],
                        sel_p[vary], color=ccolors[sig], label='{}={}'.format(zlabel, sig))
        """
        ax.plot(sel[varx], sel[vary],
                marker=mm[sig], color='k', linestyle='None', label='{}'.format(sig))
        """
    ax.legend(loc='upper right',
              bbox_to_anchor=(1.4, 0.9), fontsize=15,
              frameon=False)
    ax.grid(visible=True)
    ax.set_xlabel(r'{}'.format(xlabel))
    ax.set_ylabel(r'{}'.format(ylabel))

    xmin, xmax = dft[varx].min(), dft[varx].max()
    ax.set_xlim(xmin, xmax)


parser = OptionParser(
    description='Script to draw zp_sigma and mean_wave_eff vs atmos param errors')

parser.add_option('--fileDir', type=str, default='../zp_atmos',
                  help='file dir [%default]')

opts, args = parser.parse_args()

fileDir = opts.fileDir

fis = glob.glob('{}/*.hdf5'.format(fileDir))

params = {}
dft = multiproc(fis, params, process_multi, nproc=8)

print(dft)

# plot_this(dft)


idx = dft['mean_airmass'] > 1.15
idx &= dft['mean_airmass'] < 1.25
sel = dft[idx]

for b in 'ugrizy':
    vary = 'delta_zp_'+b
    ylabel = '$\Delta zp_'+b+'$'
    print(vary, ylabel)
    plot_this(sel,
              varx='sigma_pwv', xlabel='$\sigma_{PWV}$',
              vary=vary, ylabel=ylabel,
              figtitle='airmass=1.2')


plt.show()
