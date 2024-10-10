#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:10:52 2023

@author: gris
"""
from sn_telmodel import plt, filtercolors
from sn_telmodel.sn_telescope import Zeropoint_airmass
import numpy as np
from optparse import OptionParser
# import os


def plot(res, fitfunc, fitres=None, xvar='airmass', xleg='airmass',
         yvar='zp', yleg='zp [mag]', figtitle=''):
    """
    Function to plot results

    Parameters
    ----------
    res : array
        Data to plot.
    fitres : array, optional
        fit parameters. The default is None.
    xvar : str, optional
        x-axis var. The default is 'airmass'.
    xleg : str, optional
        y-axis label. The default is 'airmass'.
    yvar : str, optional
        y-axis var. The default is 'zp'.
    yleg : str, optional
        y-axis label. The default is 'zp [mag]'.
    figtitle: str, opt
       figure suptitle. The default is ''.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(right=0.85)
    fig.suptitle(figtitle)

    bands = 'ugrizy'
    marker = dict(zip(bands, ['o', 's', '*', '.', 'v', 'h']))
    for b in bands:
        idx = res['band'] == b
        sel = res[idx]

        xdata = sel[xvar]
        ydata = sel[yvar]
        ax.plot(xdata, ydata, label=b,
                marker=marker[b], color=filtercolors[b], mfc='None', ms=15)
        if fitres is not None:
            idxb = fitres['band'] == b
            selfit = fitres[idxb]
            ax.plot(xdata, fitfunc(
                xdata, selfit['slope'], selfit['intercept']),
                color=filtercolors[b])

    ax.set_xlabel('airmass')
    ax.set_ylabel(yleg)
    ax.grid()
    ax.legend(bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)


parser = OptionParser(description='Script to estimate zp vs airmass')

parser.add_option('--telDir', type=str, default='throughputs',
                  help='main throughputs location dir [%default]')
parser.add_option('--throughputsDir', type=str, default='baseline',
                  help='throughputs location dir [%default]')
parser.add_option('--atmosDir', type=str, default='atmos_new',
                  help='atmosphere location dir [%default]')
parser.add_option('--tag', type=str, default='1.9',
                  help='tag versions of the throughputs [%default]')
parser.add_option('--aerosol', type=float, default=0.0,
                  help='aerosol value [%default]')
parser.add_option('--pwv', type=float, default=4.0,
                  help='precipitable water vapor value [%default]')
parser.add_option('--ozone', type=float, default=300.,
                  help='ozone value [%default]')

opts, args = parser.parse_args()

telDir = opts.telDir
throughputsDir = opts.throughputsDir
atmosDir = opts.atmosDir
tag = opts.tag
aerosol = opts.aerosol
pwv = opts.pwv
ozone = opts.ozone
outName = 'zp_airmass_v{}.npy'.format(tag)

zp = Zeropoint_airmass(tel_dir=telDir,
                       through_dir=throughputsDir,
                       atmos_dir=atmosDir, tag=tag,
                       aerosol=aerosol, pwv=pwv, oz=ozone)

res = zp.get_data()
np.save('data_{}'.format(outName), res)

fitres = zp.fit(res)

print(fitres)

print(zp.get_fit_params())
np.save(outName, fitres)

fitres = np.load(outName)
res = np.load('data_{}'.format(outName))

tit = 'v{}'.format(tag)
plot(res, zp.fitfunc, fitres, figtitle=tit)
plot(res, zp.fitfunc, yvar='zp_e_sec',
     yleg='zp [pe/s]', figtitle=tit)
plt.show()
