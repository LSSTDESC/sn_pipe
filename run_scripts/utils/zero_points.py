#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:10:52 2023

@author: gris
"""

from sn_telmodel import plt, filtercolors
from sn_telmodel.sn_telescope import get_telescope
import numpy as np
from optparse import OptionParser
import os


def get_data(tel_dir='throughputs',
             through_dir='throughputs/baseline',
             atmos_dir='throughputs/atmos',
             tag='1.9'):
    """
    Function to estimate zp evolution as a function of airmass

    Returns
    -------
    res : record array
        array with the following cols:
       'airmass', 'band', 'zp', 'zp_adu_sec', 'mean_wavelength']
    """

    r = []
    for airmass in np.arange(1., 2.51, 0.1):
        tel = get_telescope(tel_dir=tel_dir,
                            through_dir=through_dir, atmos_dir=atmos_dir,
                            tag=tag, airmass=airmass, aerosol=True)
        for b in 'ugrizy':
            # b = 'g'
            # print(airmass, b, tel.zp(b))
            mean_wave = tel.mean_wavelength[b]
            rb = [airmass]
            rb.append(b)
            rb.append(tel.zp(b))
            rb.append(tel.counts_zp(b))
            rb.append(mean_wave)
            r.append(rb)

    res = np.rec.fromrecords(
        r, names=['airmass', 'band', 'zp', 'zp_adu_sec', 'mean_wavelength'])

    return res


def func(x, a, b):
    """
    Function used for fitting

    Parameters
    ----------
    x : array(float)
        x-axis var.
    a : float
        slope.
    b : float
        intercept.

    Returns
    -------
    array
        list of values.

    """

    return a*x+b


def fit(res, xvar='airmass', yvar='zp'):
    """
    Function to fit yvar vs xvar for all bands.

    Parameters
    ----------
    res : array
        data to fit.
    xvar : str, optional
        x-axis var. The default is 'airmass'.
    yvar : str, optional
        y-axis var. The default is 'zp'.

    Returns
    -------
    res : array
        slop and intercep from the fit per band.
        added mean_wavelength.

    """

    from scipy.optimize import curve_fit
    r = []
    for b in 'ugrizy':
        idx = res['band'] == b
        sel = res[idx]
        xdata = sel[xvar]
        ydata = sel[yvar]
        popt, pcov = curve_fit(func, xdata, ydata)
        mean_wave = np.mean(sel['mean_wavelength'])
        r.append((b, popt[0], popt[1], mean_wave))

    res = np.rec.fromrecords(
        r, names=['band', 'slope', 'intercept', 'mean_wavelength'])

    return res


def plot(res, fitres=None, xvar='airmass', xleg='airmass',
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
            ax.plot(xdata, func(
                xdata, selfit['slope'], selfit['intercept']),
                color=filtercolors[b])

    ax.set_xlabel('airmass')
    ax.set_ylabel(yleg)
    ax.grid()
    ax.legend(bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)


parser = OptionParser(description='Script to estimate zp vs airmass')

parser.add_option('--telDir', type=str, default='throughputs',
                  help='main throughputs location dir [%default]')
parser.add_option('--throughputsDir', type=str, default='throughputs/baseline',
                  help='throughputs location dir [%default]')
parser.add_option('--atmosDir', type=str, default='throughputs/atmos',
                  help='atmosphere location dir [%default]')
parser.add_option('--tag', type=str, default='1.9',
                  help='tag versions of the throughputs [%default]')

opts, args = parser.parse_args()

telDir = opts.telDir
throughputsDir = opts.throughputsDir
atmosDir = opts.atmosDir
tag = opts.tag


outName = 'zp_airmass_v{}.npy'.format(tag)

if not os.path.isfile(outName):
    res = get_data(tel_dir=telDir,
                   through_dir=throughputsDir,
                   atmos_dir=atmosDir, tag=tag)

    print(res)
    np.save('data_{}'.format(outName), res)

    fitres = fit(res)

    np.save(outName, fitres)

fitres = np.load(outName)
res = np.load('data_{}'.format(outName))

tit = 'v{}'.format(tag)
plot(res, fitres, figtitle=tit)
plot(res, yvar='zp_adu_sec', yleg='zp [ADU/s]', figtitle=tit)
plt.show()
