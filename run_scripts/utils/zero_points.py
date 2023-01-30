#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:10:52 2023

@author: gris
"""

import matplotlib.pyplot as plt
from sn_telmodel.sn_telescope import Telescope
import numpy as np


def get_data():
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
        tel = Telescope(airmass=airmass, aerosol=True)
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
         yvar='zp', yleg='zp [mag]'):
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

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(12, 8))

    for b in 'ugrizy':
        idx = res['band'] == b
        sel = res[idx]

        xdata = sel[xvar]
        ydata = sel[yvar]
        ax.plot(xdata, ydata, 'k.')
        if fitres is not None:
            idxb = fitres['band'] == b
            selfit = fitres[idxb]
            ax.plot(xdata, func(
                xdata, selfit['slope'], selfit['intercept']), 'r-')

    ax.set_xlabel('airmass')
    ax.set_ylabel(yleg)
    ax.grid()


res = get_data()

print(res)

fitres = fit(res)

np.save('zp_airmass.npy', fitres)

plot(res, fitres)
plot(res, yvar='zp_adu_sec', yleg='zp-> ADU/s')
plt.show()
