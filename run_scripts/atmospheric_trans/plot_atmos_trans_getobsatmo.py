#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:10:32 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from getObsAtmo.getObsAtmo import ObsAtmo
import numpy as np
from rubin_sim.phot_utils import Bandpass
from scipy.interpolate import interp1d
from sn_telmodel.sn_transtools import get_trans
import matplotlib.pyplot as plt

filtercolors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20


def plot_trans(atm_params=dict(zip(['am', 'pwv', 'oz', 'tau', 'beta'],
                                   [[1.2], [4.0], [300], [0.1], [1.4]])),
               fig=None, ax=None):
    """
    Function to plot a set of throughputs according to the atmos params

    Parameters
    ----------
    atm_params : dict, optional
        Atmospheric parameters. 
        The default is 
        dict(zip(['am', 'pwv', 'oz', 'tau', 'beta'],
                 [[1.2], [4.0], [300], [0.1], [1.4]])).
    fig : matplotlib figure, optional
        Figure fior the plot. The default is None.
    ax : matplotlib axis, optional
        Axis for the plot. The default is None.

    Returns
    -------
    None.

    """

    trans = {}

    for am in atm_params['atm']:
        for pwv in atm_params['pwv']:
            for oz in atm_params['oz']:
                for tau in atm_params['tau']:
                    for beta in atm_params['beta']:
                        leg = '({},{},{},{},{})'.format(
                            am, pwv, oz, tau, beta)
                        trans[leg] = get_trans(
                            am, pwv, oz, tau, beta, colname=['wl', 'trans'])

    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    for key, vals in trans.items():
        ax.plot(vals['wl'], vals['trans'], label=key)

    ax.legend()

    ax.set_xlabel(r'Wavelength(nm)')
    ax.set_ylabel(r'Throughput(0-1)')
    ax.grid(visible=True)


def compare_spectra(spa, spb):
    """
    Function to compare spectra

    Parameters
    ----------
    spa : rubin obs bandpass
        First spectrum.
    spb : Trubin obs bandpass
        Second spectrum.

    Returns
    -------
    None.

    """

    wave_min = np.min([np.min(spa.wavelen), np.min(spb.wavelen)])
    wave_max = np.min([np.max(spa.wavelen), np.max(spb.wavelen)])

    waves = np.arange(wave_min, wave_max, 0.1)
    spa_interp = interp1d(spa.wavelen, spa.sb,
                          bounds_error=False, fill_value=0.)
    spb_interp = interp1d(spb.wavelen, spb.sb,
                          bounds_error=False, fill_value=0.)

    transa = spa_interp(waves)
    transb = spb_interp(waves)

    fig, ax = plt.subplots()

    ax.plot(waves, transa/transb)


cols = ['atm', 'pwv', 'oz', 'tau', 'beta']
vals = [[1.5], [4.0], [300], [0.04], [1.4]]

fig, ax = plt.subplots(figsize=(12, 8))
plot_trans(dict(zip(cols, vals)), fig=fig, ax=ax)

fName = 'throughputs_1.9/atmos/atmos_15_aerosol.dat'
vva = Bandpass()
vva.read_throughput(fName)
ax.plot(vva.wavelen, vva.sb, color='k')


plt.show()
