#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:29:32 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from rubin_sim.phot_utils import Bandpass
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser

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


def load_atmos(fDir, airmass):
    """
    Function to load atmospheric transparency files

    Parameters
    ----------
    fDir : str
        Data dir.
    airmass : float
        airmass value.

    Returns
    -------
    vv : Bandpass
        trans (wave, sb).

    """

    full_name = '{}/atmos_{}.dat'.format(fDir, int(10*airm))
    vv = Bandpass()
    vv.read_throughput(full_name)

    return vv


def plot_trans(trans, airmass_ref=-1, legx='wavelength [nm]', legy='Sb [0-1]'):
    """
    Function to plot transmission

    Parameters
    ----------
    trans : dict
        Trans data.
    airmass_ref : float, optional
        ref airmass to normalize trans. The default is -1.
    legx : str, optional
        x-axis legend. The default is 'wavelength [nm]'.
    legy : str, optional
        y-axis legend. The default is 'Sb [0-1]'.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(12, 9))

    trans_ref = 1
    if airmass_ref > 0:
        trans_ref = trans[int(10*airmass_ref)].sb

    for key, vv in trans.items():
        kk = np.round(key/10, 1)
        if kk == airmass_ref:
            continue
        ax.plot(vv.wavelen, vv.sb/trans_ref,
                linestyle=lss[kk], marker=mms[kk],
                color='k',
                markevery=100, label='airmass: {}'.format(kk), mfc='None')
    ax.grid()
    ax.legend()
    ax.set_ylim([0., None])
    ax.set_xlim([290., 1200])
    ax.set_xlabel(r'{}'.format(legx))
    ax.set_ylabel(r'{}'.format(legy))


parser = OptionParser(
    description='Script to estimate atmospheric transparency')

parser.add_option('--atmos_dir', type=str,
                  default='throughputs_1.9/atmos',
                  help='atmos dir for files[%default]')
parser.add_option('--airmass', type=str,
                  default='1.2,1.5,2.0,2.2,2.5',
                  help='airmass values [%default]')

opts, args = parser.parse_args()

atmos_dir = opts.atmos_dir
airmass = list(map(float, opts.airmass.split(',')))

print(airmass)

ls = ['solid', 'dotted', 'dashed', 'dashdot']*2
marker = ['o', 's', 'v', '^', '<', '>', 'h', '.']

n = len(airmass)

lss = dict(zip(airmass, ls[:n]))
mms = dict(zip(airmass, marker[:n]))

trans = {}
for airm in airmass:
    trans[int(10*airm)] = load_atmos(atmos_dir, airm)

plot_trans(trans)

airmass_ref = 1.2
plot_trans(trans, airmass_ref=airmass_ref,
           legy='$\\frac{Sb^{airmass}}{Sb^{airmass=1.2}}$')


plt.show()
