#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:49:10 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sn_telmodel.sn_throughtools import Sigma_zp_meanwave
from optparse import OptionParser


def get_combi(thedict, parList):
    """
    Function to build a df of combination of parameters

    Parameters
    ----------
    thedict : dict
        input dict with parameters.
    parList : str
        parameter list.

    Returns
    -------
    df : pandas df
        output data: all the possible parameter combinations.

    """

    df = pd.DataFrame()
    for vv in parList:
        dfa = pd.DataFrame(thedict[vv], columns=[vv])
        if len(df) > 0:
            df = df.merge(dfa, how='cross')
        else:
            df = dfa

    return df


parser = OptionParser(
    description='Script to estimate zp and mean_wave from atmos parameters')
parser.add_option('--telDir', type=str, default='throughputs',
                  help='tel main dir [%default]')
parser.add_option('--tag', type=str, default='1.9',
                  help='tag version of the throughputs [%default]')
parser.add_option('--throughDir', type=str, default='baseline',
                  help='throughput dir [%default]')
parser.add_option('--airmass', type=float, default=1.2,
                  help='airmass value [%default]')
parser.add_option('--sigma_airmass', type=float, default=0.01,
                  help='airmass sigma value [%default]')
parser.add_option('--pwv', type=float, default=4.0,
                  help='airmass value [%default]')
parser.add_option('--sigma_pwv', type=float, default=0.2,
                  help='pwv sigma value [%default]')
parser.add_option('--ozone', type=float, default=300,
                  help='ozone value [%default]')
parser.add_option('--sigma_ozone', type=float, default=10.,
                  help='ozone sigma value [%default]')
parser.add_option('--aerosol', type=float, default=0.1,
                  help='airmass value [%default]')
parser.add_option('--sigma_aerosol', type=float, default=0.001,
                  help='aerosol sigma value [%default]')
parser.add_option('--beta', type=float, default=0.2,
                  help='beta value [%default]')
parser.add_option('--sigma_beta', type=float, default=0.00,
                  help='beta sigma value [%default]')

opts, args = parser.parse_args()

params = vars(opts)

time_ref = time.time()
tel_dir = params['telDir']
through_dir = params['throughDir']
tag = params['tag']

tel_dir = '{}_{}'.format(tel_dir, tag)
through_dir = '{}/{}'.format(tel_dir, through_dir)

site_name = 'LSST'
pressure = 743.
bands = 'ugrizy'

par_names = ['airmass', 'pwv', 'ozone', 'beta', 'aerosol']

par_means = []
par_sigmas = []
for vv in par_names:
    par_means.append(params[vv])
    par_sigmas.append(params['sigma_{}'.format(vv)])

sigma_zp = Sigma_zp_meanwave(through_dir, site_name, pressure,
                             par_names, par_means, par_sigmas,
                             save_throughputs_dir='ty_through')

res = sigma_zp(ntrials=1000, nproc=8)

print('finally', time.time()-time_ref)
print('end of processing', time.time()-time_ref)

print(res[['std_zp_y', 'std_zp_z', 'std_mean_wave_y', 'std_mean_wave_z']])

"""
for b in bands:
    fig, ax = plt.subplots(figsize=(12, 9), ncols=2)
    ax[0].hist(df['std_zp_{}'.format(b)], histtype='step', bins=200)
    ax[1].hist(df['std_mean_wave_{}'.format(b)], histtype='step', bins=200)
    ax[0].set_xlabel(r'$\Delta z_p$')
    ax[1].set_xlabel(r'$\Delta \bar{\lambda}$')
    ax[0].set_ylabel(r'Number of Entries')
    ax[1].set_ylabel(r'Number of Entries')
"""

plt.show()
