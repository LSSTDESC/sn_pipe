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


time_ref = time.time()
tel_dir = 'throughputs'
through_dir = 'baseline'
tag = '1.9'

tel_dir = '{}_{}'.format(tel_dir, tag)
through_dir = '{}/{}'.format(tel_dir, through_dir)

site_name = 'LSST'
pressure = 743.
bands = 'ugrizy'

par_names = ['airmass', 'pwv', 'ozone', 'beta', 'aerosol']
par_means = [1.2, 4.0, 300., 0.1, 0.1]
par_sigmas = [0.01, 0.2, 10., 0.0, 0.001]

dict_sigma = {}
dict_sigma['airmass'] = list(np.arange(0.01, 0.11, 0.01))
dict_sigma['pwv'] = list(np.arange(0.01, 0.5, 0.01))
dict_sigma['ozone'] = list(np.arange(8., 21, 1.))
dict_sigma['aerosol'] = list(np.arange(0.01, 0.06, 0.01))
dict_sigma['beta'] = [0.]

dict_sigma['airmass'] = [0.]
dict_sigma['pwv'] = [0.2]
dict_sigma['ozone'] = [0.]
dict_sigma['aerosol'] = [0.]
dict_sigma['beta'] = [0.]

combi_sigma = get_combi(dict_sigma, par_names)

print(combi_sigma)

df = pd.DataFrame()

print('nb de combinaisons', len(combi_sigma), combi_sigma.columns)
time_ref = time.time()
for i, row in combi_sigma.iterrows():
    par_sigmas = row[par_names].to_list()
    sigma_zp = Sigma_zp_meanwave(through_dir, site_name, pressure,
                                 par_names, par_means, par_sigmas,
                                 save_throughputs_dir='ty_through')

    res = sigma_zp(ntrials=100, nproc=1)

    df = pd.concat((df, res))


print('finally', time.time()-time_ref)
print('end of processing', time.time()-time_ref)

print(df[['std_zp_y', 'std_zp_z', 'std_mean_wave_y', 'std_mean_wave_z']])

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
