#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:08:41 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_mean_std(df):

    idx = df['MoM'] > 0.

    df_a = df[idx]
    df_b = df[~idx]

    df_a = calc(df_a, calcCol='MoM')

    print(df_a)

    df_b['sigma_w'] = 100.*np.sqrt(df_b['Cov_w0_w0_fit'])
    df_b = calc(df_b)
    print(df_b)

    return df_a, df_b


def calc(df, grpCol='season', calcCol='sigma_w'):

    var_mean = '{}_mean'.format(calcCol)
    var_std = '{}_std'.format(calcCol)
    df_b = df.groupby([grpCol]).apply(lambda x: pd.DataFrame(
        {var_mean: [x[calcCol].mean()],
         var_std: [x[calcCol].std()]})).reset_index()

    return df_b


data = pd.read_hdf('cosmo_DDF_Univ_WZ.hdf5')

print(data)

idx = data['prior'] == 'noprior'

data_noprior = data[idx]
data_prior = data[~idx]

mom_noprior, sigmaw_noprior = get_mean_std(data_noprior)
mom_prior, sigmaw_prior = get_mean_std(data_prior)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
fig.subplots_adjust(hspace=0., wspace=0.)

ax[0, 0].errorbar(sigmaw_noprior['season'],
                  sigmaw_noprior['sigma_w_mean'],
                  yerr=sigmaw_noprior['sigma_w_std'])
ax[0, 1].errorbar(sigmaw_prior['season'],
                  sigmaw_prior['sigma_w_mean'],
                  yerr=sigmaw_prior['sigma_w_std'])
ax[1, 0].errorbar(mom_noprior['season'],
                  mom_noprior['MoM_mean'],
                  yerr=mom_noprior['MoM_std'])
ax[1, 1].errorbar(mom_prior['season'],
                  mom_prior['MoM_mean'],
                  yerr=mom_prior['MoM_std'])
for i in range(2):
    for j in range(2):
        ax[i, j].grid()
plt.show()
