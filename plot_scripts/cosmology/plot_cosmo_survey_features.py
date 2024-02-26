#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:46:59 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt


def load_data(theDir, dbName, timescale, spectro_config):

    fName = '{}/cosmo_{}*.hdf5'.format(theDir, dbName)
    fis = glob.glob(fName)

    df = pd.DataFrame()

    for fi in fis:
        dd = pd.read_hdf(fi)
        df = pd.concat((df, dd))

    print(df.columns)
    dfb = df.groupby([timescale])['WFD_TiDES',
                                  'all_Fields'].mean().reset_index()

    dfb['dbName'] = dbName
    dfb['spectro_config'] = spectro_config

    return dfb


theDir = '../cosmo_fit_WFD_TiDES'

dbName = 'baseline_v3.0_10yrs'

timescale = 'year'

spectro_config = theDir.split('cosmo_fit_WFD_')[-1]

df = load_data(theDir, dbName, timescale, spectro_config)

print(df)

fig, ax = plt.subplots()

ax.plot(df[timescale], df['all_Fields'])

plt.show()
