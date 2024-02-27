#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:46:59 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
from sn_analysis.sn_tools import load_cosmo_data


theDir = '../cosmo_fit_WFD_TiDES'
theDir = '../cosmo_fit_WFD_sigmaC_test_lowz'

dbName = 'baseline_v3.0_10yrs'
dbName = 'DDF_DESC_0.75_co_0.07_1_10'

timescale = 'season'

spectro_config = theDir.split('cosmo_fit_WFD_')[-1]

df = load_cosmo_data(theDir, dbName, timescale, spectro_config)

print(df)

fig, ax = plt.subplots()

ax.errorbar(df[timescale], df['MoM_mean'], yerr=df['MoM_std'])

plt.show()
