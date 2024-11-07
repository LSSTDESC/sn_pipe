#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:47:09 2024

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
import matplotlib.pyplot as plt

inputDir = 'input/cosmology/host_effi'

files = ['host_effi_desi2_21_v3.csv']

fig, ax = plt.subplots()
for fi in files:
    path = '{}/{}'.format(inputDir, fi)
    df = pd.read_csv(path, comment='#')
    ax.plot(df['z'], df['effi'])

ax.grid(visible='True')
plt.show()
